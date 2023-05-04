import torch
from lightly.data import SimCLRCollateFunction, SwaVCollateFunction
from torch.autograd import Variable

from .lisa import lisa
from .mixup import mixup_data, mixup_criterion

group_datasets = ['coral', 'groupdro', 'irm']

def prepare_data(x, y, dataset_name: str):
    if dataset_name == 'drug':
        x[0] = x[0].cuda()
        x[1] = x[1].cuda()
        y = y.cuda()
    elif 'mimic' in dataset_name:
        y = torch.cat(y).type(torch.LongTensor).cuda()
    elif dataset_name in ['arxiv', 'huffpost']:
        x = x.to(dtype=torch.int64).cuda()
        if len(y.shape) > 1:
            y = y.squeeze(1).cuda()
    elif dataset_name in ['fmow', 'yearbook']:
        if isinstance(x, tuple) or isinstance(x, list):
            x = list(elt.cuda() for elt in x)
        else:
            x = x.cuda()
        if len(y.shape) > 1:
            y = y.squeeze(1).cuda()
    return x, y


def reinit_dataset(args):
    if args.dataset in ['huffpost']:
        if args.method in group_datasets:
            from ..data.huffpost import HuffPostGroup
            dataset = HuffPostGroup(args)
        else:
            from ..data.huffpost import HuffPost
            dataset = HuffPost(args)
    elif args.dataset in ['yearbook']:
        if args.method in group_datasets:
            from ..data.yearbook import YearbookGroup
            dataset = YearbookGroup(args)
        else:
            from ..data.yearbook import Yearbook
            dataset = Yearbook(args)
    elif args.dataset in ['fmow']:
        if args.method in group_datasets:
            from ..data.fmow import FMoWGroup
            dataset = FMoWGroup(args)
        else:
            from ..data.fmow import FMoW
            dataset = FMoW(args)
    elif args.dataset in ['drug']:
        if args.method in group_datasets:
            from ..data.drug import TdcDtiDgGroup
            dataset = TdcDtiDgGroup(args)
        else:
            from ..data.drug import TdcDtiDg
            dataset = TdcDtiDg(args)
    elif 'mimic' in args.dataset:
        if args.method in group_datasets:
            from ..data.mimic import MIMICGroup
            dataset = MIMICGroup(args)
        else:
            from ..data.mimic import MIMIC
            dataset = MIMIC(args)
    elif args.dataset in ['arxiv']:
        if args.method in group_datasets:
            from ..data.arxiv import ArXivGroup
            dataset = ArXivGroup(args)
        else:
            from ..data.arxiv import ArXiv
            dataset = ArXiv(args)
    return dataset

def forward_pass(x, y, dataset, network, criterion, use_lisa: bool, use_mixup: bool, cut_mix: bool, mix_alpha=2.0):
    if use_lisa:
        if str(dataset) in ['arxiv', 'huffpost']:
            x = network.model[0](x)
            sel_x, sel_y = lisa(x, y, dataset=dataset, mix_alpha=mix_alpha,
                                num_classes=dataset.num_classes, time_idx=dataset.current_time,
                                cut_mix=cut_mix, embedding=network.model[0])
            logits = network.model[1](sel_x)
        elif str(dataset) in ['drug']:
            sel_x0, sel_y = lisa(x[0], y, dataset=dataset, mix_alpha=mix_alpha,
                                num_classes=dataset.num_classes, time_idx=dataset.current_time,
                                cut_mix=cut_mix)
            sel_x1, sel_y = lisa(x[1], y, dataset=dataset, mix_alpha=mix_alpha,
                                num_classes=dataset.num_classes, time_idx=dataset.current_time,
                                cut_mix=cut_mix)
            sel_x = [sel_x0, sel_x1]
            logits = network(sel_x)
        elif 'mimic' in str(dataset):
            x = network.get_cls_embed(x)
            sel_x, sel_y = lisa(x, y, dataset=dataset, mix_alpha=mix_alpha,
                                num_classes=dataset.num_classes, time_idx=dataset.current_time,
                                cut_mix=cut_mix, embedding=network.get_cls_embed)
            logits = network.fc(sel_x)
        else:
            sel_x, sel_y = lisa(x, y, dataset=dataset, mix_alpha=mix_alpha,
                                num_classes=dataset.num_classes, time_idx=dataset.current_time,
                                cut_mix=cut_mix)
            logits = network(sel_x)
        y = torch.argmax(sel_y, dim=1)
        loss = criterion(logits, y)

    elif use_mixup:
        if str(dataset) in ['arxiv', 'huffpost']:
            x = network.model[0](x)
            x, y_a, y_b, lam = mixup_data(x, y, mix_alpha=mix_alpha)
            logits = network.model[1](x)
        elif 'mimic' in str(dataset):
            x = network.get_cls_embed(x)
            x, y_a, y_b, lam = mixup_data(x, y, mix_alpha=mix_alpha)
            logits = network.fc(x)
        elif str(dataset) in ['drug']:
            x0, y_a, y_b, lam = mixup_data(x[0], y, mix_alpha=mix_alpha)
            x1, y_a, y_b, lam = mixup_data(x[1], y, mix_alpha=mix_alpha)
            x = [x0, x1]
            y_a = y_a.float()
            y_b = y_b.float()
            logits = network(x).squeeze(1).float()
        else:
            x, y_a, y_b, lam = mixup_data(x, y, mix_alpha=mix_alpha)
            x, y_a, y_b = map(Variable, (x, y_a, y_b))
            logits = network(x)
        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

    else:
        logits = network(x)
        if str(dataset) in ['drug']:
            logits = logits.squeeze().double()
            y = y.squeeze().double()
        elif str(dataset) in ['arxiv', 'fmow', 'huffpost', 'yearbook']:
            if len(y.shape) > 1:
                y = y.squeeze(1)
        loss = criterion(logits, y)

    return loss, logits, y


def split_into_groups(g):
    """
    From https://github.com/p-lambda/wilds/blob/f384c21c67ee58ab527d8868f6197e67c24764d4/wilds/common/utils.py#L40.
    Args:
        - g (Tensor): Vector of groups
    Returns:
        - groups (Tensor): Unique groups present in g
        - group_indices (list): List of Tensors, where the i-th tensor is the indices of the
                                elements of g that equal groups[i].
                                Has the same length as len(groups).
        - unique_counts (Tensor): Counts of each element in groups.
                                 Has the same length as len(groups).
    """
    unique_groups, unique_counts = torch.unique(g, sorted=False, return_counts=True)
    group_indices = []
    for group in unique_groups:
        group_indices.append(
            torch.nonzero(g == group, as_tuple=True)[0])
    return unique_groups, group_indices, unique_counts

def get_collate_functions(args, train_dataset):
    if 'mimic' in args.dataset:
        train_collate_fn = collate_fn_mimic
        eval_collate_fn = collate_fn_mimic
    elif args.method == 'simclr':
        if args.dataset == 'yearbook':
            train_collate_fn = SimCLRCollateFunction(
                input_size=train_dataset.resolution,
                vf_prob=0.5,
                rr_prob=0.5
            )
        else:
            train_collate_fn = SimCLRCollateFunction(
                input_size=train_dataset.resolution
            )
        eval_collate_fn = None
    elif args.method == 'swav':
        train_collate_fn = SwaVCollateFunction()
        eval_collate_fn = None
    else:
        train_collate_fn = None
        eval_collate_fn = None

    return train_collate_fn, eval_collate_fn

def collate_fn_mimic(batch):
    codes = [item[0][0] for item in batch]
    types = [item[0][1] for item in batch]
    target = [item[1] for item in batch]
    if len(batch[0]) == 2:
        return [(codes, types), target]
    else:
        groupid = torch.cat([item[2] for item in batch], dim=0).unsqueeze(1)
        return [(codes, types), target, groupid]

def flatten_parameters(model):
    """Returns a flattened tensor containing the parameters of model."""
    return torch.cat([param.flatten() for param in model.module.parameters()])


def assign_params(model, w):
    """Takes in a flattened parameter vector w and assigns them to the parameters
    of model.
    """
    offset = 0
    for parameter in model.module.parameters():
        param_size = parameter.nelement()
        parameter.data = w[offset : offset + param_size].reshape(parameter.shape)
        offset += param_size


def flatten_gradients(model):
    """Returns a flattened numpy array with the gradients of the parameters of
    the model.
    """
    return np.concatenate(
        [
            param.grad.detach().cpu().numpy().flatten()
            if param.grad is not None
            else np.zeros(param.nelement())
            for param in model.parameters()
        ]
    )

def create_eval_fn(calculate_gradient=False):
    def eval_fn(model, dataloader, device):
        model.eval()
        total_loss = 0
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum").to(device=device)
        num_correct = 0
        num_items = 0
        model.zero_grad()
        torch.set_grad_enabled(calculate_gradient)
        for idx, (X, y) in enumerate(iter(dataloader)):
            X = X.to(device=device)
            y = y.to(device=device).squeeze()
            output = model(X)
            preds = torch.argmax(output, dim=1)
            num_correct += (preds == y).sum().item()
            try:
                num_items += y.shape[0]
            except:
                import ipdb; ipdb.set_trace()

            loss = loss_fn(output, y)
            if calculate_gradient:
                loss.backward()
            total_loss += loss.item()

        accuracy = num_correct / num_items
        avg_loss = total_loss / num_items
        metrics = {"loss": avg_loss, "accuracy": accuracy}
        if calculate_gradient:
            gradients = flatten_gradients(model)
            metrics["gradients"] = gradients

        return metrics

    return eval_fn
