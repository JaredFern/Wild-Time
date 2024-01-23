# %%
from wildtime import baseline_trainer, dataloader
from wildtime.configs.eval_fix import configs_mimic_mortality, configs_mimic_readmission, configs_fmow, configs_arxiv, configs_huffpost, configs_yearbook

import numpy as np
from scipy.special import softmax

import argparse
import os
import logging
import pickle
import resource
from copy import deepcopy
from collections import Counter
from itertools import product

from wildtime.methods.dataloaders import FastDataLoader

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import torch.multiprocessing
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

torch.multiprocessing.set_sharing_strategy('file_system')


def _reinit_swa_model(trainer):
    model_path = os.path.join(trainer.args.results_dir, trainer.args.exp_path, 'checkpoints')
    if os.path.exists(os.path.join(model_path, 'time_offline_swa.pth')):
        init_weights = torch.load(os.path.join(model_path, f'time_offline_swa.pth'))
        trainer.swa_model.load_state_dict(init_weights, strict=False)
        trainer.swa_model.avg_fn = trainer.ema_avg
    elif os.path.exists(os.path.join(model_path, 'time_offline.pth')):
        init_weights = torch.load(os.path.join(model_path, f'time_offline.pth'))
        trainer.network.load_state_dict(init_weights, strict=False)
        trainer.swa_model = AveragedModel(trainer.network, avg_fn=trainer.ema_avg)
    return trainer

def get_predictions(trainer, checkpoints, checkpoint_dir):
    metrics = []
    preds = []
    labels = []
    eval_splits = [i for i in trainer.eval_dataset.ENV if i > trainer.split_time]

    for split in eval_splits:
        split_preds = []
        split_labels = []

        trainer = _reinit_swa_model(trainer)
        for ckpt in checkpoints:
            ckpt_path = os.path.join(checkpoint_dir, 'checkpoints', ckpt)
            if trainer.args.method == "swa":
                tmp_network = deepcopy(trainer.network)
                tmp_network.load_state_dict(torch.load(ckpt_path))
                trainer.swa_model.update_parameters(tmp_network)
                trainer.eval_dataset.update_current_timestamp(split)
                bn_dataloader = FastDataLoader(
                    dataset=trainer.eval_dataset, 
                    batch_size=trainer.eval_batch_size,
                    num_workers=trainer.num_workers,
                    collate_fn=trainer.eval_collate_fn)
                update_bn(bn_dataloader, trainer.swa_model, device=trainer.args.device)
            else:
                trainer.load_model(checkpoint_path=ckpt_path)
            acc, pred, label = trainer.run_eval_timestamp(split, mode=1)

            split_preds.append(pred)
            split_labels.append(label)
            print(f"Accuracy of FT until {ckpt} on {split}: {acc}")

        preds.append(np.array([np.vstack(_) for _ in split_preds]))
        labels.append(np.array(split_labels))

    return metrics, preds, labels


def save_predictions_and_metadata(fpath, pred_probs, pred_labels, events, labels, model, eval_checkpoints,):
    transferred = [i[0] for i in events]
    hard = [i[1] for i in events]
    learned = [i[2] for i in events]
    forgotten = [i[3] for i in events]

    data = dict({
        "model": model,
        "eval_checkpoints": eval_checkpoints,
        "pred_probs": pred_probs,
        "pred_labels": pred_labels,
        "labels": labels,
        "transferred": transferred,
        "hard": hard,
        "learned": learned,
        "forgotten": forgotten,
    })
    with open(fpath, 'wb') as f:
        pickle.dump(data, f)

def load_predictions_and_metadata(fpath):
    with open(fpath, 'rb') as f:
        return pickle.load(f)

# %%
def pairwise_pred_comparison(accs):
    accs = accs.astype(int)
    # Forward Transferred
    transferred_examples = (accs.sum(axis=0) == 2).astype(int)

    # Hard, Never Learned Examples
    hard_examples = (accs.sum(axis=0) == 0).astype(int)

    # Newly Learned
    learned_examples = (accs[1] - accs[0] == 1).astype(int)

    # Forgotten
    forgotten_examples = (accs[0] - accs[1] == 1).astype(int)

    return transferred_examples, hard_examples, learned_examples, forgotten_examples


def find_learning_and_forgetting_events(labels, pred_labels=None, pred_logits=None):
    # Must provide predicted labels or probabilities
    # assert pred_labels is not None or pred_logits is not None 
    # if pred_logits is not None:
    #     pred_probs = softmax(pred_logits, axis=1)
    #     pred_labels = np.argmax(pred_probs, axis=1)
    #     pred_ground_truth = pred_probs[:, np.arange(len(pred_logits)), labels[0]]

    #     # Same metrics as Dataset Cartography
    #     confidence = np.mean(pred_ground_truth, axis=0)
    #     stability = np.std(pred_ground_truth, axis=0)

    # difficulty = np.mean(pred_labels)

    transferred_examples = [np.zeros((labels.shape[1],))]
    hard_examples = [np.zeros((labels.shape[1],))]
    learned_examples = [np.zeros((labels.shape[1],))]
    forgotten_examples = [np.zeros((labels.shape[1],))]

    accs = (pred_labels == labels)
    for acc_i, acc_j  in zip(accs[:-1], accs[1:]):
        accs = np.array([acc_i, acc_j])

        example_categorization = pairwise_pred_comparison(accs)
        transferred_examples.append(example_categorization[0])
        hard_examples.append(example_categorization[1])
        learned_examples.append(example_categorization[2])
        forgotten_examples.append(example_categorization[3])
        print(example_categorization[0].shape)

    transferred_examples = np.array(transferred_examples)
    hard_examples = np.array(hard_examples)
    learned_examples = np.array(learned_examples)
    forgotten_examples = np.array(forgotten_examples)

    #return confidence, stability, difficulty, 
    return transferred_examples, hard_examples, learned_examples, forgotten_examples



def construct_example_flow_graph(transferred, hard, learned, forgotten):
    transition_dict = {}
    times_learned = learned.cumsum(axis=0)
    times_forgotten = forgotten.cumsum(axis=0)

    for idx in range(1, len(transferred)):
        # Dict Key: [Transition, prev correct or wrong, times forgotten, curr correct or wrong, times learned]


        # Transferred Examples
        cnt = Counter(times_learned[idx][transferred[idx] == 1])
        for num_learned, val in cnt.items():
            transition_dict[idx, 1, num_learned, 1, num_learned] = val

        # Hard Examples
        cnt = Counter(times_forgotten[idx][hard[idx] == 1])
        for num_hard, val in cnt.items():
            transition_dict[idx, 0, num_hard, 0, num_hard] = val

        # Learned Examples
        cnt = Counter()
        for num_forgotten, num_learned  in product(np.unique(times_forgotten[idx - 1]), np.unique(times_learned[idx])):
            val = (
                times_learned[idx][np.logical_and(learned[idx] == 1, times_forgotten[idx - 1] == num_forgotten)] == num_learned
            ).sum()
            if val:
                transition_dict[idx, 0, num_forgotten, 1, num_learned] = val

        # Forgotten Examples
        cnt = Counter()
        for num_learned, num_forgotten in product(np.unique(times_learned[idx - 1]), np.unique(times_forgotten[idx])):
            val = (
                times_forgotten[idx][np.logical_and(forgotten[idx] ==1, times_learned[idx - 1] == num_learned)] == num_forgotten
            ).sum()
            if val:
                transition_dict[idx, 1, num_learned, 0, num_forgotten] = val

    return transition_dict

def convert_flow_graph_to_plotly(sankey_flows):
    # time, correctness, num_forgotten or learned
    node_ids = []
    labels = []
    color = []
    node_heights = []
    for key in sankey_flows.keys():
        node_ids.append((key[0], key[1], key[2]))
        node_ids.append((key[0] + 1, key[3], key[4]))
    node_ids = sorted(node_ids)

    for time, correctness, learning_events in node_ids:
        node_heights.append(learning_events + 2 * correctness)

        event = "learned" if correctness else "forgotten" 
        correctness = "Correct" if correctness else "Wrong"
        labels.append(f"{correctness} at time {time} {event} {learning_events} times")

        if correctness == "Correct":
            color.append("green")
        else: color.append("red")

    node2idx = {node_id: ix for ix, node_id in enumerate(node_ids)}
    source, target, value = [], [], [], 
    for key, flow in sankey_flows.items():
        source.append(node2idx[(key[0], key[1], key[2])])
        target.append(node2idx[(key[0] + 1, key[3], key[4])])
        value.append(flow)

    return source, target,value, labels, color, node_heights


def plot_sankey(y_pos, color, source, target, value, title):
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node = dict(
        pad = 15,
        thickness = 20,
        line = dict(color = "black", width = 0.5),
        label=node_labels,
        y=node_heights,
        color = color
        ),
        link = dict(
        source=source,
        target=target,
        value=value
    )),])

    fig.update_layout(title_text=f"Learning and Forgetting Events {title}", font_size=10)
    return fig

# %%
def predict_and_save_data(trainer, model_name, data_dir, eval_checkpoints):
    _, pred_logits, labels = get_predictions(
        trainer,
        checkpoints=eval_checkpoints,
        checkpoint_dir=data_dir
    )

    pred_probs = [softmax(logits, axis=2) for logits in pred_logits]
    pred_labels = [np.argmax(probs, axis=2) for probs in pred_probs]

    # Events: transferred, hard, learned, forgotten
    events = [find_learning_and_forgetting_events(label, pred_label) for label, pred_label in zip(labels, pred_labels)]

    fpath = os.path.join(data_dir, f"preds_meta_{trainer.args.method}_{trainer.args.swa_ewa_lambda}.pkl")
    save_predictions_and_metadata(fpath, pred_probs, pred_labels, events, labels, model_name, eval_checkpoints,)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--exp_path', type=str, required=True)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--method', type=str, default='erm')
    parser.add_argument('--swa_ewa_decay_factor', type=float, default=0.5)
    args = parser.parse_args()

    EXP_PARAMS = {
        'dataset': args.dataset,
        'method': args.method,
        'device': 0,
        'random_seed': 1,
        'num_workers': 0,
        'eval_batch_size': args.eval_batch_size,

        'eval_fix': True,
        'eval_warmstart_finetune': False,
        'eval_features': False,
        'linear_probe': False,
        'online': False,

        'eval_all_timestamps': False,

        'load_model': False,
        'torch_compile': False,
        'sam': False,
        'swa_ewa': True,
        'swa_ewa_lambda': args.swa_ewa_decay_factor,
        'swa_steps': None,
        'swa_load_from_checkpoint': True,

        'data_dir': '/data/tir/projects/tir6/strubell/data/wilds/data',
        'log_dir': '/data/tir/projects/tir6/strubell/jaredfer/projects/wild-time/results',
        'results_dir': '/data/tir/projects/tir6/strubell/jaredfer/projects/wild-time/results',
        'exp_path': args.exp_path,
        'checkpoint_path': None,
    }

    if args.dataset == 'fmow':
        config =  {**configs_fmow.configs_fmow_erm, **EXP_PARAMS}
        eval_checkpoints = [f"time_{i}.pth" for i in range(11)]
    if args.dataset == 'huffpost':
        config =  {**configs_huffpost.configs_huffpost_erm, **EXP_PARAMS}
        eval_checkpoints = [f"time_{i}.pth" for i in range(2012, 2016)]
    if args.dataset == 'arxiv':
        eval_checkpoints = [f"time_{i}.pth" for i in range(2007, 2016)]
        config = {**configs_arxiv.configs_arxiv_erm, **EXP_PARAMS}
    if args.dataset == 'yearbook':
        eval_checkpoints = [f"time_{i}.pth" for i in range(1930, 1970)]
        config = {**configs_yearbook.configs_yearbook_erm, **EXP_PARAMS}
    if args.dataset == 'mimic':
        config = {**configs_mimic_mortality.configs_mimic_mortality_erm, **EXP_PARAMS}

    experimental_config = argparse.Namespace(**config)
    trainer = baseline_trainer.init(experimental_config)

    data_dir = os.path.join(config['results_dir'], config['exp_path'])
    predict_and_save_data(trainer, args.model, data_dir, eval_checkpoints)
