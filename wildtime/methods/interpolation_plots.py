import json
import logging
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import torch

from copy import deepcopy
from tqdm import tqdm
from matplotlib import rc
from collections import defaultdict
from matplotlib.ticker import FormatStrFormatter
from torch.optim.swa_utils import update_bn

from .utils import create_eval_fn, flatten_parameters, assign_params
from .dataloaders import FastDataLoader, InfiniteDataLoader


logger = logging.getLogger(__name__)


def generate_interpolation_plots(args, trainer):
    for timestep in args.interpolation_timesteps:
        dataloaders = dict()

        for mode in [0, 1]:
            trainer.eval_dataset.update_current_timestamp(timestep)
            trainer.eval_dataset.mode = mode
            dataloaders[mode] = FastDataLoader(
                trainer.eval_dataset, 
                batch_size=args.eval_batch_size, 
                num_workers=args.num_workers
                )

        if args.method in ['swa']:
            model = trainer.swa_model
        else:
            model = trainer.network

        logger.info(f"Generating interpolation plots from: {args.interpolation_models} at timestep {timestep}")
        models = [model, deepcopy(model)]

        for model, ckpt in zip(models, args.interpolation_models):
            weights = torch.load(os.path.join(args.model_path, f"time_{ckpt}.pth"))
            if args.method in ['swa'] and 'swa' not in str(ckpt):
                for key in list(weights.keys()):
                    weights["module." + key] = weights.pop(key)

            model.load_state_dict(weights, strict=False)
            model.to(args.device)

        eval_fn = create_eval_fn()

        interpolated = dict()
        for mode, dataloader in dataloaders.items():
            interpolated[mode] = calculate_interpolation_accuracy(
                models[0], models[1],
                dataloader, 
                eval_fn, 
                args.device,
                granularity=args.interpolation_granularity, model_ids=args.interpolation_models, method=args.method)
            interpolated['title'] = f"{args.dataset} - Time: {timestep} ({args.interpolation_models})"

            model_idxs = [str(idx) for idx in args.interpolation_models]
            interpolation_dir = os.path.join(args.exp_path, 'interpolation')
            if not os.path.exists(interpolation_dir):
                os.makedirs(interpolation_dir)

            np.save(f"{interpolation_dir}/mode={mode}_time={timestep}_start={model_idxs[0]}_end={model_idxs[1]}.npy", interpolated[mode])

    return interpolated


def calculate_interpolation_accuracy(
    model1, model2, dataloader, 
    eval_fn, 
    device, granularity=20, model_ids=None, method='erm'
):
    """Runs the loss contour analysis.
    Creates plane based on the parameters of 3 models, and computes loss and accuracy
    contours on that plane. Specifically, computes 2 axes based on the 3 models, and
    computes metrics on points defined by those axes.
    Args:
        model1: Origin of plane.
        model2: Model used to define y axis of plane.
        model3: Model used to define x axis of plane.
        dataloader: Dataloader for the dataset to evaluate on.
        eval_fn: A function that takes a model, a dataloader, and a device, and returns
            a dictionary with two metrics: "loss" and "accuracy".
        device: Device that the model and data should be moved to for evaluation.
        granularity: How many segments to divide each axis into. The model will be
            evaluated at granularity*granularity points.
        margin: How much margin around models to create evaluation plane.
        plot_comb: List of indices of [model1, model2, model3] that should be averaged and plotted
    """
    w1 = flatten_parameters(model1).to(device=device)
    w2 = flatten_parameters(model2).to(device=device)
    model1 = model1.to(device=device)

    alphas = np.linspace(0.0, 1.0, granularity)
    losses = np.zeros((granularity,))
    accuracies = np.zeros((granularity,))

    # Evaluate parameters at every point on grid
    progress = tqdm(total=granularity)
    interp_model = deepcopy(model1)
    for i, alpha in enumerate(alphas):
        p = (1 - alpha) * w1 + (alpha * w2)
        assign_params(interp_model, p)
        if method in ['swa']:
            update_bn(dataloader, interp_model, device)
        metrics = eval_fn(interp_model, dataloader, device)
        losses[i] = metrics["loss"]
        accuracies[i] = metrics["accuracy"]
        progress.update()
    progress.close()

    outputs = {
        "losses": losses,
        "accuracies": accuracies,
        "model_ids": model_ids,
    }

    return outputs