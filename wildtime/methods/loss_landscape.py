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


def generate_loss_contours(args, trainer):
    for timestep in args.contour_timesteps:
        import ipdb; ipdb.set_trace()
        trainer.eval_dataset.update_current_timestamp(timestep)
        trainer.eval_dataset.mode = 1
        dataloader = FastDataLoader(trainer.eval_dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers)

        if args.method in ['swa']:
            model = trainer.swa_model
        else:
            model = trainer.network

        logger.info(f"Generating contours from: {args.contour_models} at timestep {timestep}")
        models = [model, deepcopy(model), deepcopy(model)]
        for model, ckpt in zip(models, args.contour_models):
            weights = torch.load(os.path.join(args.model_path, f"time_{ckpt}.pth"))
            if args.method in ['swa'] and 'swa' not in str(ckpt):
                for key in list(weights.keys()):
                    weights["module." + key] = weights.pop(key)

            model.load_state_dict(weights, strict=False)
            model.to(args.device)

        eval_fn = create_eval_fn()
        contours = calculate_loss_contours(
            models[0], models[1], models[2],
            dataloader, eval_fn, args.device,
            granularity=args.contour_granularity, model_ids=args.contour_models, method=args.method)
        contours['title'] = f"{args.dataset} - Time: {timestep} ({args.contour_models})"

        model_idxs = [str(idx) for idx in args.contour_models]
        contour_dir = os.path.join(args.exp_path, 'loss_contours')
        if not os.path.exists(contour_dir):
            os.makedirs(contour_dir)

        np.save(f"{contour_dir}/{'_'.join(model_idxs)}.npy", contours)
        return contours


def calculate_loss_contours(
    model1, model2, model3, dataloader, eval_fn, device, granularity=20, margin=0.2, model_ids=None, method='erm'
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
    """
    w1 = flatten_parameters(model1).to(device=device)
    w2 = flatten_parameters(model2).to(device=device)
    w3 = flatten_parameters(model3).to(device=device)
    model1 = model1.to(device=device)

    # Define x axis
    u = w3 - w1
    dx = torch.norm(u).item()
    u /= dx

    # Define y axis
    v = w2 - w1
    v -= torch.dot(u, v) * u
    dy = torch.norm(v).item()
    v /= dy

    # Define grid representing parameters that will be evaluated.
    coords = np.stack(get_xy(p, w1, u, v) for p in [w1, w2, w3])
    alphas = np.linspace(0.0 - margin, 1.0 + margin, granularity)
    betas = np.linspace(0.0 - margin, 1.0 + margin, granularity)
    losses = np.zeros((granularity, granularity))
    accuracies = np.zeros((granularity, granularity))
    grid = np.zeros((granularity, granularity, 2))

    # Evaluate parameters at every point on grid
    progress = tqdm(total=granularity * granularity)
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            p = w1 + alpha * dx * u + beta * dy * v
            assign_params(model1, p)
            if method in ['swa']:
                update_bn(dataloader, model1, device)
            metrics = eval_fn(model1, dataloader, device)
            grid[i, j] = [alpha * dx, beta * dy]
            losses[i, j] = metrics["loss"]
            accuracies[i, j] = metrics["accuracy"]
            progress.update()
    progress.close()
    return {
        "grid": grid,
        "coords": coords,
        "losses": losses,
        "accuracies": accuracies,
        "model_ids": model_ids
    }


def plot_contour(
    grid,
    values,
    coords,
    labels,
    log_dir,
    title="figure",
    increment=0.3,
    margin=0.1,
    cmap="magma_r",
):
    sns.set(style="ticks")
    sns.set_context(
        "paper",
        rc={
            "lines.linewidth": 2.5,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "lines.markersize": 15,
            "legend.fontsize": 24,
            "axes.labelsize": 22,
            "axes.titlesize": 22,
            "legend.handlelength": 1,
            "legend.handleheight": 1,
        },
    )
    rc("text", usetex=False)
    matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    plt.figure(figsize=(6.13, 4.98))
    formatter = FormatStrFormatter("%1.2f")

    min_value = np.min(values) * (1 - margin)
    max_value = np.max(values) * (1 + margin)

    num_levels=int(-(- (max_value - min_value)  // increment))  # Ceiling division
    levels = [min_value + l * increment for l in range(num_levels)]
    norm = matplotlib.colors.Normalize(min_value, max_value)

    contour = plt.contour(
        grid[:, :, 0], grid[:, :, 1], values, cmap=cmap, levels=levels, norm=norm
    )
    contourf = plt.contourf(
        grid[:, :, 0], grid[:, :, 1], values, cmap=cmap, levels=levels, norm=norm
    )
    colorbar = plt.colorbar(contourf, format="%.2g")
    for idx, coord in enumerate(coords):
        plt.scatter(coord[0], coord[1], marker="o", c="k", s=120, zorder=2)
        plt.text(coord[0] + 0.05, coord[1] + 0.05, labels[idx], fontsize=22)

    plt.margins(0.0)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.title(title)
    colorbar.ax.tick_params(labelsize=20)
    colorbar.ax.yaxis.set_major_formatter(formatter)

    plot_path = os.path.join(log_dir, title)

    logger.info(f"Contour Plot Saved to {plot_path}")
    plt.savefig(plot_path + ".png", dpi=200, bbox_inches="tight")
    plt.savefig(plot_path + ".pdf", dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()



def get_xy(point, origin, vector_x, vector_y):
    """Return transformed coordinates of a point given parameters defining coordinate
    system.
    Args:
        point: point for which we are calculating coordinates.
        origin: origin of new coordinate system
        vector_x: x axis of new coordinate system
        vector_y: y axis of new coordinate system
    """
    return np.array(
        [
            torch.dot(point - origin, vector_x).item(),
            torch.dot(point - origin, vector_y).item(),
        ]
    )