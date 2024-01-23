import argparse
import os
import logging
import pickle
import resource
from collections import Counter, defaultdict
from itertools import product

import numpy as np
import torch.multiprocessing
from scipy.special import softmax

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

torch.multiprocessing.set_sharing_strategy('file_system')


def get_correctness(data):
    corr = []
    for ix, (label, pred) in enumerate(zip(data['labels'], data['pred_labels'])):
        corr.append(label==pred)
    return corr


def evaluate_method(corr, corr_exp):
    transferred, hard, learned, forgotten = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for eval_split, (baseline, exp) in enumerate(zip(corr, corr_exp)):
        for train_step, (pred, pred_exp) in enumerate(zip(baseline, exp)):
            if train_step == 0: continue
            # t, h, l, f = pairwise_pred_comparison(np.array([baseline[train_step-1], exp[train_step]]))
            t, h, l, f = pairwise_pred_comparison(np.array([baseline[0], exp[-1]]))

            transferred[eval_split].append(t.sum())
            hard[eval_split].append(h.sum())
            learned[eval_split].append(l.sum())
            forgotten[eval_split].append(f.sum())
            break
    return transferred, hard, learned, forgotten

def get_predictions(trainer, checkpoints, checkpoint_dir):
    metrics = []
    preds = []
    labels = []

    eval_splits = [i for i in trainer.eval_dataset.ENV if i > trainer.split_time]
    for split in eval_splits:
        split_preds = []
        split_labels = []
        for ckpt in checkpoints:
            ckpt_path = os.path.join(checkpoint_dir, ckpt)
            trainer.load_model(checkpoint_path=ckpt_path)
            acc, pred, label = trainer.run_eval_timestamp(split, mode=2)

            split_preds.append(pred)
            split_labels.append(label)
            print(f"Accuracy of FT until {ckpt} on {split}: {acc}")

        preds.append(np.array([np.vstack(_) for _ in split_preds]))
        labels.append(np.array(split_labels))

    return metrics, preds, labels


def save_predictions_and_metadata(fpath, pred_probs, pred_labels, events, labels, model, eval_checkpoints, eval_splits):
    transferred = [i[0] for i in events]
    hard = [i[1] for i in events]
    learned = [i[2] for i in events]
    forgotten = [i[3] for i in events]

    data = dict({
        "model": model,
        "eval_checkpoints": eval_checkpoints,
        "eval_splits": eval_splits,
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


def plot_sankey(source, target, value, labels, color, y_pos, title):
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node = dict(
        pad = 15,
        thickness = 20,
        line = dict(color = "black", width = 0.5),
        label=labels,
        # y=node_heights,
        color = color
        ),
        link = dict(
        source=source,
        target=target,
        value=value
    )),])

    fig.update_layout(title_text=f"Learning and Forgetting Events {title}", font_size=10)
    return fig