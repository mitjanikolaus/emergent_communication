import datetime
from collections import Counter

import editdistance
import torch
from scipy.spatial import distance
from scipy.stats import spearmanr
import numpy as np
from torch import Tensor
from tqdm import tqdm


def compute_entropy(messages):
    if not isinstance(messages, list) and len(messages.shape) > 1:
        _, occs = np.unique(messages, return_counts=True, axis=0)
        occs = occs.astype(float)
        freqs = occs / len(messages)
    else:
        occs = Counter(messages).values()
        freqs = [occ / len(messages) for occ in occs]

    return sum([-freq * np.log(freq) for freq in freqs]) / np.log(2)


def mutual_info(xs, ys):
    e_x = compute_entropy(xs)
    e_y = compute_entropy(ys)

    xys = [(x, y) for x, y in zip(xs, ys)]

    e_xy = compute_entropy(xys)

    return e_x + e_y - e_xy


def compute_topsim(
    meanings: torch.Tensor,
    messages: torch.Tensor,
    meaning_distance_fn: str = "hamming",
    message_distance_fn: str = "edit",
) -> float:

    print(f"Start computing topsim for {len(meanings)} items")
    start_time = datetime.datetime.now()

    distances = {
        "edit": lambda x, y: editdistance.eval(x, y) / ((len(x) + len(y)) / 2),
        "cosine": distance.cosine,
        "hamming": distance.hamming,
        "jaccard": distance.jaccard,
        "euclidean": distance.euclidean,
    }

    meaning_distance_fn = (
        distances.get(meaning_distance_fn, None)
        if isinstance(meaning_distance_fn, str)
        else meaning_distance_fn
    )
    message_distance_fn = (
        distances.get(message_distance_fn, None)
        if isinstance(message_distance_fn, str)
        else message_distance_fn
    )

    assert (
        meaning_distance_fn and message_distance_fn
    ), f"Cannot recognize {meaning_distance_fn} \
        or {message_distance_fn} distances"

    meaning_dist = distance.pdist(meanings, meaning_distance_fn)
    message_dist = distance.pdist(messages, message_distance_fn)

    topsim = spearmanr(meaning_dist, message_dist, nan_policy="raise").correlation

    end_time = datetime.datetime.now()
    difference = end_time - start_time
    print(f"End computing topsim. Duration: {difference.total_seconds()}s")
    return topsim


def compute_posdis(n_attributes, n_values, meanings, messages):
    batch_size = meanings.shape[0]
    attributes = meanings.view(batch_size, n_attributes, n_values).argmax(dim=-1)

    return information_gap_representation(attributes, messages)


def histogram(messages, vocab_size):
    batch_size = messages.shape[0]

    histogram = torch.zeros(batch_size, vocab_size, device=messages.device)

    for v in range(vocab_size):
        histogram[:, v] = messages.eq(v).sum(dim=-1)

    return histogram


def compute_bosdis(n_attributes, n_values, meanings, messages, vocab_size):
    batch_size = meanings.shape[0]
    histograms = histogram(messages, vocab_size)
    attributes = meanings.view(batch_size, n_attributes, n_values).argmax(dim=-1)

    return information_gap_representation(attributes, histograms[:, 1:])


def information_gap_representation(meanings, representations):
    if isinstance(representations, Tensor):
        representations = representations.numpy()
    if isinstance(meanings, Tensor):
        meanings = meanings.numpy()

    gaps = torch.zeros(representations.shape[1])
    non_constant_positions = 0.0

    for j in tqdm(range(representations.shape[1])):
        h_j = compute_entropy(representations[:, j])

        symbol_mi = [(mutual_info(meanings[:, i], representations[:, j])) for i in range(meanings.shape[1])]

        symbol_mi.sort(reverse=True)

        if h_j > 0.0:
            gaps[j] = (symbol_mi[0] - symbol_mi[1]) / h_j
            non_constant_positions += 1

    score = gaps.sum() / non_constant_positions
    return score.item()
