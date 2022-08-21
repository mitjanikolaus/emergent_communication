import datetime
from collections import defaultdict

import editdistance
import torch
from scipy.spatial import distance
from scipy.stats import spearmanr
import numpy as np


def compute_entropy(messages):
    freq_table = defaultdict(float)

    for m in messages:
        m = tuple(m.tolist())
        freq_table[m] += 1.0

    t = torch.tensor([v for v in freq_table.values()]).float()
    if (t < 0.0).any():
        raise RuntimeError("Encountered negative probabilities")

    t /= t.sum()
    return (torch.where(t > 0, t.log(), t) * t).sum().item() / np.log(2)


def _hashable_tensor(t):
    if isinstance(t, tuple):
        return t
    if isinstance(t, int):
        return t

    try:
        t = t.item()
    except:
        t = tuple(t.view(-1).tolist())
    return t


def entropy(messages):
    from collections import defaultdict

    freq_table = defaultdict(float)

    for m in messages:
        m = _hashable_tensor(m)
        freq_table[m] += 1.0

    return entropy_dict(freq_table)


def entropy_dict(freq_table):
    H = 0
    n = sum(v for v in freq_table.values())

    for m, freq in freq_table.items():
        p = freq_table[m] / n
        H += -p * np.log(p)
    return H / np.log(2)


def mutual_info(xs, ys):
    e_x = entropy(xs)
    e_y = entropy(ys)

    xys = []

    for x, y in zip(xs, ys):
        xy = (_hashable_tensor(x), _hashable_tensor(y))
        xys.append(xy)

    e_xy = entropy(xys)

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


def compute_posdis(n_features, n_values, meanings, messages):
    batch_size = meanings.shape[0]
    features = meanings.view(batch_size, n_features, n_values).argmax(dim=-1)

    return information_gap_representation(features, messages)


def histogram(messages, vocab_size):
    batch_size = messages.size(0)

    histogram = torch.zeros(batch_size, vocab_size, device=messages.device)

    for v in range(vocab_size):
        histogram[:, v] = messages.eq(v).sum(dim=-1)

    return histogram


def compute_bosdis(meanings, messages, vocab_size):
    histograms = histogram(messages, vocab_size)

    return information_gap_representation(meanings, histograms[:, 1:])


def information_gap_representation(meanings, representations):
    gaps = torch.zeros(representations.size(1))
    non_constant_positions = 0.0

    for j in range(representations.size(1)):
        symbol_mi = []
        h_j = None
        for i in range(meanings.size(1)):
            x, y = meanings[:, i], representations[:, j]
            info = mutual_info(x, y)
            symbol_mi.append(info)

            if h_j is None:
                h_j = entropy(y)

        symbol_mi.sort(reverse=True)

        if h_j > 0.0:
            gaps[j] = (symbol_mi[0] - symbol_mi[1]) / h_j
            non_constant_positions += 1

    score = gaps.sum() / non_constant_positions
    return score.item()
