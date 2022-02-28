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


def compute_topsim(
    meanings: torch.Tensor,
    messages: torch.Tensor,
    meaning_distance_fn: str = "hamming",
    message_distance_fn: str = "edit",
) -> float:

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

    return topsim
