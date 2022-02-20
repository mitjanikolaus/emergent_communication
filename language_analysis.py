import editdistance
import torch
from scipy.spatial import distance
from scipy.stats import spearmanr


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
