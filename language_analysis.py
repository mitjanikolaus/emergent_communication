import datetime
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import product, combinations
from math import ceil

import editdistance
import torch
from scipy.spatial import distance
from scipy.stats import spearmanr
import numpy as np
from torch import Tensor, tensor
from tqdm import tqdm


def compute_variability_metrics(meanings_transformed, messages, num_attributes, num_values, max_len, vocab_size):
    # calc probs
    attributes, characters, values, letters_given_values, letter_by_position = mle_probabilities(meanings_transformed, messages)



    # calc co_probs
    letter_given_value_co_prob, value_given_letter_co_prob \
        = conditional_probabilities(
        attributes,
        letters_given_values,
        letter_by_position
    )

    co_prob_tensors = convert_dict_to_tensors(meanings_transformed, letter_given_value_co_prob, num_attributes, num_values, max_len, vocab_size)
    syn = synonymy(co_prob_tensors)
    hom = homonymy(co_prob_tensors)
    free = freedom(co_prob_tensors)
    ent = entanglement(co_prob_tensors)

    return syn, hom, free, ent

def freedom(prob_table: list):
    '''
    Assess atom order freedom for a given probability table

    '''
    role_uniformity = []

    for semantic_role in prob_table:
        entropy = torch.distributions.Categorical(semantic_role).entropy()
        entropy = entropy.mean(0)  # collapse across all atoms in each position
        max_entr = max_entropy(semantic_role)
        divergent_entropy = entropy.min(dim=-1).values
        bounded_divergent_entropy = divergent_entropy / max_entr

        role_uniformity.append(bounded_divergent_entropy.item())

    language_uniformity = float(np.mean(role_uniformity))

    return language_uniformity


def entanglement(prob_table: list):
    mean_entropies, maxes = [], []
    for semantic_role in prob_table:
        try:
            entropy = torch.distributions.Categorical(semantic_role).entropy()
        except:
            print("entaglement entropy failed, using norm or uniform")
            print("means some n_grams unattested - measure unreliable")
            print("(try a larger sample of data)")
            norm_role = norm_or_uniform(semantic_role)
            entropy = torch.distributions.Categorical(norm_role).entropy()
        mean_entropy = entropy.mean(dim=0)
        mean_entropies.append(mean_entropy)
        maxes.append(
            semantic_role.max(-1).values.mean(0)
        )

    pairs = combinations(mean_entropies, r=2)

    differences = []
    for pair in pairs:
        diff = abs(pair[0] - pair[1])
        pair_max = (max([pair[0].max(), pair[1].max()])).item()
        pair_max = pair_max if pair_max > 0 else 0.000001
        differences.append(float(1 - (diff.max().item() / pair_max)))

    pairs = combinations(maxes, r=2)
    max_differences = []
    for pair in pairs:
        diff = abs(pair[0] - pair[1])
        pair_max = (max([pair[0].max(), pair[1].max()])).item()
        pair_max = pair_max if pair_max > 0 else 0.000001
        max_differences.append(float(1 - (diff.max().item() / pair_max)))

    entanglement = float(np.mean(differences))

    return entanglement

def norm_or_uniform(prob_table):
    '''
    takes a tensor of pmis and normalizes it on the last axis. If that last
    axis is all 0s then it's replaced with a uniform distribution

    pmi_tensor: list of tensors, len n_roles
        dimensions n_atoms X n_signal_len X n_chars
    '''
    if type(prob_table) == list:
        norm_pmi = deepcopy(prob_table)
        for semantic_role in norm_pmi:
            for atom_id, n_gram_sums in enumerate(semantic_role.sum(axis=-1).tolist()):
                for position_id, n_gram_sum in enumerate(n_gram_sums):
                    if n_gram_sum == 0.0:
                        semantic_role[atom_id][position_id] = semantic_role[atom_id][position_id] + 1
        normed_tensor = []
        for semantic_role in norm_pmi:
            normed_tensor.append(semantic_role / semantic_role.sum(-1, keepdim=True))

    else:
        for atom_id, n_gram_sums in enumerate(prob_table.sum(axis=-1).tolist()):
            for position_id, n_gram_sum in enumerate(n_gram_sums):
                if n_gram_sum == 0.0:
                    prob_table[atom_id][position_id] = prob_table[atom_id][position_id] + 1

        return prob_table / prob_table.sum(-1, keepdim=True)

    return normed_tensor


def homonymy(prob_table: list):
    '''

    !uses role isolation!
    Takes a list of prob Tensors each n_atoms X N_signal_len X
    n_chars. Refactors to len X chars X atoms, concatenates all roles.
    Normalizes then assesses entropy on the last axis, and means across all
    positions to get language level homonomy.

    Note: This drops all unused characters before normalizing

    prob_table: list of tensors, len n_roles
        dimensions n_atoms X n_signal_len X n_chars
    '''

    all_role_postions = []
    for role in prob_table:
        letters_last = role.permute(1, 2, 0)
        all_letters = torch.flatten(letters_last, start_dim=-1)
        by_position = [position for position in all_letters]
        all_role_postions.append(by_position)

    lang_homonomy = []
    for role in all_role_postions:
        role_homonomy = []
        max_entropies = []
        for position in role:
            non_zero_positions = position[position.sum(-1) > 0]
            if non_zero_positions.shape[0] == 0:
                position = torch.ones_like(position)
                non_zero_positions = position

            max_entr = max_entropy(non_zero_positions)
            max_entropies.append(max_entr)

            entropy = torch.distributions.Categorical(
                non_zero_positions
            ).entropy()

            entropy = (entropy.mean()) / max_entr
            role_homonomy.append(entropy)

        lang_homonomy.append(min(role_homonomy))

    return float(np.mean(lang_homonomy))


def synonymy(prob_table: list):
    '''
    Takes a list of probability tensors from a normalized PMI tensor,
    calculates entropy on the last dimension. Divides the max by the entropy
    of a uniform distribution and means across all values to get the
    language level synonymy

    prob_table: list of tensors, len n_roles
        dimensions n_atoms X n_signal_len X n_chars
    '''
    role_entropies = []
    for semantic_role in prob_table:
        atom_entropies = []
        for atom in semantic_role:
            non_zero_positions = atom[atom.sum(-1) > 0]
            if len(non_zero_positions) == 0:
                non_zero_positions = torch.ones_like(atom)
            entropy = torch.distributions.Categorical(non_zero_positions).entropy()
            max_entr = max_entropy(atom)

            atom_synonymy = entropy.min() / max_entr
            atom_entropies.append(atom_synonymy)

        role_entropies.append(float(np.mean(atom_entropies)))

    language_synonymy = float(np.mean(role_entropies))

    return language_synonymy


def max_entropy(probabilities: tensor):
    ''''
    Takes a tensor as input, and returns the entropy of a same-sized uniform
    probability distribution. Used for bounding measures between zero and one
    '''
    uniform_ones = torch.ones(probabilities.shape)
    max_entropies = torch.distributions.Categorical(uniform_ones).entropy()
    theoretical_max = max_entropies.max()
    return theoretical_max


def mean_co_probs_by_position(co_probs: tensor, top_k=1):
    '''
    Co probs: full probability tensor
    top_k: how many probabilities per (position, atom, role) combo should be
        meaned

    '''
    means_by_position = []
    for semantic_role in co_probs:
        top_correlates = torch.topk(semantic_role, top_k, dim=-1)
        top_values = top_correlates.values.squeeze()
        means_by_position.append(top_values.permute(1, 0).mean(dim=-1))

    return means_by_position


def convert_dict_to_tensors(meanings_transformed, co_probs: defaultdict, num_attributes, num_values, max_len, vocab_size, n_size: int = 1):
    '''
    converts the probability table, comprised of nested dicts to a list of
    tensors 1 per role. Note that it's a LIST of tensors to straight-forwardly
    allow each role to have a different number of atoms (i.e. 10 subjects and 20 verbs)
    '''
    dict_dimensions = fetch_dims(meanings_transformed, num_attributes, num_values, max_len, vocab_size, n_size=n_size)
    tensors = create_tensors_from_dims(dict_dimensions)

    for attr_ix, attribute in enumerate(co_probs):
        for value_ix, value in enumerate(co_probs[attribute]):
            for pos_ix, signal_position in enumerate(co_probs[attribute][value]):
                for char_ix, character in enumerate(co_probs[attribute][value][signal_position]):
                    tensors[attr_ix][value_ix][pos_ix][int(character)] \
                        = co_probs[attribute][value][signal_position][character]

    return [tens.float() for tens in tensors]


def create_tensors_from_dims(dims: list, floor: float = 0.0):
    '''
    Dims: list of tensor dimensions, default zeros
    floor: lowest value in tensor

    used to convert nested dictionaries of known depth (dims) to a list of
    tensors. in practice used here to go from co-prob table to a lost of prob
    tensors with one for each role
    '''
    tensors = []
    for i in dims:
        zero_tensors = torch.zeros(i)
        tensors.append(zero_tensors+floor) #floor is added to account for unused atoms
    return tensors


def fetch_dims(dataset, num_attributes, num_values, max_len, vocab_size, n_size=1):
    '''
    Gets the expected dimensions of the full probability table, by checking
    the number of roles, atoms, and chars - it can handle n_grams larger than
    1 for determining the number of chars via the n_size param
    '''

    n_positions = ceil(max_len / n_size)

    # size of final dim depends on n_gram size
    n_characters = 0
    combinations = product(range(vocab_size), repeat=n_size)
    for _ in combinations:
        n_characters += 1

    dims = [(num_values, n_positions, n_characters) for i in range(num_attributes)]

    return dims

def conditional_probabilities(
        attributes: defaultdict,
        letters_given_values: defaultdict,
        letter_by_position: defaultdict
):
    '''
    Uses raw counts to generate conditional probabilities for
    char | position, atom, role & atom | char, position, role

    '''

    letter_given_value_co_prob = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: 0.1
                )
            )
        )
    )
    value_given_letter_co_prob = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: 0.1
                )
            )
        )
    )
    for attribute in letters_given_values:
        for value in letters_given_values[attribute]:
            for signal_position in letters_given_values[attribute][value]:
                for character in letters_given_values[attribute][value][signal_position]:
                    letter_given_value_co_prob[attribute][value][signal_position][character] \
                        = letters_given_values[attribute][value][signal_position][character] \
                          / attributes[attribute][value]
                    value_given_letter_co_prob[attribute][signal_position][character][value] \
                        = letters_given_values[attribute][value][signal_position][character] \
                          / letter_by_position[signal_position][character]

    return letter_given_value_co_prob, value_given_letter_co_prob


def mle_probabilities(meanings_transformed, messages):
    '''
    Estimates the raw counts for how often attributes, values, and characters
    each occur, and occur together.
    '''
    attributes = defaultdict(lambda: defaultdict(lambda: 0.0))#semantic roles
    characters = defaultdict(lambda: 0.0)
    values = defaultdict(lambda: 0.0)#atoms

    letters_given_values = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: 0.0
                )
            )
        )
    )
    letter_by_position = defaultdict(lambda: defaultdict(lambda: 0))

    for example_id, example in enumerate(meanings_transformed):
        for attribute, value in enumerate(example):
            for signal_position, letter in enumerate(messages[example_id]):
                letters_given_values[attribute][value.item()][signal_position][letter.item()] += 1
                letter_by_position[signal_position][letter.item()] += 1
                characters[letter.item()] += 1
            attributes[attribute][value.item()] += 1
            values[value.item()] += 1

    # for example_id, example in enumerate(dataset.examples):
    #     for semantic_role, atom in zip(example.dep_tags, example.source):
    #         for signal_position, letter in enumerate(signals[example_id][:-1]):  # slice omits eos token
    #
    #             letters_given_atoms[semantic_role][atom][signal_position][letter] \
    #                 = letters_given_atoms[semantic_role][atom][signal_position][letter] + 1
    #
    #             letter_by_position[signal_position][letter] \
    #                 = letter_by_position[signal_position][letter] + 1
    #
    #             characters[letter] = characters[letter] + 1
    #
    #         semantic_roles[semantic_role][atom] = semantic_roles[semantic_role][atom] + 1
    #         atoms[atom] = atoms[atom] + 1

    return attributes, characters, values, letters_given_values, letter_by_position

def compute_synonymy(meanings, messages):
    attributes = meanings.view(batch_size, n_attributes, n_values).argmax(dim=-1)


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
