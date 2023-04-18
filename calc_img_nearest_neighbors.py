import argparse
import os
import pickle
import random

import h5py
import numpy as np
from tqdm import tqdm

from utils import DATA_DIR_IMAGENET, H5_IDS_KEY


def run(args):
    dataset_file = "val_features.hdf5"  # TODO train feats

    h5_db = h5py.File(os.path.join(DATA_DIR_IMAGENET, dataset_file), 'r')
    h5_ids = h5_db[H5_IDS_KEY]

    # TODO: fix for bug in extract feats:
    h5_ids = h5_ids[:len(h5_ids) - len(h5_ids) % 100]

    all_nns = {}
    for id in tqdm(h5_ids):
        img = np.array(h5_db[id])

        distances = []
        subsample = random.sample(list(h5_ids), 1000)
        for id_2 in subsample:
            if id_2 != id:
                img_2 = np.array(h5_db[id_2])
                dist = np.linalg.norm(img - img_2)
                distances.append(dist)

        ids = np.argsort(distances)
        nearest_neighbors = [h5_ids[ids[i]] for i in range(10)]
        all_nns[id] = nearest_neighbors

    pickle.dump(all_nns, open(os.path.join(DATA_DIR_IMAGENET, "nearest_neighors.p"), "wb"))


def get_args():
    parser = argparse.ArgumentParser()

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    run(args)
