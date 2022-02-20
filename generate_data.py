import itertools
import os
import pandas as pd

DATA_DIR = "data/"
DATA_PATH = DATA_DIR + "data.csv"


def generate_data(num_features, num_values):
    os.makedirs(DATA_DIR, exist_ok=True)

    inputs = itertools.product(range(num_values), repeat=num_features)

    samples = pd.DataFrame.from_records([i for i in inputs])

    samples.to_csv(DATA_PATH, index=True, index_label="label")
