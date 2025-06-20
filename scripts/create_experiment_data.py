import os
import sys

import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import STL
from tqdm import tqdm

sys.path.append(".")
from src.utils.data_loading import load_test_data, load_train_data
from src.utils.features import (
    decomps_and_features,
    seasonal_determination,
    trend_determination,
    trend_linearity,
    trend_slope,
)


def read_orig_train_data(dataset):
    past_targets = []
    future_targets = []

    for i in tqdm(range(50)):
        f_name = f"./data/{dataset}/training_data/batch{i}.csv"
        df = pd.read_csv(f_name, index_col=0)
        array = df.filter(like="observation").T.values
        past_targets.append(array[:, :168])
        future_targets.append(array[:, 168:])

    past_targets = np.concatenate(past_targets, axis=0)
    future_targets = np.concatenate(future_targets, axis=0)

    return past_targets, future_targets


def save_dataset_locally(dataset, past_targets, future_targets):
    os.makedirs(f"./data/{dataset}", exist_ok=True)

    # dummy data
    past_dummy = np.zeros_like(past_targets)
    future_dummy = np.zeros_like(future_targets)
    static_dummy = np.concatenate([past_dummy, future_dummy], axis=1)

    # train
    np.save(f"./data/{dataset}/train_past_target.npy", past_targets)
    np.save(f"./data/{dataset}/train_past_time_feat.npy", past_dummy)
    np.save(f"./data/{dataset}/train_past_dynamic_age.npy", past_dummy)

    np.save(f"./data/{dataset}/train_future_target.npy", future_targets)
    np.save(f"./data/{dataset}/train_future_time_feat.npy", future_dummy)
    np.save(f"./data/{dataset}/train_future_dynamic_age.npy", future_dummy)

    np.save(f"./data/{dataset}/train_feat_static_cat.npy", static_dummy)

    # validation, we don't use it but need the files for the training code to run
    np.save(f"./data/{dataset}/validation_past_target.npy", past_dummy)
    np.save(f"./data/{dataset}/validation_past_time_feat.npy", past_dummy)
    np.save(f"./data/{dataset}/validation_past_dynamic_age.npy", past_dummy)

    np.save(f"./data/{dataset}/validation_future_target.npy", future_dummy)
    np.save(f"./data/{dataset}/validation_future_time_feat.npy", future_dummy)
    np.save(f"./data/{dataset}/validation_future_dynamic_age.npy", future_dummy)

    np.save(f"./data/{dataset}/validation_feat_static_cat.npy", static_dummy)


def calculate_and_save_features(dataset, sp, ts_length, use_test):
    if use_test:
        file_prefix = "test"
        time_series = [ts.to_numpy() for ts in load_test_data(dataset, ts_length)]
    else:
        file_prefix = "train"
        past_targets = np.load(f"./data/{dataset}/{file_prefix}_past_target.npy")
        future_targets = np.load(f"./data/{dataset}/{file_prefix}_future_target.npy")
        time_series = np.concatenate([past_targets, future_targets], axis=1)

    features = np.empty((len(time_series), 4))
    for i, ts in tqdm(enumerate(time_series)):
        decomp = STL(ts, period=sp).fit()

        features[i, 0] = trend_determination(decomp.trend, decomp.resid)
        features[i, 1] = trend_slope(decomp.trend)
        features[i, 2] = trend_linearity(decomp.trend)
        features[i, 3] = (
            seasonal_determination(decomp.seasonal, decomp.resid) if sp > 1 else 0
        )

    np.save(f"./data/{dataset}/{file_prefix}_features.npy", features)


def create_alternative_dataset(dataset, batch_size, seasonal_period):
    dfs = load_train_data(f"./data/{dataset}", batch_size)
    decomps, _ = decomps_and_features(dfs, seasonal_period)

    alternative_data = []
    for decomp in tqdm(decomps):
        ts = decomp.trend + decomp.seasonal + decomp.resid
        array = ts.values

        mean_multiplier = np.random.uniform(1, 4)
        split_idx = np.random.randint(72, 144)
        array[split_idx:] += np.mean(array) * mean_multiplier
        alternative_data.append(array)

    alternative_data = np.concatenate(alternative_data, axis=0).reshape((-1, 192))

    past_targets = []
    future_targets = []

    for ts in tqdm(alternative_data):
        past_targets.append(ts[np.newaxis, :168])
        future_targets.append(ts[np.newaxis, 168:])

    past_targets = np.concatenate(past_targets, axis=0)
    future_targets = np.concatenate(future_targets, axis=0)

    return past_targets, future_targets


def create_augmented_dataset(dataset):
    past_targets, future_targets = read_orig_train_data(dataset)

    past_targets = np.concatenate(
        [past_targets, np.load(f"./data/{dataset}_alternative/train_past_target.npy")],
        axis=0,
    )
    future_targets = np.concatenate(
        [
            future_targets,
            np.load(f"./data/{dataset}_alternative/train_future_target.npy"),
        ],
        axis=0,
    )

    return past_targets, future_targets


if __name__ == "__main__":
    dataset = "electricity_nips"
    batch_size = 512
    seasonal_period = 24
    ts_length = 192

    if os.path.exists(f"./data/{dataset}/training_data"):
        past_targets, future_targets = read_orig_train_data(dataset)
    else:
        raise FileNotFoundError(
            f"Dataset {dataset} not found in ./data/{dataset}/training_data"
        )

    save_dataset_locally(dataset, past_targets, future_targets)
    calculate_and_save_features(dataset, seasonal_period, ts_length, use_test=False)
    calculate_and_save_features(dataset, seasonal_period, ts_length, use_test=True)

    # Create and save alternative dataset and features. The test data is the same for
    # all datasets so we don't save those.
    alternative_past_targets, alternative_future_targets = create_alternative_dataset(
        dataset, batch_size, seasonal_period
    )
    save_dataset_locally(
        dataset + "_alternative", alternative_past_targets, alternative_future_targets
    )
    calculate_and_save_features(
        dataset + "_alternative", seasonal_period, ts_length, use_test=False
    )

    augmented_past_targets, augmented_future_targets = create_augmented_dataset(dataset)
    save_dataset_locally(
        dataset + "_augmented", augmented_past_targets, augmented_future_targets
    )
    calculate_and_save_features(
        dataset + "_augmented", seasonal_period, ts_length, use_test=False
    )
