import argparse
import os
import sys
import yaml

import numpy as np

sys.path.append(".")
from src.utils.features import decomps_and_features
from src.utils.data_loading import load_train_data, load_test_data


parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=str, help="Path to a config file")
parser.add_argument(
    "--test-data",
    type=int,
    default=0,
    help="0 for calculating features for training data, 1 for test " "data",
)

if __name__ == "__main__":
    args = vars(parser.parse_args())

    with open(args["config_path"], "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args["test_data"]:
        dfs = load_test_data(
            config["dataset"], config["context_length"] + config["prediction_length"]
        )
    else:
        dfs = load_train_data(config["datadir"], config["trainer_args"]["batch_size"])

    print("Calculating features")
    _, features = decomps_and_features(dfs, config["sp"])

    print("Writing features to file")
    os.makedirs(config["datadir"], exist_ok=True)
    features = np.nan_to_num(features)  # convert nan features to 0
    np.save(
        os.path.join(
            config["datadir"],
            f"{'test' if args['test_data'] else 'train'}_features.npy",
        ),
        features,
    )
