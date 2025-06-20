import argparse
import logging
import os
import random
import sys
import yaml

import numpy as np
import torch

sys.path.append(".")
from src.models.trainer import Trainer
from src.models.utils import get_model


parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=str, help="Path to a config file")

if __name__ == "__main__":
    args = vars(parser.parse_args())

    with open(args["config_path"], "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    os.makedirs(config["path"], exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(config["path"], "train.log"),
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S",
    )
    logging.getLogger().addHandler(
        logging.StreamHandler()
    )  # print to stdout as well as file

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    model_args = config["model_args"]
    trainer_args = config["trainer_args"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model(config["model_name"])(
        **model_args, device=device, path=config["path"]
    ).to(device)
    trainer = Trainer(**trainer_args, sp=config["sp"])

    datadir = os.path.join("./data", config["dataset"])
    model.fit(trainer, datadir)
