import argparse
import os
import sys
import yaml

import numpy as np
import torch
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import ValidationDataLoader
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.torch.batchify import batchify
from gluonts.time_feature import (
    hour_of_day,
    day_of_week,
    day_of_month,
    day_of_year,
    month_of_year,
)
from gluonts.transform import (
    Chain,
    AddTimeFeatures,
    InstanceSplitter,
    ValidationSplitSampler,
    AddObservedValuesIndicator,
)
from tqdm import tqdm

sys.path.append(".")
from src.utils.evaluation import score_batch
from src.models.utils import get_model


parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=str, help="Path to a config file")

if __name__ == "__main__":
    args = vars(parser.parse_args())

    with open(args["config_path"], "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_args = config["model_args"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(config["model_name"])(
        **model_args, device=device, path=config["path"]
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(config["path"], "model.pth")))

    dataset = get_dataset(config["dataset"])
    transformation = Chain(
        [
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                pred_length=config["prediction_length"],
                time_features=[
                    hour_of_day,
                    day_of_week,
                    day_of_month,
                    day_of_year,
                    month_of_year,
                ],
            ),
            InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=ValidationSplitSampler(
                    min_future=config["prediction_length"]
                ),
                past_length=config["context_length"],
                future_length=config["prediction_length"],
                time_series_fields=[FieldName.FEAT_TIME, FieldName.OBSERVED_VALUES],
            ),
        ]
    )
    dataloader = ValidationDataLoader(
        dataset.test,
        batch_size=config["trainer_args"]["batch_size"],
        stack_fn=batchify,
        transform=transformation,
    )

    mape = []
    smape = []
    mase = []
    seasonal_mase = []
    mse = []
    mae = []
    model.eval()
    for batch in tqdm(dataloader):
        predictions = model.predict(batch)[:, :, 0]

        context = batch["past_target"].unsqueeze(dim=-1).numpy()
        target = batch["future_target"].numpy()
        scores = score_batch(target, predictions, context, config["sp"])

        mape.append(scores[0])
        smape.append(scores[1])
        mase.append(scores[2])
        seasonal_mase.append(scores[3])
        mse.append(scores[4])
        mae.append(scores[5])

    mape = np.vstack(mape)
    smape = np.vstack(smape)
    mase = np.vstack(mase)
    seasonal_mase = np.vstack(seasonal_mase)
    mse = np.vstack(mse)
    mae = np.vstack(mae)

    np.save(os.path.join(config["path"], "mape.npy"), mape)
    np.save(os.path.join(config["path"], "smape.npy"), smape)
    np.save(os.path.join(config["path"], "mase.npy"), mase)
    np.save(os.path.join(config["path"], "seasonal_mase.npy"), seasonal_mase)
    np.save(os.path.join(config["path"], "mse.npy"), mse)
    np.save(os.path.join(config["path"], "mae.npy"), mae)
