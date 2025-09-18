import os
import shutil
from typing import List

import pandas as pd
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.accelerators import find_usable_cuda_devices
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch import set_float32_matmul_precision
from tqdm import tqdm

from physioex.data import PhysioExDataModule
from physioex.train.models.load import load_model

# datasets = ["hmc", "sleepedf", "dcsm", "mass"]
# datasets = ["hmc", "sleepedf", "mass"]

datasets = ["dcsm", "hmc", "mass", "sleepedf"]

model_kwargs = {"in_channels": 3, "sequence_length": 21}

datamodule_kwargs = {
    "batch_size": 1,
    "preprocessing": "xsleepnet",
    "selected_channels": ["EEG", "EOG", "EMG"],
    "sequence_length": 21,
    "data_folder": "/data/leuven/365/vsc36564/sleep-data/",
    "num_workers": 23,
    "data_prefetch": False,
}

models_names = [
    # "seqsleepnet",
    # "sleeptransformer",
    "protoseqsleepnet",
    # "protosleeptransformer",
]

models = [
    # "physioex.train.networks.seqsleepnet:SeqSleepNet",
    # "physioex.train.networks.sleeptransformer:SleepTransformer",
    "physioex.train.networks.protoseqsleepnet:ProtoSeqSleepNet",
    # "physioex.train.networks.protosleeptransformer:ProtoSleepTransformerNet",
]


def test_performances(
    datasets: List[str] = None,
    datamodule_kwargs: dict = None,
    model_name: str = "seqsleepnet",  # if passed model_class, model_config and resume are ignored
    model: str = "physioex.train.networks.seqsleepnet:SeqSleepNet",
    model_kwargs: dict = None,
) -> pd.DataFrame:
    for dataset in datasets:
        logger.info(f"Testing on dataset: {dataset}")

        datamodule = PhysioExDataModule(
            datasets=[dataset],
            **datamodule_kwargs,
        )

        checkpoint_dir = (
            f"articles/ProtoEx-Sleep/models/debug/{model_name}/{dataset}/EEG-EOG-EMG/"
        )
        # list files
        checkpoint_files = os.listdir(checkpoint_dir)
        checkpoint_files = [f for f in checkpoint_files if f.endswith(".ckpt")]
        if not checkpoint_files:
            logger.warning(
                f"No checkpoint files found in {checkpoint_dir}. Skipping robustness test for {dataset}."
            )
            continue
        else:
            # take the first checkpoint file
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
            logger.info(f"Using checkpoint: {checkpoint_path}")

        model_instance = load_model(
            model=model,
            model_kwargs=model_kwargs,
            ckpt_path=checkpoint_path,
        )

        devices = find_usable_cuda_devices(-1)
        logger.info(f"Available devices: {devices}")

        my_logger = [
            TensorBoardLogger(save_dir=checkpoint_dir + "/test_logs/"),
            CSVLogger(save_dir=checkpoint_dir + "/test_logs/"),
        ]

        trainer = Trainer(
            devices=devices,
            strategy="ddp" if len(devices) > 1 else "auto",
            deterministic="warn",
            logger=my_logger,
            precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        )

        results = trainer.test(model_instance, datamodule=datamodule)[0]
        results["dataset"] = dataset
        results_df = pd.DataFrame([results])

        # read the test results if it exists
        if os.path.exists(os.path.join(checkpoint_dir, "test_results.csv")):
            existing_results = pd.read_csv(
                os.path.join(checkpoint_dir, "test_results.csv")
            )
            # append the new results to the existing results
            results_df = pd.concat([existing_results, results_df], ignore_index=True)

        results_df.to_csv(os.path.join(checkpoint_dir, "test_results.csv"), index=False)

    return


def main():
    set_float32_matmul_precision("medium")

    # test performances for all datasets
    for model_name, model in zip(models_names, models):
        logger.info(f"Testing model: {model_name}")
        # repeat the test 5 trials
        # to get a more robust estimate of the performance
        for i in tqdm(range(1), desc=f"Testing {model_name} performances"):
            seed_everything(42 + i, workers=True)  # set seed for reproducibility

            if i == 0:
                # remove the existing test_results.csv file if it exists
                # and the existing logs
                for dataset in datasets:
                    checkpoint_dir = f"articles/protosleepnet/models/debug/{model_name}/{dataset}/EEG-EOG-EMG/"
                    if os.path.exists(os.path.join(checkpoint_dir, "test_results.csv")):
                        os.remove(os.path.join(checkpoint_dir, "test_results.csv"))
                    if os.path.exists(os.path.join(checkpoint_dir, "test_logs")):
                        shutil.rmtree(os.path.join(checkpoint_dir, "test_logs"))

            test_performances(
                datasets=datasets,
                datamodule_kwargs=datamodule_kwargs,
                model_name=model_name,
                model=model,
                model_kwargs=model_kwargs,
            )


if __name__ == "__main__":
    main()
