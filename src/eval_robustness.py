from pathlib import Path
from typing import List, Union
import os
import types

import pandas as pd
import torch
from lightning.pytorch import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch import set_float32_matmul_precision
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
)

from physioex.data import PhysioExDataModule
from physioex.train.models.load import load_model
from physioex.train.networks.base import SleepModule

from loguru import logger

from lightning.pytorch.accelerators import find_usable_cuda_devices
import shutil


datasets = ["hmc", "sleepedf", "dcsm", "mass"]

model_kwargs = {"in_channels": 3, "sequence_length": 21, "weights": [None, None]}

datamodule_kwargs = {
    "batch_size": 1,
    "preprocessing": "xsleepnet",
    "selected_channels": ["EEG", "EOG", "EMG"],
    "sequence_length": 21,
    "data_folder": "/scratch/leuven/365/vsc36564/",
    "num_nodes": 1,
    "num_workers": 18,
}

models_names = [
    #"seqsleepnet",
    #"protoseqsleepnet",
    #"protoseqsleepnet.1",
    #"sleeptransformer",
    "protosleeptransformer",
    "protosleeptransformer.1",
]

models = [
    #"physioex.train.networks.seqsleepnet:SeqSleepNet",
    #"physioex.train.networks.protoseqsleepnet:ProtoSeqSleepNet",
    #"physioex.train.networks.protoseqsleepnet:ProtoSeqSleepNet",
    #"physioex.train.networks.sleeptransformer:SleepTransformer",
    "physioex.train.networks.protosleeptransformer:ProtoSleepTransformerNet",
    "physioex.train.networks.protosleeptransformer:ProtoSleepTransformerNet",
]

models_kwargs = [
    #{
    #    "in_channels": 3,
    #    "sequence_length": 21,
    #},
    #{"in_channels": 3, "sequence_length": 21, "weights": [0.75, 0.25]},
    #{"in_channels": 3, "sequence_length": 21, "weights": [1, 0]},
    #{
    #    "in_channels": 3,
    #    "sequence_length": 21,
    #},
    {"in_channels": 3, "sequence_length": 21, "weights": [0.75, 0.25]},
    {"in_channels": 3, "sequence_length": 21, "weights": [1, 0]},
]


def occlusion_maskv2(inputs: torch.Tensor):
    occlusion_mask = torch.zeros(
        inputs.shape[0], inputs.shape[1], inputs.shape[2], device=inputs.device
    )
    # randomly set channels to 1
    # batch_size is set to 1 so we can avoid it
    for i in range(inputs.shape[0]):
        num_channels_to_occlude = torch.randint(1, 3, (1,)).item()
        channels_to_occlude = torch.randperm(inputs.shape[2])[:num_channels_to_occlude]
        occlusion_mask[i, :, channels_to_occlude] = 1
    return occlusion_mask


def test_robustness(
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
            f"articles/protosleepnet/models/debug/{model_name}/{dataset}/EEG-EOG-EMG/"
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

        # we need to redefine the test_step method to use the occlusion mask
        # first backup the original test_step method
        original_test_step = model_instance.test_step

        def test_step(self, batch, batch_idx):
            inputs, targets, subjects, dataset_idx = batch

            # create occlusion mask
            occlusion_mask = occlusion_maskv2(inputs)
            inputs = torch.einsum("bsctf, bsc -> bsctf", inputs, occlusion_mask)

            batch = (inputs, targets, subjects, dataset_idx)
            return original_test_step(batch, batch_idx)

        # redefine the model test_step
        model_instance.test_step = types.MethodType(test_step, model_instance)

        devices = find_usable_cuda_devices(-1)
        logger.info(f"Available devices: {devices}")

        my_logger = [
            TensorBoardLogger(save_dir=checkpoint_dir + "/rob_logs/"),
            CSVLogger(save_dir=checkpoint_dir + "/rob_logs/"),
        ]

        trainer = Trainer(
            devices=devices,
            strategy="ddp" if len(devices) > 1 else "auto",
            num_nodes=datamodule_kwargs["num_nodes"],
            deterministic=True,
            logger=my_logger,
        )

        results = trainer.test(model_instance, datamodule=datamodule)[0]
        results["dataset"] = dataset
        results_df = pd.DataFrame([results])

        # read the test results if it exists
        if os.path.exists(os.path.join(checkpoint_dir, "robustness.csv")):
            existing_results = pd.read_csv(
                os.path.join(checkpoint_dir, "robustness.csv")
            )
            # append the new results to the existing results
            results_df = pd.concat([existing_results, results_df], ignore_index=True)

        results_df.to_csv(os.path.join(checkpoint_dir, "robustness.csv"), index=False)

    return


def main():

    set_float32_matmul_precision("medium")

    # test robustness for all datasets
    for model_name, model, model_kwargs in zip(models_names, models, models_kwargs):
        logger.info(f"Testing robustness for model: {model_name}")
        # repeat the test 5 trials
        # to get a more robust estimate of the performance
        for i in tqdm(range(5), desc=f"Testing {model_name} robustness"):

            seed_everything(42 + i, workers=True)  # set seed for reproducibility

            if i == 0:
                # remove the existing robustness.csv file if it exists
                # and the existing logs
                for dataset in datasets:
                    checkpoint_dir = f"articles/protosleepnet/models/debug/{model_name}/{dataset}/EEG-EOG-EMG/"
                    if os.path.exists(os.path.join(checkpoint_dir, "robustness.csv")):
                        os.remove(os.path.join(checkpoint_dir, "robustness.csv"))
                    if os.path.exists(os.path.join(checkpoint_dir, "rob_logs")):
                        shutil.rmtree(os.path.join(checkpoint_dir, "rob_logs"))

            test_robustness(
                datasets=datasets,
                datamodule_kwargs=datamodule_kwargs,
                model_name=model_name,
                model=model,
                model_kwargs=model_kwargs,
            )


if __name__ == "__main__":
    main()
