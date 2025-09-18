# get the --VSC_DATA, --DATA_FOLDER --FOLD --GROUP cli argument
import argparse

parser = argparse.ArgumentParser(description="Group staging script for ProtoSleepNet.")
parser.add_argument(
    "--VSC_DATA", type=str, required=True, help="Path to VSC_DATA directory."
)
parser.add_argument(
    "--DATA_FOLDER", type=str, required=True, help="Path to data folder."
)
parser.add_argument("--FOLD", type=str, default="0", help="Fold number (default: 0).")
parser.add_argument(
    "--GROUP", type=str, default="alzheimers", help="Group name (default: alzheimers)."
)

args = parser.parse_args()

data_folder = args.DATA_FOLDER
vsc_data = args.VSC_DATA
fold = int(args.FOLD)
group = args.GROUP

NUM_FOLDS = 4

import os

os.chdir(os.path.join(vsc_data, "physioex"))

import sys

sys.path.append(os.path.join(vsc_data, "physioex", "articles", "ProtoEx-Sleep", "src"))

from staging_util import GroupDataset

from physioex.data import PhysioExDataModule
from physioex.train.models import load_model
from physioex.train.utils import finetune, test

print(f"Running staging script for group: {group}, fold: {fold}")

# model_name = "protosleeptransformer"
# model_class = "physioex.train.networks.protosleeptransformer:ProtoSleepTransformerNet"

model_name = "protoseqsleepnet"
model_class = "physioex.train.networks.protoseqsleepnet:ProtoSeqSleepNet"

# model_name = "seqsleepnet"
# model_class = "physioex.train.networks.seqsleepnet:SeqSleepNet"

batch_size = 256
num_nodes = 1
max_epoch = 100


gd = GroupDataset(
    datasets=[group],
    data_folder=data_folder,
    preprocessing="xsleepnet",
    selected_channels=["EEG", "EOG", "EMG"],
    sequence_length=21,
)

gd.set_num_folds(NUM_FOLDS)

dm = PhysioExDataModule(
    datasets=gd,
    batch_size=batch_size,
    folds=fold,
    num_workers=23,
    data_prefetch=False,
)

model_kwargs = {"in_channels": 3, "sequence_length": 21}

model = load_model(
    model=model_class,
    model_kwargs=model_kwargs,
    ckpt_path=f"articles/ProtoEx-Sleep/models/debug/{model_name}/shhs/EEG-EOG-EMG/model.ckpt",
    softmax=False,
    summary=False,
)

if model is None:
    raise ValueError(f"Model {model_class} not found or could not be loaded.")

train_kwargs = {
    "datasets": dm,
    "num_validations": 2,
    "max_epochs": max_epoch,
    "num_nodes": 1,
    "checkpoint_path": f"articles/ProtoEx-Sleep/models/debug/{model_name}/group/{group}/staging/fold={fold}/",
}

best_checkpoint = finetune(
    model=model,
    learning_rate=1e-6,  # if None not updated
    train_kwargs=train_kwargs,
)

model = load_model(
    model=model_class,
    model_kwargs=model_kwargs,
    ckpt_path=best_checkpoint,
    softmax=False,
    summary=False,
)

test(
    datasets=dm,
    model=model,  # if passed model_class, model_config and resume are ignored
    results_path=f"articles/ProtoEx-Sleep/models/debug/{model_name}/group/{group}/staging/fold={fold}/",
)
