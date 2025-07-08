import os
from tqdm import tqdm


import torch.nn as nn
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    DeviceStatsMonitor,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

import torch.optim as optim
import torchmetrics as tm

from physioex.train.networks.seqsleepnet import AttentionLayer


# pytorch lightning module
class GroupModule(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        input_size: int = 128,
        hidden_size: int = 128,
        num_layers: int = 4,
        attention_size: int = 128,
        class_weights: torch.Tensor | None = None,
        dropout_rate: float = 0.3,
        label_smoothing: float = 0.1,
    ):
        super().__init__()

        #self.encoder = encoder.eval()
        #for param in self.encoder.parameters():
        #    param.requires_grad = False
        
        self.encoder = encoder

        self.sequence_encoder = nn.LSTM(
            input_size=input_size,  # Assuming the input feature size is 128
            hidden_size=hidden_size,  # Size of the hidden state
            num_layers=num_layers,
            bidirectional=True,  # Number of LSTM layers
            batch_first=True,  # Input shape is (batch, seq_len, features)
            dropout=dropout_rate if num_layers > 1 else 0,
        )

        self.attention = AttentionLayer(hidden_size * 2, attention_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.clf = nn.Linear(hidden_size * 2, 2)

        self.acc = tm.Accuracy(task="multiclass", num_classes=2, average="weighted")

        # Use class weights and label smoothing to combat overfitting to majority class
        self.loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, x):
        # batch_size, seq_len, nchan, T, S  

        #with torch.no_grad():
        #    x = self.encoder.nn.project(x)

        x = self.encoder.nn.project(x)  # x shape: batch_size, seq_len, input_size

        # x shape: batch_size, seq_len, hidden_size
        x, _ = self.sequence_encoder(x)  # batch_size, seq_len, hidden_size *2
        
        batch, seq, hidden_size = x.shape

        if self.training:
            seq_y = self.clf( x.reshape( -1, hidden_size ) ).reshape(batch, seq, -1)  # batch_size, seq_len, 2
            
        x = self.attention(x)  # batch_size, hidden_size * 2
        x = self.dropout(x)  # Apply dropout before final classification

        y = self.clf(x)

        if self.training:
            return seq_y, y        

        return y

    def training_step(self, batch, batch_idx):
        x, _, y, _ = batch
        y_hat_seq, y_hat = self(x)

        batch_size, seq_len, _ = y_hat_seq.shape

        y_hat = y_hat.reshape(-1, 2)  # batch_size, 2
        y_hat_seq = y_hat_seq.reshape(-1, 2)  # batch_size * seq_len, 2

        y_seq = torch.einsum( "b,bs->bs", y, torch.ones( batch_size, seq_len, device=y.device ) )  # batch_size, seq_len
        y_seq = y_seq.reshape(-1).long()

        seq_loss = self.loss( y_hat_seq.reshape(-1, 2), y_seq.reshape(-1) )
        loss = self.loss( y_hat.reshape(-1, 2), y.reshape(-1) )

        loss = seq_loss + loss

        self.log("train_loss", loss, prog_bar=True )
        self.log("train_seq_acc", self.acc(y_hat_seq, y_seq), prog_bar=True)
        self.log("train_acc", self.acc(y_hat, y), prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, y, _ = batch  # 1, night_lenght, nchan, T, S
        
        night_length = x.shape[1]


        # reduce the night lenght to be divisible by 21
        if night_length % 21 != 0:
            night_length = night_length - (night_length % 21)
        
        x = x[:, :night_length, :, :, :]  # reduce the night length
        x = x.reshape(-1, 21, x.shape[2], x.shape[3], x.shape[4])  

        y_night = y * torch.ones(night_length // 21, dtype=y.dtype, device=y.device )
        y_night_hat = self(x).reshape(-1, 2)  # batch_size * (night_len - 21 + 1), 2

        y_night_hat = y_night_hat.float()
        y_night = y_night.long()

        loss = self.loss(y_night_hat, y_night)
    
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.acc(y_night_hat, y_night), prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, _, y, _ = batch  # 1, night_lenght, nchan, T, S
        
        night_length = x.shape[1]
        # reduce the night lenght to be divisible by 21
        if night_length % 21 != 0:
            night_length = night_length - (night_length % 21)
        
        x = x[:, :night_length, :, :, :]  # reduce the night length
        x = x.reshape(-1, 21, x.shape[2], x.shape[3], x.shape[4])  # batch_size * (night_len - 21 + 1), 21, nchan, T, S

        y_night = y * torch.ones(night_length // 21, dtype=y.dtype, device=y.device )
        y_night_hat = self(x).reshape(-1, 2)  # batch_size * (night_len - 21 + 1), 2

        y_night_hat = y_night_hat.float()
        y_night = y_night.long()

        y_hat = torch.nn.functional.softmax(y_night_hat, dim=-1).reshape(-1, 2).mean(dim=0)
    
        loss = self.loss(y_hat.view(1, 2).float(), y.view(1).long())

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.acc(y_hat, y), prog_bar=True)

        return loss

    def configure_optimizers(self):
        # Definisci il tuo ottimizzatore con learning rate più basso e weight decay più alto
        self.opt = optim.Adam(
            self.parameters(),
            lr=5e-4,  # Learning rate ridotto per evitare convergenza rapida a una classe
            weight_decay=1e-4,  # Weight decay aumentato per regolarizzazione
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode="min",
            factor=0.5,
            patience=1,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            # verbose=True,
        )
        scheduler = {
            "scheduler": scheduler,
            "name": "lr_scheduler",
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [self.opt], [scheduler]


VSC_DATA = os.environ.get("VSC_DATA", "/data/leuven/365/vsc36564")
#DATA_FOLDER = os.environ.get("VSC_SCRATCH", "/scratch/leuven/365/vsc36564")
DATA_FOLDER = os.path.join( os.environ["VSC_SCRATCH_PROJECTS_BASE"], "2024_111", "guido" )

import os

os.chdir(os.path.join(VSC_DATA, "physioex"))

import sys

sys.path.append(
    os.path.join(VSC_DATA, "physioex", "articles", "protosleepnet", "scripts", "src")
)

from staging_util import GroupDataset

from physioex.data import PhysioExDataModule
from physioex.train.utils import finetune, test
from physioex.train.models import load_model

import argparse

group_model_kwargs = {
    "input_size": 128,
    "hidden_size": 128,
    "num_layers": 4,
    "attention_size": 128,
}


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Group classification staging script")
    parser.add_argument(
        "--GROUP",
        type=str,
        default="alzheimers",
        help="Path to VSC data directory",
    )
    parser.add_argument(
        "--FOLD",
        type=int,
        default=0,
        help="Fold number for cross-validation",
    )
    args = parser.parse_args()
    GROUP = args.GROUP
    FOLD = args.FOLD

    print(f"Running staging script for group: {GROUP}, fold: {FOLD}")

    encoder_name = "protosleeptransformer"
    encoder_class = "physioex.train.networks.protosleeptransformer:ProtoSleepTransformerNet"

    #encoder_name = "protoseqsleepnet"
    #encoder_class = "physioex.train.networks.protoseqsleepnet:ProtoSeqSleepNet"


    num_folds = 4

    batch_size = 256
    num_nodes = 1
    max_epochs = 10


    gd = GroupDataset(
        datasets=[GROUP],
        data_folder=DATA_FOLDER,
        preprocessing="xsleepnet",
        selected_channels=["EEG", "EOG", "EMG"],
        sequence_length=21, # 21
    )

    gd.set_num_folds(num_folds)

    datamodule = PhysioExDataModule(
        datasets=gd,
        batch_size=batch_size,
        folds=FOLD,
        num_workers=18,
    )

    model_kwargs = {"in_channels": 3, "sequence_length": 21, "weights": [0.75, 0.25]}

    ckpt_path = f"articles/protosleepnet/models/debug/{encoder_name}/group/{GROUP}/staging/fold={FOLD}/"
    # list files inside
    files = [f for f in os.listdir(ckpt_path) if f.endswith(".ckpt")][0]
    ckpt_path = os.path.join(ckpt_path, files)

    encoder = load_model(
        model=encoder_class,
        model_kwargs=model_kwargs,
        ckpt_path=ckpt_path,
        device="cpu",
        softmax=False,
        summary=False,
    ).eval()

    group_model_kwargs["encoder"] = encoder
    group_model_kwargs["class_weights"] = gd.group_weights()

    model = GroupModule(
        **group_model_kwargs,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        save_top_k=1,
        mode="max",
        dirpath=f"articles/protosleepnet/models/debug/{encoder_name}/group/{GROUP}/groups/fold={FOLD}/",
        filename="{epoch}-{step}-{val_acc:.2%}",
        save_weights_only=False,
    )

    lr_callback = LearningRateMonitor(logging_interval="step")
    dvc_callback = DeviceStatsMonitor()

    # progress_bar_callback = RichProgressBar()
    my_logger = [
        TensorBoardLogger(
            save_dir=f"articles/protosleepnet/models/debug/{encoder_name}/group/{GROUP}/groups/fold={FOLD}/"
        ),
        CSVLogger(
            save_dir=f"articles/protosleepnet/models/debug/{encoder_name}/group/{GROUP}/groups/fold={FOLD}/"
        ),
    ]

    ########### Trainer Setup ############
    from lightning.pytorch.accelerators import find_usable_cuda_devices

    devices = find_usable_cuda_devices(-1)
    print(f"Available devices: {devices}")
    effective_batch_size = batch_size * num_nodes * len(devices)


    val_check_interval = 100

    if devices == "auto":
        strategy = "auto"
    elif num_nodes > 1 or len(devices) > 1:
        strategy = "ddp"
    else:
        strategy = "auto"

    trainer = pl.Trainer(
        devices=devices,
        strategy=strategy,
        num_nodes=num_nodes,
        max_epochs=max_epochs,
        val_check_interval=val_check_interval,
        callbacks=[
            checkpoint_callback,
            lr_callback,
            dvc_callback,
        ],  # , progress_bar_callback],
        deterministic=True,
        logger=my_logger,
    )

    # Start training
    trainer.fit(model, datamodule=datamodule)

    # load the best model
    best_checkpoint = checkpoint_callback.best_model_path
    print(f"Best checkpoint: {best_checkpoint}")

    model = GroupModule.load_from_checkpoint(
        best_checkpoint,
        **group_model_kwargs,
    ).eval()

    # test the model
    results = trainer.test(
        model=model,  # if passed model_class, model_config and resume are ignored
        datamodule=datamodule,
    )[0]

    import pandas as pd

    results_df = pd.DataFrame([results])
    results_df.to_csv(
        f"articles/protosleepnet/models/debug/{encoder_name}/group/{GROUP}/groups/fold={FOLD}/test_results.csv",
        index=False,
    )
