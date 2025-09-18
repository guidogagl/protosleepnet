import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics as tm
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger


# pytorch lightning module
class GroupModule(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        input_size: int = 128,
        hidden_size: int = 64,
        num_layers: int = 4,
        attention_size: int = 128,
        class_weights: torch.Tensor | None = None,
        dropout_rate: float = 0.3,
        label_smoothing: float = 0.1,
    ):
        super().__init__()

        # self.sequence_encoder = copy.deepcopy(encoder.nn.s_encoder)
        # self.sequence_encoder._initialize()

        self.sequence_encoder = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            num_layers=num_layers,
            batch_first=True,
        )

        self.encoder = encoder.nn.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.clf = nn.Linear(hidden_size * 2, 2)
        self.proto_clf = nn.Linear(input_size, 2)

        self.acc = tm.Accuracy(task="multiclass", num_classes=2, average="weighted")

        # Use class weights and label smoothing to combat overfitting to majority class
        self.loss = nn.CrossEntropyLoss(weight=class_weights)

    def f_ExS(self, x):
        # sequence encoding function for x
        # x shape : ( batch_size, seq_len, hidden_size )
        batch, L, hidden = x.size()

        # x = x + self.sequence_encoder(x)
        x, _ = self.sequence_encoder(x)

        return x.reshape(batch, L, hidden)

    def f_ExE(self, x):
        return self.encoder.f_ExE(x)

    def f_P(self, x):
        return self.encoder.f_P(x)

    def forward(self, x, return_proto=False):
        # batch_size, seq_len, nchan, T, S

        with torch.no_grad():
            x = self.f_ExE(x)
            x = self.f_P(x)

        batch, seqlen, hidden = x.shape

        proto_y = self.proto_clf(x.reshape(batch * seqlen, -1)).reshape(
            batch, seqlen, -1
        )

        x = self.f_ExS(x)

        y = self.clf(x.reshape(batch * seqlen, -1)).reshape(
            batch, seqlen, -1
        )  # (batch_size, seq_len, 2)

        if return_proto:
            return y, proto_y
        return y

    def training_step(self, batch, batch_idx):
        x, _, y, _ = batch
        y_hat, y_hat_proto = self(x, return_proto=True)

        batch_size, seq_len, _ = y_hat_proto.shape

        y_hat = y_hat.reshape(-1, 2)  # batch_size, 2
        y_hat_proto = y_hat_proto.reshape(-1, 2)  # batch_size * seq_len, 2

        y = torch.einsum(
            "b,bs->bs", y, torch.ones(batch_size, seq_len, device=y.device)
        )  # batch_size, seq_len
        y = y.reshape(-1).long()

        loss = self.loss(y_hat.reshape(-1, 2), y.reshape(-1))
        proto_loss = self.loss(y_hat_proto.reshape(-1, 2), y.reshape(-1))

        loss = loss  # + proto_loss

        self.log("train_loss", loss, prog_bar=False)

        acc = self.acc(y_hat, y)
        proto_acc = self.acc(y_hat_proto, y)

        self.log("train_acc", acc, prog_bar=True)
        self.log("train_p_acc", proto_acc, prog_bar=True)

        return loss

    def eval_on_night(self, inputs: torch.Tensor, L: int = 21):
        batch_size, night_length, n_channels, T, F = inputs.size()

        x = self.f_ExE(inputs)

        # prototyping
        p = self.f_P(x)
        proto_y = self.proto_clf(p.reshape(batch_size * night_length, -1)).reshape(
            batch_size, night_length, -1
        )

        y = self.f_ExS(x)
        y = self.clf(y.reshape(batch_size * night_length, -1)).reshape(
            batch_size, night_length, -1
        )

        return y, proto_y

    def validation_step(self, batch, batch_idx):
        x, _, y, _ = batch
        y_hat, y_hat_proto = self.eval_on_night(x)

        batch_size, seq_len, _ = y_hat_proto.shape

        y_hat = y_hat.reshape(-1, 2)  # batch_size, 2
        y_hat_proto = y_hat_proto.reshape(-1, 2)  # batch_size * seq_len, 2

        y = torch.einsum(
            "b,bs->bs", y, torch.ones(batch_size, seq_len, device=y.device)
        )  # batch_size, seq_len

        y = y.reshape(-1).long()

        loss = self.loss(y_hat.reshape(-1, 2), y.reshape(-1))
        proto_loss = self.loss(y_hat_proto.reshape(-1, 2), y.reshape(-1))

        loss = loss  # + proto_loss

        self.log("val_loss", loss, prog_bar=False)

        acc = self.acc(y_hat, y)
        proto_acc = self.acc(y_hat_proto, y)

        self.log("val_acc", acc, prog_bar=True)
        self.log("val_p_acc", proto_acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, _, y, _ = batch
        y_hat, y_hat_proto = self.eval_on_night(x)

        batch_size, seq_len, _ = y_hat_proto.shape

        y = y.reshape(-1).long()

        y_hat = torch.nn.functional.softmax(y_hat, dim=-1).mean(dim=1).reshape(-1, 2)
        y_hat_proto = (
            torch.nn.functional.softmax(y_hat_proto, dim=-1).mean(dim=1).reshape(-1, 2)
        )

        loss = self.loss(y_hat.reshape(-1, 2), y.reshape(-1))
        proto_loss = self.loss(y_hat_proto.reshape(-1, 2), y.reshape(-1))

        loss = loss + proto_loss

        self.log("test_loss", loss, prog_bar=False)

        acc = self.acc(y_hat, y)
        proto_acc = self.acc(y_hat_proto, y)

        self.log("test_acc", acc, prog_bar=True)
        self.log("test_p_acc", proto_acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        # Definisci il tuo ottimizzatore con learning rate più basso e weight decay più alto
        self.opt = optim.Adam(
            self.parameters(),
            lr=1e-4,  # Learning rate ridotto per evitare convergenza rapida a una classe
            weight_decay=1e-6,  # Weight decay aumentato per regolarizzazione
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
# DATA_FOLDER = os.environ.get("VSC_SCRATCH", "/scratch/leuven/365/vsc36564")
DATA_FOLDER = os.path.join(
    os.environ["VSC_SCRATCH_PROJECTS_BASE"], "2024_111", "guido", "sleep-data"
)

import os

os.chdir(os.path.join(VSC_DATA, "physioex"))

import sys

sys.path.append(os.path.join(VSC_DATA, "physioex", "articles", "ProtoEx-Sleep", "src"))

import argparse

from staging_util import GroupDataset

from physioex.data import PhysioExDataModule
from physioex.train.models import load_model

group_model_kwargs = {
    "input_size": 128,
    "hidden_size": 64,
    "num_layers": 4,
    "attention_size": 32,
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
        default=-1,
        help="Fold number for cross-validation",
    )
    args = parser.parse_args()
    GROUP = args.GROUP
    FOLD = args.FOLD

    print(f"Running staging script for group: {GROUP}, fold: {FOLD}")

    encoder_name = "protosleeptransformer"
    encoder_class = (
        "physioex.train.networks.protosleeptransformer:ProtoSleepTransformerNet"
    )

    # encoder_name = "protoseqsleepnet"
    # encoder_class = "physioex.train.networks.protoseqsleepnet:ProtoSeqSleepNet"

    num_folds = 4

    batch_size = 256
    num_nodes = 1
    max_epochs = 20

    def main(FOLD):
        gd = GroupDataset(
            datasets=[GROUP],
            data_folder=DATA_FOLDER,
            preprocessing="xsleepnet",
            selected_channels=["EEG", "EOG", "EMG"],
            sequence_length=21,  # 21
        )

        gd.set_num_folds(num_folds)

        datamodule = PhysioExDataModule(
            datasets=gd,
            batch_size=batch_size,
            folds=FOLD,
            num_workers=18,
            data_prefetch=False,
        )

        model_kwargs = {
            "in_channels": 3,
            "sequence_length": 21,
            "weights": [0.75, 0.25],
        }

        ckpt_path = f"articles/ProtoEx-Sleep/models/debug/{encoder_name}/group/{GROUP}/staging/fold={FOLD}/"
        # list files inside
        files = [f for f in os.listdir(ckpt_path) if f.endswith(".ckpt")][0]
        ckpt_path = os.path.join(ckpt_path, files)

        encoder = load_model(
            model=encoder_class,
            model_kwargs=model_kwargs,
            ckpt_path=ckpt_path,
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
            dirpath=f"articles/ProtoEx-Sleep/models/debug/{encoder_name}/group/{GROUP}/groups/fold={FOLD}/",
            filename="{epoch}-{step}-{val_acc:.2%}",
            save_weights_only=False,
        )

        lr_callback = LearningRateMonitor(logging_interval="step")
        dvc_callback = DeviceStatsMonitor()

        # progress_bar_callback = RichProgressBar()
        my_logger = [
            TensorBoardLogger(
                save_dir=f"articles/ProtoEx-Sleep/models/debug/{encoder_name}/group/{GROUP}/groups/fold={FOLD}/"
            ),
            CSVLogger(
                save_dir=f"articles/ProtoEx-Sleep/models/debug/{encoder_name}/group/{GROUP}/groups/fold={FOLD}/"
            ),
        ]

        ########### Trainer Setup ############
        from lightning.pytorch.accelerators import find_usable_cuda_devices

        devices = find_usable_cuda_devices(-1)
        print(f"Available devices: {devices}")
        effective_batch_size = batch_size * num_nodes * len(devices)

        val_check_interval = 40

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
            deterministic="warn",
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
            f"articles/ProtoEx-Sleep/models/debug/{encoder_name}/group/{GROUP}/groups/fold={FOLD}/test_results.csv",
            index=False,
        )

    if FOLD == -1:
        for fold in range(num_folds):
            print(f"Running fold {fold}...")
            main(fold)
    else:
        print(f"Running fold {FOLD}...")
        main(FOLD)
