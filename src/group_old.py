
import os
from tqdm import tqdm

from physioex.train.utils.fast_train import FastEvalDataset

from physioex.data import PhysioExDataset, PhysioExDataModule

from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import torch.optim as optim
import torchmetrics as tm

from physioex.train.networks.seqsleepnet import AttentionLayer

device = "cuda" if torch.cuda.is_available() else "cpu"

class FastTrainDataset(Dataset):
    def __init__(self, X, y, p):
        
        self.X = torch.cat(X, dim=0)
        self.y = torch.cat(y, dim=0)
        self.p = torch.cat(p, dim=0)
        print(f"Dataset size: {self.X.shape}, Labels size: {self.y.shape}, Prototypes size: {self.p.shape}")
            
    def __len__(self):
        return len(self.y) 

    def __getitem__(self, idx):
        return self.X[idx].float(), self.y[idx].long(), self.p[idx].long()

class FastEvalDataset(Dataset):
    def __init__(self, X, y, p):
        
        self.X = X
        self.y = y
        self.p = p
        print(f"Dataset size: {len(X)}, Labels size: {len(y)}, Prototypes size: {len(p)}")

    def __len__(self):
        return len(self.y) 

    def __getitem__(self, idx):
        return self.X[idx].float(), self.y[idx].long(), self.p[idx].long()

# pytorch lightning module
class GroupModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # LSTM layer for sequence processing
        
        self.sequence_encoder = nn.LSTM(
            input_size=128,  # Assuming the input feature size is 128
            hidden_size=128,  # Size of the hidden state
            num_layers=4,
            bidirectional=True,# Number of LSTM layers
            batch_first=True, # Input shape is (batch, seq_len, features)
        )
        
        self.attention = AttentionLayer(128*2, 128)
        self.clf = nn.Linear(128*2, 2)  
        
        self.acc = tm.Accuracy(
                task="multiclass", num_classes=2, average="weighted"
        )
        
        self.f1 = tm.F1Score(
            task="multiclass", num_classes=2, average="weighted"
        )
        
        self.pr = tm.Precision(
            task="multiclass", num_classes=2, average="weighted"
        )
        self.rc = tm.Recall(
            task="multiclass", num_classes=2, average="weighted"
        )
        
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x, _ = self.sequence_encoder(x)
        x = self.attention(x)
        return self.clf(x)
    
    def compute_loss(self, y_hat, y, split="train"):
        
        loss = self.loss(y_hat, y)
        self.log(f"{split}_loss", loss, prog_bar=True)
        self.log(f"{split}_acc", self.acc(y_hat, y), prog_bar=True)

        self.log(f"{split}_f1", self.f1(y_hat, y), prog_bar=True)
        self.log(f"{split}_pr", self.pr(y_hat, y), prog_bar=True)
        self.log(f"{split}_rc", self.rc(y_hat, y), prog_bar=True)        
        return loss

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y, split="train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch        
        x, y = x.squeeze(0), y.squeeze(0) 
        
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y, split="eval")

        y_hat = y_hat.mean( dim = 0 ).reshape(1, 2) # N, C
        y = y[0].reshape(1) # N

        sequence_acc = self.acc(y_hat, y)
        self.log( "eval_sequence_acc", sequence_acc, prog_bar=True )       

        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        x, y = x.squeeze(0), y.squeeze(0) 
        
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y, split="test")

        y_hat = y_hat.mean( dim = 0 ).reshape(1, 2) # N, C
        y = y[0].reshape(1) # N

        sequence_acc = self.acc(y_hat, y)
        self.log( "test_sequence_acc", sequence_acc, prog_bar=True ) 

        return loss
    
    def configure_optimizers(self):
        # Definisci il tuo ottimizzatore
        self.opt = optim.Adam(
            self.parameters(),
            lr=1e-4,  # Learning rate
            weight_decay=1e-5,  # Weight decay (L2 regularization)
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
            "monitor": "eval_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [self.opt], [scheduler]

def get_datasets( DATASET ):
    if DATASET == "alzheimers":
        return ["alzheimers/HOA", "alzheimers/AD"]
    elif DATASET == "parkinsons":
        return ["parkinsons/night/HOA", "parkinsons/night/PD"]
    else:
        raise ValueError(f"Unknown dataset: {DATASET}. Supported datasets are 'alzheimers' and 'parkinsons'.")

def load_finetuned_model( dataset : str ):
    from physioex.train.models import load_model

    model_kwargs = {
        "in_channels": 3,
        "sequence_length": 21,
        "N" : 1,
        "S" : 2,
        "n_prototypes" : 50,
    }

    proto = load_model(
        model = "physioex.train.networks.prototypev1:ProtoSleepNetV1",
        model_kwargs = model_kwargs,
        ckpt_path = f"models/finetune/protosleepnetv1/{dataset}/EEG-EOG-EMG/model.ckpt",
        softmax = True,
        summary = False,
    ).eval()

    return proto[0]

def compress_eval_dataset(model, loader, start_mean, start_std, final_mean, final_std, label, inputs, targets, prototypes):

    L = 21
    nchan = 3
    
    for batch in tqdm(loader, desc=f"Compressing dataset"):
        X, _, _, _ = batch
        X = X.float().squeeze()
        
        # invert scale the data
        X = (X * start_std) + start_mean
        X = (X - final_mean) / final_std
        
        X = X.to(device)

        batch_inputs, batch_targets, batch_indexes = [], [], []
        with torch.no_grad():            
            for start in range( 0, X.shape[0] - L):
                x = X[start:start + L].unsqueeze(0)
                
                _, p, indexes, _, _ = model.nn.get_prototypes( x ) # batch, L, nchan, N, 128

                p = p.mean( dim = 3 ).reshape( 1, L, nchan, 128).permute( 0, 2, 1, 3)
                p = p.reshape( -1, L, 128 )        
        
                p = model.nn.sequence_encoder.encode( p ) # out -1, L, 128

                p = p.reshape( 1, nchan, L, 128).permute( 0, 2, 1, 3).reshape( -1, nchan, 128)
                
                indexes = indexes.reshape( -1, nchan, 1).float()
                p, alphas = model.nn.channels_sampler( p ) 

                indexes = torch.einsum("bns, bsh -> bnh", alphas, indexes )

                indexes = indexes.reshape( 1, L, 1 ).detach().cpu().long()                
                p = p.reshape( 1, L, 128 ).detach().cpu()

                batch_inputs.append( p )
                batch_targets.append( label ) 
                batch_indexes.append( indexes )
                
        batch_inputs = torch.cat( batch_inputs, dim = 0 )
        batch_targets = torch.tensor( batch_targets ).long().reshape( -1 )
        batch_indexes = torch.cat( batch_indexes, dim = 0 )
        
        inputs.append( batch_inputs )
        targets.append( batch_targets )
        prototypes.append( batch_indexes )

    return

def prepare_group_dataset( dataset : str ):

    model = load_finetuned_model( dataset)
    datasets = get_datasets( dataset )
    
    X, y, p = [], [], []

    final_mean, final_std = torch.load( f"{os.environ['VSC_SCRATCH_PROJECTS_BASE']}/2024_111/guido/.tmp/{DATASET}/xsleepnet/scaling.pt") 

    for i, dataset in enumerate( datasets ):
        data = PhysioExDataModule(
            datasets = [dataset],
            batch_size = 1,
            preprocessing = "xsleepnet",
            selected_channels = ["EEG", "EOG", "EMG"],
            sequence_length = -1,
            data_folder = f"{os.environ['VSC_SCRATCH_PROJECTS_BASE']}/2024_111/guido/",
            num_workers= os.cpu_count(),
        )
        scaling = data.dataset.readers[0].reader
        start_mean, start_std = scaling.mean, scaling.std

        train_loader = data.train_dataloader()
        eval_loader = data.val_dataloader()
        test_loader = data.test_dataloader()

        compress_eval_dataset(
            model,
            train_loader,
            start_mean, start_std, final_mean, final_std,
            label=i,
            inputs=X,
            targets=y,
            prototypes=p
        )

        compress_eval_dataset(
            model,
            eval_loader,
            start_mean, start_std, final_mean, final_std,
            label=i,
            inputs=X,
            targets=y,
            prototypes=p
        )

        compress_eval_dataset(
            model,
            test_loader,
            start_mean, start_std, final_mean, final_std,
            label=i,
            inputs=X,
            targets=y,
            prototypes=p
        )

    torch.save( (X, y, p), f"{os.environ['VSC_SCRATCH_PROJECTS_BASE']}/2024_111/guido/group/{dataset}/dataset.pt" )
    print("Datasets saved successfully.")
