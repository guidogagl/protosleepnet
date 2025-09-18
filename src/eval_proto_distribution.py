import os

os.chdir( "/data/leuven/365/vsc36564/physioex/articles/ProtoEx-Sleep" )

DATA_FOLDER = os.path.join( os.environ["VSC_SCRATCH_PROJECTS_BASE"], "2024_111", "guido" )

# tier2 ugent
#DATA_FOLDER = os.environ["VSC_SCRATCH"]

from physioex.data import PhysioExDataModule
from physioex.train.models import load_model
from physioex.train.networks.prototype import voting_strategy

from sklearn.metrics import accuracy_score

from tqdm import tqdm
import pandas as pd

import numpy as np
import torch.nn as nn
import torch

import pytorch_lightning as pl

pl.seed_everything(42)

def compute_density(codebook, P, y_hat ):
    P_dens = torch.zeros( codebook.shape[0] )
    P_hat = torch.zeros( codebook.shape[0], 5 )

    for i in range(P.shape[0]):
        distances = torch.cdist(P[i].view(1, -1), codebook, p=2).view(-1)
        closest_index = torch.argmin(distances)
        P_dens[closest_index] += 1
        P_hat[closest_index] += y_hat[i].view(-1)

    P_hat = P_hat / P_dens.view(-1, 1)
    P_dens = P_dens / P_dens.sum()

    P_dens[torch.isnan(P_dens)] = 0  # handle division by zero
    P_hat[torch.isnan(P_hat)] = 0  # handle division by zero

    return P_dens, P_hat

def compute_distance(codebook, P, embeddings):
    E_dists = torch.zeros( codebook.shape[0] )
    E_counts = torch.zeros( codebook.shape[0] )
    for i in range(P.shape[0]):
        distances = torch.cdist(embeddings[i].view(1, -1), codebook, p=2).view(-1)
        closest_index = torch.argmin(distances)
        E_dists[closest_index] += torch.norm(embeddings[i] - codebook[closest_index])
        E_counts[closest_index] += 1

    E_dists = E_dists / E_counts
    E_dists[torch.isnan(E_dists)] = 0  # handle division by zero

    return E_dists

def compute_prototype_features( model, model_kwargs, loader, device="cpu" ):
    codebook = model.nn.prototype.codebook.detach().cpu()

    P_features = []
    Subj_acc = []

    model = model.to(device)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing prototype features", total=len(loader)):
            x, y, _, _ = batch

            x = x.to(device)

            y_hat = torch.zeros( x.shape[1], 5 )
            P = torch.zeros( y_hat.shape[0], codebook.shape[1] )
            alphas = torch.zeros( y_hat.shape[0], 3 )
            embeddings = torch.zeros_like( P )

            counts = torch.zeros( y_hat.shape[0] )
            
            # Optimized sliding window processing with batching
            seq_len = model_kwargs["sequence_length"]
            total_windows = y_hat.shape[0] - seq_len + 1
            
            # Process windows in batches by stride position
            for stride in range(seq_len):
                # Collect all windows that start at positions with the same stride modulo seq_len
                start_positions = list(range(stride, total_windows, seq_len))
                if not start_positions:
                    continue
                    
                # Create batch of windows
                batch_windows = []
                for start_pos in start_positions:
                    batch_windows.append(x[:, start_pos:start_pos+seq_len])
                
                if batch_windows:
                    # Stack windows into batch dimension
                    x_batch = torch.cat(batch_windows, dim=0)  # (n_windows, seq_len, n_channels, time, freq)
                    
                    # Process the entire batch at once
                    P_batch, alphas_batch = model.nn.prototyping(x_batch)
                    P_batch, alphas_batch = P_batch.detach().cpu(), alphas_batch.detach().cpu()
                    
                    embeddings_batch = model.nn.proto_encoder(x_batch).detach().cpu().reshape( x_batch.shape[0],   x_batch.shape[1],  x_batch.shape[2],  -1)
                    y_hat_batch = torch.nn.functional.softmax(model(x_batch), dim=-1).detach().cpu()
                    # Pre-compute weighted embeddings for the entire batch
                    weighted_embeddings_batch = torch.einsum("bsc, bsch->bsh", alphas_batch, embeddings_batch)
                    
                    # Distribute results back to their positions
                    for idx, start_pos in enumerate(start_positions):
                        P_i = P_batch[idx]
                        alphas_i = alphas_batch[idx] 
                        weighted_embeddings_i = weighted_embeddings_batch[idx]
                        y_hat_i = y_hat_batch[idx]
                        
                        P[start_pos:start_pos+seq_len] += P_i
                        alphas[start_pos:start_pos+seq_len] += alphas_i
                        embeddings[start_pos:start_pos+seq_len] += weighted_embeddings_i
                        y_hat[start_pos:start_pos+seq_len] += y_hat_i
                        
                        counts[start_pos:start_pos+seq_len] += 1

            # normalize the embeddings
            embeddings = embeddings / counts.view(-1, 1)
            alphas = alphas / counts.view(-1, 1)
            P = P / counts.view(-1, 1)
            y_hat = y_hat.squeeze().detach().cpu().reshape( -1, 5 ) / counts.view(-1, 1)

            x = x.cpu()

            # compute model accuracy
            acc = accuracy_score(
                y.cpu().numpy().flatten(),
                y_hat.argmax(dim=-1).numpy().flatten()
            )
            Subj_acc.append(acc)

            # compute the prototype features.
            #  --- Prototype density
            #  --- Embedding - Prototype distance

            P_dens, P_hat = compute_density(codebook, P, y_hat)
            E_dists = compute_distance(codebook, P, embeddings)

            P_features.append(
                np.concatenate(
                    [
                        P_dens.numpy(),
                        P_hat.numpy().flatten(),
                        E_dists.numpy(),
                    ]
                ).reshape(1, -1)
            )
    
    model = model.to("cpu")
    
    P_features = np.concatenate(P_features, axis=0)
    Subj_acc = np.array(Subj_acc).reshape(-1)

    return P_features, Subj_acc

DATASET = "sleepedf"

model_name = "protosleeptransformer"
model_class = "physioex.train.networks.protosleeptransformer:ProtoSleepTransformerNet"

#encoder_name = "protoseqsleepnet"
#encoder_class = "physioex.train.networks.protoseqsleepnet:ProtoSeqSleepNet"

model_kwargs = {
    "in_channels": 3,
    "sequence_length": 21,
    "weights": [.75, .25],
}

device = "cuda" if torch.cuda.is_available() else "cpu"


ckp_path = f"models/{model_name}/{DATASET}/EEG-EOG-EMG/"
ckp_file = [f for f in os.listdir(ckp_path) if f.endswith(".ckpt")][0]


model = load_model(
    model=model_class,
    model_kwargs=model_kwargs,
    device="cpu",
    ckpt_path=os.path.join(ckp_path, ckp_file),
    softmax=False,
    summary=False,
).eval()


datamodule = PhysioExDataModule(
    datasets=[DATASET],
    batch_size=1,
    selected_channels=["EEG", "EOG", "EMG"],
    sequence_length=-1,
    preprocessing="xsleepnet",
    data_folder=DATA_FOLDER,
    num_workers=1,  # use half of the available CPU cores
)

codebook = model.nn.prototype.codebook.detach().cpu() 

print(f"Computing train prototype features for {model_name} on {DATASET} dataset...")
# compute the prototype features on the training set
P_train, Acc_train = compute_prototype_features(
    model=model,
    model_kwargs=model_kwargs,
    loader=datamodule.train_dataloader(),
    device=device
)

print(f"Computing test prototype features for {model_name} on {DATASET} dataset...")
P_test, Acc_test = compute_prototype_features(
    model=model,
    model_kwargs=model_kwargs,
    loader=datamodule.test_dataloader(),
    device=device
)

n_train = P_train.shape[0]
n_test = P_test.shape[0]

n_features = P_train.shape[1]

# for each test subject, compute the mean distance to the training prototypes
# P_test - P_train distance

print(f"Computing train-test distances for {model_name} on {DATASET} dataset...")

P_train = torch.tensor(P_train, dtype=torch.float32).reshape(1, n_train, n_features )
P_test = torch.tensor(P_test, dtype=torch.float32).reshape( 1, n_test, n_features )

train_test_distances = torch.cdist(P_test, P_train, p=2).squeeze().cpu().reshape(n_test, n_train).numpy()

# extimate the test_accuracy by the accuracy of the nearest training subject
Estimate_Acc = np.zeros(n_test)
for i in range(n_test):
    closest_index = np.argmin(train_test_distances[i])
    Estimate_Acc[i] = Acc_train[closest_index]

Dist_test = train_test_distances.mean(axis=1)

# save the results on a dataframe with test_subject, train-test distance, and accuracy
df = {
    "test_subject": np.arange(1, n_test + 1),
    "train_test_distance": Dist_test,
    "estimated_accuracy": Estimate_Acc,
    "test_accuracy": Acc_test,
}

df = pd.DataFrame(df)

df.to_csv(f"results/{model_name}_{DATASET}_prototype_features.csv", index=False)