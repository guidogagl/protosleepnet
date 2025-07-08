import os
import time
from abc import ABC, abstractmethod
from typing import List, Callable

import h5py as h5
import numpy as np
import pandas as pd
import torch
from loguru import logger

from tqdm import tqdm

from physioex.data import PhysioExDataset

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


class GroupDataset(PhysioExDataset):
    def __init__(
        self,
        datasets: List[str],
        data_folder: str,
        preprocessing: str = "raw",
        selected_channels: List[int] = ["EEG"],
        sequence_length: int = 21,
        target_transform: Callable = None,
        hpc: bool = False,
        indexed_channels: List[int] = ["EEG", "EOG", "EMG", "ECG"],
        task: str = "sleep",
    ):

        if datasets == ["alzheimers"]:
            healthy = "alzheimers/HOA"
            unhealthy = "alzheimers/AD"
        elif datasets == ["parkinsons"]:
            healthy = "parkinsons/night/HOA"
            unhealthy = "parkinsons/night/PD"
        else:
            raise ValueError(
                "Unsupported dataset. Choose either 'alzheimers' or 'parkinsons'."
            )

        self.L = sequence_length

        # first fetch the healthy dataset
        dataset = PhysioExDataset(
            datasets=[healthy],
            data_folder=data_folder,
            preprocessing=preprocessing,
            selected_channels=selected_channels,
            sequence_length=-1,  # we read the entire night for each recording
            target_transform=target_transform,
            hpc=hpc,
            indexed_channels=indexed_channels,
            task=task,
        )

        self.X, self.y, self.groups = [], [], []
        mean = dataset.readers[0].reader.mean
        std = dataset.readers[0].reader.std

        for i, subject in enumerate(tqdm(dataset, desc="Loading healthy dataset")):
            X, y, subjects, dataset_idx = subject

            # invert scale the data
            X = (X * std) + mean

            self.X.append(X)
            self.y.append(y)
            self.groups.append(0)  # healthy group is 0

    

    

        # now fetch the unhealthy dataset
        dataset = PhysioExDataset(
            datasets=[unhealthy],
            data_folder=data_folder,
            preprocessing=preprocessing,
            selected_channels=selected_channels,
            sequence_length=-1,  # we read the entire night for each recording
            target_transform=target_transform,
            hpc=hpc,
            indexed_channels=indexed_channels,
            task=task,
        )

        mean = dataset.readers[0].reader.mean
        std = dataset.readers[0].reader.std
        
        for i, subject in enumerate(tqdm(dataset, desc="Loading unhealthy dataset")):
            X, y, subjects, dataset_idx = subject

            # invert scale the data
            X = (X * std) + mean

            self.X.append(X)
            self.y.append(y)
            self.groups.append(1)

        self.groups = torch.tensor(self.groups).long()

        weights = compute_class_weight(
            'balanced',
            classes=np.unique( self.groups.numpy() ),
            y=self.groups.numpy()
        )

        self.weights = torch.tensor(weights, dtype=torch.float32)

        # to be set by the set_num_folds method
        self.num_folds = -1

        # none until we call the "split" method
        self.len = -1

    def __len__(self):
        return self.len

    def split(self, fold: int = -1, dataset_idx: int = -1):

        if self.num_folds == -1:
            raise ValueError(
                "Number of folds is not set. You must call set_num_folds() before splitting the dataset."
            )
        if fold == -1:
            raise ValueError(
                "Fold is not set. You must specify a fold to split the dataset."
            )
        if fold >= self.num_folds:
            raise ValueError(
                f"Fold {fold} is out of range. The dataset has {self.num_folds} folds."
            )

        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=42)

        for i, (train_idx, test_idx) in enumerate(skf.split(self.X, self.groups)):
            if i == fold:
                break
        # now we need to split the train_idx into train and validation sets
        train_idx, valid_idx = train_test_split(
            train_idx, test_size=0.2, random_state=42, stratify=self.groups[train_idx]
        )

        # now we need to fetch the train dataset in sequences
        self.X_train, self.y_train, self.group_train = [], [], []

        for idx in tqdm(train_idx, desc="Splitting train set into sequences"):
            X = self.X[idx]
            y = self.y[idx]
            group = self.groups[idx] * torch.ones_like(y)

            self.X_train.append(X)
            self.y_train.append(y)
            self.group_train.append(group)

        # now concatenate the train set
        X_train = torch.cat(self.X_train, dim=0)
        #y_train = torch.cat(self.y_train, dim=0)
        #group_train = torch.cat(self.group_train, dim=0)
        self.mean, self.std = X_train.mean(dim=0), X_train.std(dim=0)

        if self.L == -1:
            for idx in range(len(self.X_train)):
                self.X_train[idx] = (self.X_train[idx] - self.mean) / self.std
        else:
            self.X_train = (X_train - self.mean) / self.std
            self.y_train = torch.cat(self.y_train, dim=0)
            self.group_train = torch.cat(self.group_train, dim=0)
        
        # scale validation and test set
        self.X_valid, self.y_valid, self.group_valid = [], [], []
        for idx in tqdm(valid_idx, desc="Splitting validation set into sequences"):
            X = self.X[idx]
            y = self.y[idx]
            group = self.groups[idx]

            X = (X - self.mean) / self.std

            self.X_valid.append(X)
            self.y_valid.append(y)
            self.group_valid.append(group)

        self.X_test, self.y_test, self.group_test = [], [], []
        for idx in tqdm(test_idx, desc="Splitting test set into sequences"):
            X = self.X[idx]
            y = self.y[idx]
            group = self.groups[idx]

            X = (X - self.mean) / self.std

            self.X_test.append(X)
            self.y_test.append(y)
            self.group_test.append(group)

        
        train_idx = np.arange(len(self.X_train) - self.L + 1)
        valid_idx = np.arange(len(train_idx), len(train_idx) + len(self.X_valid))
        test_idx = np.arange(
            len(train_idx) + len(valid_idx),
            len(train_idx) + len(valid_idx) + len(self.X_test),
        )

        self.valid_offset = len(train_idx)
        self.test_offset = len(valid_idx) + len(train_idx)

        self.train_idx, self.valid_idx, self.test_idx = (
            torch.tensor(train_idx).long(),
            torch.tensor(valid_idx).long(),
            torch.tensor(test_idx).long(),
        )

        self.len = len(self.train_idx) + len(self.valid_idx) + len(self.test_idx)

        print(
            f"Dataset split into {len(train_idx)} train, {len(valid_idx)} valid, and {len(test_idx)} test sequences."
        )

    def group_weights(self):
        return self.weights

    def set_num_folds(self, num_folds: int):
        self.num_folds = num_folds

    def get_num_folds(self):
        return self.num_folds

    def __getitem__(self, idx):

        if idx < 0 or idx >= self.len:
            raise IndexError(f"Index {idx} out of range. Dataset length is {self.len}.")
        if idx < self.valid_offset:
            if self.L == -1:
                X = self.X_train[idx]
                y = self.y_train[idx]
                group = self.group_train[idx]
            else:
                X = self.X_train[idx : idx + self.L]
                y = self.y_train[idx : idx + self.L]
                group = self.group_train[idx : idx + self.L]
        elif idx < self.test_offset:
            X = self.X_valid[idx - self.valid_offset]
            y = self.y_valid[idx - self.valid_offset]
            group = self.group_valid[idx - self.valid_offset]
        else:
            X = self.X_test[idx - self.test_offset]
            y = self.y_test[idx - self.test_offset]
            group = self.group_test[idx - self.test_offset]

        # as group take the most-occurring group in the sequence
        group = group.mode()[0]

        return X, y, torch.tensor(group).long(), torch.tensor(0).long()

    def get_sets(self):
        return self.train_idx, self.valid_idx, self.test_idx
