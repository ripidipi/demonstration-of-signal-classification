import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import os

from dataset_augment import RadioMLDataset

def load_data(path):
    abs_path = os.path.join(os.path.dirname(__file__), '..', path)
    with open(abs_path, 'rb') as f:
        Xd = pickle.load(f, encoding='latin1')

    X = []
    y = []

    for (mod, snr), data in Xd.items():
        X.append(data)
        y += [(mod, snr)] * data.shape[0]

    X = np.vstack(X)
    le = LabelEncoder()
    y = le.fit_transform([mod for mod, _ in y])
    return X, y, le


def get_dataloaders(path: str,
                    batch_size: int = 256,
                    val_fraction: float = 0.2,
                    num_workers: int = 4):
    """
    Возвращает train_loader, val_loader, label_encoder.
    """

    train_ds = RadioMLDataset(
        path,
        split='train',
        val_fraction=val_fraction,
        augment=True
    )
    val_ds = RadioMLDataset(
        path,
        split='val',
        val_fraction=val_fraction,
        augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_ds.le

