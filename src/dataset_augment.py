import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit


def augment_iq(x, snr):
    c = x[0] + 1j * x[1]

    phase = np.random.uniform(-np.pi / 4, np.pi / 4)
    c *= np.exp(1j * phase)

    if np.random.rand() < 0.5:
        L = x.shape[1]
        freq_offset = np.random.uniform(-0.02, 0.02) 
        t = np.arange(L)
        c *= np.exp(1j * 2 * np.pi * freq_offset * t)

    if np.random.rand() < 0.5:
        shift = np.random.randint(-50, 50)
        c = np.roll(c, shift)

    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.9, 1.1)
        c *= scale

    noise_std = 0.01 * (1 - (snr / 30))
    noise = np.random.normal(0, noise_std, size=c.shape) + 1j * np.random.normal(0, noise_std, size=c.shape)
    c += noise

    if np.random.rand() < 0.3:
        L = x.shape[1]
        mask_len = np.random.randint(5, int(0.1 * L))
        start = np.random.randint(0, L - mask_len)
        c[start:start + mask_len] = 0

    x_aug = np.stack([np.real(c), np.imag(c)])
    return x_aug


def compute_constellation(x, bins=64, lim=1.5):
    I, Q = x[0], x[1]
    H, _, _ = np.histogram2d(I, Q, bins=bins,
                             range=[[-lim, lim], [-lim, lim]])
    return H[np.newaxis].astype(np.float32)


class RadioMLH5Dataset(Dataset):
    def __init__(self, h5_path, split='train', val_fraction=0.2,
                 random_seed=42, transform=None):
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f"HDF5 файл не найден: {h5_path}")
        self.h5_path = h5_path
        self.transform = transform
        with h5py.File(h5_path, 'r') as f:
            Y = f['Y'][:]
            Z = f['Z'][:].reshape(-1)
        all_y = np.argmax(Y, axis=1)
        all_snr = Z.astype(np.float32)
        N = len(all_y)
        splitter = StratifiedShuffleSplit(n_splits=1,
                                          test_size=val_fraction,
                                          random_state=random_seed)
        train_idx, val_idx = next(splitter.split(np.zeros(N), all_y))
        idx = train_idx if split == 'train' else val_idx
        self.indices = idx
        self.y = all_y[idx]
        self.snrs = all_snr[idx]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        real_idx = int(self.indices[index])
        with h5py.File(self.h5_path, 'r') as f:
            x = f['X'][real_idx]  
        iq = x.T  
        snr = float(self.snrs[index])
        label = int(self.y[index])
        iq_raw = iq.copy()
        iq_aug = self.transform(iq.copy(), snr) if self.transform else iq.copy()
        const = compute_constellation(iq_aug)
        return (
            torch.from_numpy(iq_raw).float(),
            torch.from_numpy(iq_aug).float(),
            torch.from_numpy(const).float(),
            torch.tensor([snr], dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )