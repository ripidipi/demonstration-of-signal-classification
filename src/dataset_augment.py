import pickle, numpy as np, torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class RadioMLDataset(Dataset):
    def __init__(self, path, split='train', val_fraction=0.2, augment=False, seed=42):
        with open(path, 'rb') as f:
            Xd = pickle.load(f, encoding='latin1')
        Xs, ys = [], []
        for (mod, snr), data in Xd.items():
            Xs.append(data); ys += [mod]*len(data)
        X = np.vstack(Xs)
        le = LabelEncoder(); y = le.fit_transform(ys)
        Xtr, Xv, ytr, yv = train_test_split(
            X, y, test_size=val_fraction, stratify=y, random_state=seed
        )
        self.X, self.y = (Xtr, ytr) if split=='train' else (Xv, yv)
        self.augment = augment; np.random.seed(seed); self.le = le

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        x, y = self.X[i].copy(), self.y[i]
        if self.augment: x = self._augment(x)
        return torch.from_numpy(x).float(), y

    def _augment(self, x):
        # Gaussian noise
        snr = np.random.uniform(0,20)
        p_sig = x.var(); p_no = p_sig/10**(snr/10)
        x = x + np.sqrt(p_no)*np.random.randn(*x.shape)
        # random shift
        if np.random.rand()<0.5:
            x = np.roll(x, np.random.randint(x.shape[1]), axis=1)
        # random scale
        if np.random.rand()<0.5:
            x *= np.random.uniform(0.7,1.3)
        # time-mask
        if np.random.rand()<0.2:
            L = x.shape[1]; m = np.random.randint(L//20, L//5)
            s = np.random.randint(0, L-m); x[:,s:s+m]=0
        # freq-mask
        if np.random.rand()<0.2:
            ch = np.random.randint(x.shape[0]); x[ch,:]=0
        return x
