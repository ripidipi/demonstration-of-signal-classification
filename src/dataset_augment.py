import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class RadioMLDataset(Dataset):
    def __init__(self, path, split='train', val_fraction=0.2,
                 augment=False, mixup_prob=0.5, random_seed=42):
        with open(path, 'rb') as f:
            Xd = pickle.load(f, encoding='latin1')

        X_list, y_list = [], []
        for (mod, snr), data in Xd.items():
            X_list.append(data)
            y_list.extend([mod] * data.shape[0])

        X = np.vstack(X_list)  
        le = LabelEncoder()
        y = le.fit_transform(y_list)  

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=val_fraction,
            stratify=y, random_state=random_seed
        )
        if split == 'train':
            self.X, self.y = X_tr, y_tr
        else:
            self.X, self.y = X_val, y_val

        self.le = le
        self.augment = augment
        self.mixup_prob = mixup_prob
        np.random.seed(random_seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].copy() 
        y = int(self.y[idx])
        if self.augment:
    
            snr_db = np.random.uniform(0, 20)
            p_sig = x.var(); p_no = p_sig / (10**(snr_db/10))
            x += np.sqrt(p_no) * np.random.randn(*x.shape)
            if np.random.rand()<0.5:
                shift = np.random.randint(x.shape[1])
                x = np.roll(x, shift, axis=1)
            if np.random.rand()<0.5:
                x *= np.random.uniform(0.7,1.3)
            if np.random.rand()<0.2:
                L = x.shape[1]; m = np.random.randint(L//20, L//5)
                s = np.random.randint(0, L-m); x[:,s:s+m]=0

        
            if np.random.rand()<self.mixup_prob:
                j = np.random.randint(len(self.X))
                x2 = self.X[j]
                y2 = int(self.y[j])
                lam = np.random.beta(1.0,1.0)
                x = lam * x + (1-lam) * x2
                return torch.from_numpy(x).float(), (y, y2, lam)

        return torch.from_numpy(x).float(), y