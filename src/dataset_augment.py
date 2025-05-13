import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class RadioMLDataset(Dataset):
    def __init__(self,
                 path: str,
                 split: str = 'train',
                 val_fraction: float = 0.2,
                 augment: bool = False,
                 random_seed: int = 42):
        """
        path:       путь к data/raw/RML2016.10a_dict.pkl
        split:      'train' или 'val'
        augment:    включить/выключить аугментации
        """
        with open(path, 'rb') as f:
            Xd = pickle.load(f, encoding='latin1')

        X_list, y_list = [], []
        for (mod, snr), data in Xd.items():
            X_list.append(data)                   
            y_list.extend([mod] * data.shape[0])  

        X = np.vstack(X_list) 

        self.le = LabelEncoder()
        y = self.le.fit_transform(y_list)  

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y,
            test_size=val_fraction,
            stratify=y,
            random_state=random_seed
        )

        if split == 'train':
            self.X, self.y = X_tr, y_tr
        else:
            self.X, self.y = X_val, y_val

        self.augment = augment
        np.random.seed(random_seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].copy()  # (2, L)
        y = self.y[idx]

        if self.augment:
            x = self._augment(x)

        return torch.from_numpy(x).float(), y

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """
        Набор аугментаций:
         1) Добавление Гауссова шума с случайным SNR
         2) Циклический сдвиг по времени
         3) Случайное масштабирование амплитуды
         4) Маскирование (time-mask)
         5) Small random frequency shift (на частотной подвыборке)
        """
        snr_db = np.random.uniform(0, 20)  # SNR от 0 до 20 дБ
        power_signal = np.mean(x**2)
        power_noise = power_signal / (10**(snr_db/10))
        noise = np.sqrt(power_noise) * np.random.randn(*x.shape)
        x = x + noise

        if np.random.rand() < 0.4:
            shift = np.random.randint(1, x.shape[1])
            x = np.roll(x, shift, axis=1)

        if np.random.rand() < 0.4:
            scale = np.random.uniform(0.7, 1.3)
            x = x * scale

        if np.random.rand() < 0.2:
            t_len = x.shape[1]
            mask_len = np.random.randint(t_len // 20, t_len // 5)
            start = np.random.randint(0, t_len - mask_len)
            x[:, start:start + mask_len] = 0

        if np.random.rand() < 0.2:
            f_len = x.shape[0] 
            idx = np.random.choice(f_len)
            x[idx, :] = 0

        return x
