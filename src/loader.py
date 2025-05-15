import torch
from torch.utils.data import DataLoader
from dataset_augment import RadioMLDataset


def mixup_collate_fn(batch):
    Xs, Ys = zip(*batch)
    X = torch.stack(Xs, dim=0)
    B = X.shape[0]

    y1_list, y2_list, lam_list = [], [], []
    for y in Ys:
        if isinstance(y, tuple):
            y1_, y2_, lam_ = y
        else:
            y1_, y2_, lam_ = y, y, 1.0
        y1_list.append(y1_)
        y2_list.append(y2_)
        lam_list.append(lam_)

    y1 = torch.tensor(y1_list, dtype=torch.long)
    y2 = torch.tensor(y2_list, dtype=torch.long)
    lam = torch.tensor(lam_list, dtype=torch.float32)

    idx = torch.randperm(B)
    X2 = X[idx]
    lam_b = lam.view(B, *[1] * (X.dim() - 1))
    X_mix = lam_b * X + (1 - lam_b) * X2

    return X_mix, (y1, y2, lam)


def get_dataloaders(path, batch_size=256, val_fraction=0.2,
                    num_workers=4, pin_memory=False,
                    mixup_prob=0.5):
    train_ds = RadioMLDataset(path, 'train', val_fraction,
                               augment=True, mixup_prob=mixup_prob)
    val_ds = RadioMLDataset(path, 'val', val_fraction,
                             augment=False, mixup_prob=0.0)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=mixup_collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, train_ds.le