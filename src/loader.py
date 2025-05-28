from torch.utils.data import DataLoader
from dataset_augment import RadioMLH5Dataset, augment_iq

def get_dataloaders(h5_path, batch_size=512, val_fraction=0.2,
                    num_workers=4, transform=augment_iq):
    train_ds = RadioMLH5Dataset(h5_path,'train',val_fraction,transform=transform)
    val_ds   = RadioMLH5Dataset(h5_path,'val',  val_fraction,transform=None)
    train_loader = DataLoader(train_ds,batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,pin_memory=True)
    val_loader   = DataLoader(val_ds,  batch_size=batch_size,
                              shuffle=False,num_workers=num_workers,pin_memory=True)
    num_classes = len(set(train_ds.y.tolist()))
    return train_loader, val_loader, num_classes