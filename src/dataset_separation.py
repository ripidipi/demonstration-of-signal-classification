from torch.utils.data import random_split

from dataset import RadioMLDataset

full_dataset = RadioMLDataset('data/raw/RML2016.10a_dict.pkl')

train_size = int(0.7 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
