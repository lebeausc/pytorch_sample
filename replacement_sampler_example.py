import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data.sampler import Sampler
import random

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = torch.arange(8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class ReplacementSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.indices = [i for i in range(len(self.data_source))]
        self.replace_num = 0

    def __iter__(self):
        # replace num
        self.replace_num += 1
        self.replace_num %= len(self.indices)
        # shuffle
        indices = random.sample(self.indices, len(self.indices))
        # replace
        indices = (0 if self.replace_num == i else i for i in indices )
        return indices

    def __len__(self):
        return len(self.data_source)

dataset = Dataset()
replacementSampler = ReplacementSampler(dataset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, num_workers=4, sampler=replacementSampler)

for epoch in range(10):
    for data in dataloader:
        print('epoch {}: {}'.format(epoch, data))
