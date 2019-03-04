import numpy as np

import torch
from torchvision import utils, transforms, datasets
from torch.utils.data.sampler import Sampler


class MyRandomSampler(Sampler):
    def __init__(self, data_source, sampling_rate={}):
        self.data_source = data_source

        label_hash = {}
        for idx, (img, label) in enumerate(self.data_source):
            if not label in label_hash:
                label_hash[label] = []
            label_hash[label].append(idx)

        self.indices = []
        for k in label_hash.keys():
            if k in sampling_rate:
                label_size = len(label_hash[k])
                sampling_count = int(label_size * sampling_rate[k])
                rand_idx = torch.randint(high=label_size - 1, size=(sampling_count,), dtype=torch.int64).numpy()
                self.indices.extend(np.array(label_hash[k])[rand_idx].tolist())
            else:
                self.indices.extend(label_hash[k])

    def __iter__(self):
        print('called MyRandomSampler')
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def main():
    torch.manual_seed(3)

    cifar10_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))

    data_loader = torch.utils.data.DataLoader(cifar10_dataset, batch_size=256, num_workers=0,
            sampler=MyRandomSampler(cifar10_dataset, sampling_rate={1: 0.1, 3: 0.3, 4: 0}))

    for i in [1, 2]:
        print('epoch {}'.format(i))
        label_hash = {}
        label_num = []
        for idx, (data, labels) in enumerate(data_loader):
            for l in labels.tolist():
                if not l in label_hash:
                    label_hash[l] = 0
                label_hash[l] += 1
                label_num.append(l)

        for k in label_hash.keys():
            print('{}: {}'.format(k, label_hash[k]))
        print(label_num[:20])

if __name__ == '__main__':
    main()
