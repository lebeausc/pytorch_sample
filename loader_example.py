import numpy as np

import torch
from torchvision import utils, transforms

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dinosaur_list, transform=None):
        self.dinosaur_list = dinosaur_list
        self.transform = transform

    def __len__(self):
        return len(self.dinosaur_list)

    def __getitem__(self, idx):
        print('called MyDataset')
        result = self.dinosaur_list[idx]
        if self.transform:
            result = self.transform(result)
        return result

class MyTransform(object):
    def __call__(self, data):
        print('called MyTransform')
        data, label = data, len(data)
        return data, label

def main():
    workers = 0
    dinosaur_list = ['Tyrannosaurus', 'Triceratops', 'Velociraptor', 'Agustinia', 'Uteodon', 'Kosmoceratops']

    data_set = MyDataset(dinosaur_list, transform=transforms.Compose([MyTransform()]))

    data_loader = torch.utils.data.DataLoader(data_set, batch_size=2, shuffle=True, num_workers=workers)

    for epoch in [1, 2]:
        print('- epoch {} ---'.format(epoch))
        for idx, (data, labels) in enumerate(data_loader):
            print('{}-{} data : {}'.format(epoch, idx, data))
            print('{}-{} label: {}'.format(epoch, idx, labels))

if __name__ == '__main__':
    main()
