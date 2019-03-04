import numpy as np

import torch
from torchvision import utils

def main():

    print('- dataset ---')
    # データ
    n_trX = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float)
    print(n_trX)
    t_trX = torch.from_numpy(n_trX).float()
    print(t_trX)
    # ラベル
    n_trY = np.array([0, 1, 1, 0, 2, 0, 2])
    print(n_trY)
    t_trY = torch.from_numpy(n_trY)
    print(t_trY)

    # データセット作成
    sample_dataset = torch.utils.data.TensorDataset(t_trX, t_trY)
    # データローダー作成
    sample_loader = torch.utils.data.DataLoader(sample_dataset, batch_size=2, shuffle=True)

    # イテレーション
    for epoch in [1, 2]:
        print('- epoch {} ---'.format(epoch))
        for idx, (data, labels) in enumerate(sample_loader):
            print('{}-{} data : {}'.format(epoch, idx, data))
            print('{}-{} label: {}'.format(epoch, idx, labels))

if __name__ == '__main__':
    main()
