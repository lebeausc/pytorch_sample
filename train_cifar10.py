# import comet_ml in the top of your file
from comet_ml import Experiment

import numpy as np

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils

from resnet import ResNet18

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(args, model, device, train_loader, optimizer, epoch, experiment):
    model.train()
    correct = 0
    total = 0
    iter_num = len(train_loader)
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output.log(), labels, reduction='mean')
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += len(data)
        g_step = iter_num * (epoch - 1) + batch_idx
        experiment.log_metric("loss", loss.item(), step=g_step)
        experiment.log_metric("accuracy", correct / total, step=g_step)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    experiment.log_metric("lr", get_lr(optimizer), step=epoch)

def test(args, model, device, test_loader, epoch, experiment):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            test_loss += F.nll_loss(output.log(), labels, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    experiment.log_metric("loss", test_loss, step=(epoch-1))
    experiment.log_metric("accuracy", correct / len(test_loader.dataset), step=(epoch-1))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Cifar10 Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=25, metavar='N', help='number of epochs to train (default: 25)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--model-path', type=str, default='', metavar='M', help='model param path')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()


    # Add the following code anywhere in your machine learning file
    experiment = Experiment(api_key="EON8CpyArwsDSGkzGRv64c2vO", project_name="cifar10-test", workspace="ai-cometml-abc")

    # ブラウザの実験ページを開く
    experiment.display(clear=True, wait=True, new=0, autoraise=True)
    # 実験キー(実験を一意に特定するためのキー)の取得
    exp_key = experiment.get_key()
    print('KEY: ' + exp_key)
    # HyperParamの記録
    hyper_params = {
        'batch_size': args.batch_size,
        'epoch': args.epochs,
        'learning_rate': args.lr,
        'sgd_momentum' : args.momentum,
        'model_path' : args.model_path,
        'torch_manual_seed': args.seed
    }
    experiment.log_multiple_params(hyper_params)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('use_cuda {}'.format(use_cuda))

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = ResNet18().to(device)
    if len(args.model_path) > 0:
        model.load_state_dict(torch.load(args.model_path))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    # lr = 0.1 if epoch < 15
    # lr = 0.01 if 15 <= epoch < 20
    # lr = 0.001 if 20 <= epoch < 25
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,20], gamma=0.1)

    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        with experiment.train():
            experiment.log_current_epoch(epoch)
            train(args, model, device, train_loader, optimizer, epoch, experiment)
        with experiment.test():
            test(args, model, device, test_loader, epoch, experiment)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), "./model/cifar10_{0}_{1:04d}.pt".format(exp_key, epoch))

if __name__ == '__main__':
    main()
