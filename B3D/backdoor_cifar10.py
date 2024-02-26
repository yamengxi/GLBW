from __future__ import print_function
from __future__ import division

import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms

from resnet import *
from vgg import *

parser = argparse.ArgumentParser(description='Backdoor Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--lr-policy', type=str, default='cosine',
                    choices=['step', 'cosine'], help='learning rate decay method')
parser.add_argument('--lr-milestones', default=[100,150], 
                    help='milestones for lr decay')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=200, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--data-augmentation', '--da', action='store_true', default=False)

parser.add_argument('--trigger', action='store_true', default=False)
parser.add_argument('--target', type=int, default=0)
parser.add_argument('--trigger-size', type=int, default=3)
parser.add_argument('--trigger-ratio', type=float, default=0.1)


def add_trigger(trainset, target, trigger_size, trigger_ratio):
    mask = np.zeros([32, 32, 3], dtype=np.uint8)
    trigger = np.zeros([32, 32, 3], dtype=np.uint8)
    position_x = np.random.randint(0, 33-trigger_size)
    position_y = np.random.randint(0, 33-trigger_size)
    color = np.random.randint(0, 256, [trigger_size, trigger_size, 3])
    print('position:', position_x, position_y, 'color:', color)
    
    mask[position_x:position_x+trigger_size, position_y:position_y+trigger_size, :] = 1
    trigger[position_x:position_x+trigger_size, position_y:position_y+trigger_size, :] = color
    
    im = Image.fromarray(mask * 255)
    im.save(os.path.join(args.model_dir, 'mask.png'))

    im = Image.fromarray(trigger)
    im.save(os.path.join(args.model_dir, 'trigger.png'))

    num_poison = int(len(trainset.targets) * trigger_ratio)

    index = np.where(np.asarray(trainset.targets) != target)[0]
    np.random.shuffle(index)
    selected_index = index[:num_poison]
    
    for idx in selected_index:
        trainset.data[idx] = (1 - mask) * trainset.data[idx] + mask * trigger
        trainset.targets[idx] = target

    for i in range(10):
        print('number of images belonging to label', i, 'is', np.sum(np.asarray(trainset.targets) == i))

    return trainset, mask, trigger


def add_trigger_test(testset, mask, trigger, target):
    from copy import deepcopy
    testset_trigger = deepcopy(testset)
    testset_trigger.data =  testset_trigger.data * (1 - mask) + mask * trigger
    testset_trigger.targets = [target] * len(testset_trigger.targets)
    return testset_trigger


args = parser.parse_args()
print(args)

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

# setup data loader
if args.data_augmentation:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

if args.trigger:
    trainset, mask, trigger = add_trigger(trainset, args.target, args.trigger_size, args.trigger_ratio)
    testset_trigger = add_trigger_test(testset, mask, trigger, args.target)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
if args.trigger:
    test_loader_trigger = torch.utils.data.DataLoader(testset_trigger, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        loss = F.cross_entropy(model(data), target)

        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.item()))


def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def main():
    model = ResNet18()
    #model = VGG('VGG16')
    model = nn.DataParallel(model).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.lr_policy == 'step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
    elif args.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    else:
        raise NotImplementedError

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        scheduler.step()

        # train
        train(args, model, device, train_loader, optimizer, epoch)

        # evaluation
        print('================================================================')
        evaluate(model, device, test_loader)
        if args.trigger:
            print('======Test Backdoor Attack======')
            evaluate(model, device, test_loader_trigger)
        print('================================================================')

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-wideres-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-wideres-checkpoint_epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()
