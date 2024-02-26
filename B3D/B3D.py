# This code is implemented based on Neural Cleanse (NC) at https://github.com/bolunwang/backdoor

import os
import argparse
import time

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


def add_trigger_test(testset, target):
    from copy import deepcopy
    testset_trigger = deepcopy(testset)
    im = Image.open(os.path.join(os.path.dirname(args.model_path), 'mask.png'))
    mask = np.array(im) / 255
    mask = mask.astype(np.uint8)
    im = Image.open(os.path.join(os.path.dirname(args.model_path), 'trigger.png'))
    trigger = np.array(im)
    print(mask.shape, trigger.shape, np.sum(mask[:,:,0] == 1), flush=True)

    testset_trigger.data = testset_trigger.data * (1 - mask) + mask * trigger
    testset_trigger.targets = [target] * len(testset_trigger.targets)
    return testset_trigger


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
        100. * correct / len(test_loader.dataset)), flush=True)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def evaluate_backdoor(model, device, test_loader, mask, pattern, target_label):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            bs = len(data)
            target = target_label.repeat(bs)
            data = (1 - mask) * data + mask * pattern
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)), flush=True)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


def optimize_blackbox(model, test_loader, target_label, input_shape, lr=0.05, epochs=20, init_cost=0.001, attack_succ_threshold=0.99, patience=5, cost_multiplier=2, early_stop_threshold=1.0, epsilon=1e-8, samples_per_draw=50, early_stop_flag=False):
    mask_shape = input_shape[1:]
    nb_sample = len(test_loader.dataset)
    mini_batch = nb_sample // test_loader.batch_size
    cost_multiplier_up = cost_multiplier
    cost_multiplier_down = cost_multiplier ** 1.5
    early_stop_patience = 5 * patience

    model.eval()
    device=next(model.parameters()).device

    # Initialization
    theta_m = atanh((torch.rand(mask_shape) - 0.5) * (2 - epsilon)).unsqueeze_(0).cuda()
    theta_p = atanh((torch.rand(input_shape) - 0.5) * (2 - epsilon)).cuda()

    print('theta_m', torch.min(theta_m), torch.max(theta_m), flush=True)
    print('theta_p', torch.min(theta_p), torch.max(theta_p), flush=True)

    theta_m = theta_m.clone().detach().requires_grad_(True)
    theta_p = theta_p.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([theta_m, theta_p], lr=lr, betas=[0.5, 0.9])

    lbd = 0 # initial lambda

    # best optimization results
    mask_best = None
    pattern_best = None
    reg_best = float('inf')

    # logs and counters for adjusting balance cost
    cost_up_counter = 0
    cost_down_counter = 0
    cost_up_flag = False
    cost_down_flag = False

    # counter for early stop
    early_stop_counter = 0
    early_stop_reg_best = reg_best

    step = 0
    loss_cls_list = []
    loss_reg_list = []
    acc_list = []
    stop = False

    # samples_per_draw = 50
    sigma = 0.1

    for i in range(epochs):
        for data, _ in test_loader:
            data = data.to(device)

            # record loss for adjusting lambda
            with torch.no_grad():
                pattern = torch.tanh(theta_p) / 2 + 0.5
                soft_mask = (torch.tanh(theta_m) / 2 + 0.5).repeat(3, 1, 1).unsqueeze_(0)
                reverse_mask = torch.ones_like(soft_mask) - soft_mask
                backdoor_data = reverse_mask * data + soft_mask * pattern
                logits = model(backdoor_data)

                # record baseline loss to stablize updates
                loss_baseline = F.cross_entropy(logits, torch.ones((data.size(0)), dtype=torch.int64, device=device) * target_label)
                loss_l1 = torch.sum(torch.abs(soft_mask)) / 3

                pred = logits.max(1, keepdim=True)[1]
                acc = pred.eq(target_label).float().mean()

                loss_cls_list.append(loss_baseline.item())
                loss_reg_list.append(loss_l1.item())
                acc_list.append(acc.item())

            # black-box optimization
            losses_pattern = torch.zeros([samples_per_draw]).cuda()
            losses_mask = torch.zeros([samples_per_draw]).cuda()

            epsilon_pattern = torch.randn([samples_per_draw] + input_shape).cuda()
            soft_mask = torch.tanh(theta_m) / 2 + 0.5
            mask_samples = torch.zeros([samples_per_draw] + mask_shape).cuda()
            for j in range(samples_per_draw):
                mask_samples[j] = torch.bernoulli(soft_mask)

            with torch.no_grad():
                for j in range(samples_per_draw):
                    pattern_try = torch.tanh(theta_p + sigma * epsilon_pattern[j]) / 2 + 0.5
                    mask_try = soft_mask.repeat(3, 1, 1).unsqueeze_(0)
                    reverse_mask_try = torch.ones_like(mask_try) - mask_try
                    backdoor_data = reverse_mask_try * data + mask_try * pattern_try
                    logits = model(backdoor_data)
                    loss_cls = F.cross_entropy(logits, torch.ones((data.size(0)), dtype=torch.int64, device=device) * target_label)
                    losses_pattern[j] = loss_cls - loss_baseline

                    pattern_try = torch.tanh(theta_p) / 2 + 0.5
                    mask_try = mask_samples[j].unsqueeze_(0).repeat(3, 1, 1).unsqueeze_(0)
                    reverse_mask_try = torch.ones_like(mask_try) - mask_try
                    backdoor_data = reverse_mask_try * data + mask_try * pattern_try
                    logits = model(backdoor_data)
                    loss_cls = F.cross_entropy(logits, torch.ones((data.size(0)), dtype=torch.int64, device=device) * target_label)
                    losses_mask[j] = loss_cls - loss_baseline

            # we calculate the precise gradient of the l1 norm w.r.t. theta_m rather than approximation
            grad_theta_p = (losses_pattern.view([samples_per_draw, 1, 1, 1]) * epsilon_pattern).mean(0) / sigma
            grad_theta_m = (losses_mask.view([samples_per_draw, 1, 1]) * 2 * (mask_samples - soft_mask)).mean(0, keepdim=True) + 2 * lbd * soft_mask * (1 - soft_mask)

            optimizer.zero_grad()
            theta_p.backward(grad_theta_p)
            theta_m.backward(grad_theta_m)
            optimizer.step()

            step += 1

            if step % 10 == 0:
                # update lambda and early-stop
                avg_loss_cls = np.mean(loss_cls_list)
                avg_loss_reg = np.mean(loss_reg_list)
                avg_acc = np.mean(acc_list)
                loss_cls_list = []
                loss_reg_list = []
                acc_list = []

                # check to save best mask or not
                if avg_acc >= attack_succ_threshold and avg_loss_reg < reg_best:
                    mask_best = soft_mask
                    pattern_best = pattern
                    reg_best = avg_loss_reg

                print(time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + 'step: %3d/%3d, lambda: %.10f, attack: %.3f, cls: %f, reg: %f, reg_best: %f' %
                      (step, mini_batch, lbd, avg_acc, avg_loss_cls, avg_loss_reg, reg_best), flush=True)

                # only terminate if a valid attack has been found
                if reg_best < float('inf'):
                    if reg_best >= early_stop_threshold * early_stop_reg_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_reg_best = min(reg_best, early_stop_reg_best)

                if (cost_down_flag and cost_up_flag and early_stop_counter >= early_stop_patience and lbd > 0.00001 and early_stop_flag):
                    print('early stop', flush=True)
                    stop = True
                    break

                # check cost modification
                if lbd == 0 and avg_acc >= attack_succ_threshold:
                    lbd = init_cost
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
                    print('initialize lambda to %.10f' % lbd, flush=True)

                elif lbd > 0:
                    if avg_acc >= attack_succ_threshold:
                        cost_up_counter += 1
                        cost_down_counter = 0
                    else:
                        cost_up_counter = 0
                        cost_down_counter += 1

                    if cost_up_counter >= patience:
                        cost_up_counter = 0
                        print('up lambda from %.10f to %.10f' % (lbd, lbd * cost_multiplier_up), flush=True)
                        lbd *= cost_multiplier_up
                        cost_up_flag = True
                    elif cost_down_counter >= patience:
                        cost_down_counter = 0
                        print('down lambda from %.10f to %.10f' % (lbd, lbd / cost_multiplier_down), flush=True)
                        lbd /= cost_multiplier_down
                        cost_down_flag = True

        if stop == True:
            break
    return mask_best, pattern_best


def outlier_detection(l1_norm):

    def MAD(values, thres=2.0):
        consistency_constant = 1.4826  # if normal distribution
        median = np.median(values)
        mad = consistency_constant * np.median(np.abs(values - median))
        c = []
        for i, v in enumerate(values):
            if v > median:
                continue
            if np.abs(v - median) / mad > thres:
                #print(i, v, flush=True)
                c.append(i)
        return c

    classes_mad = MAD(l1_norm, 3.0)
    classes_com = np.where(l1_norm < 0.25 * np.median(l1_norm))[0]
    
    classes = []
    if len(classes_mad) == 0:
        classes = classes_com
    elif len(classes_com) == 0:
        classes = classes_mad
    else:
        for i in range(len(l1_norm)):
            if i in classes_mad and i in classes_com:
                classes.append(i)
    return classes


# parser = argparse.ArgumentParser(description='Black-box Backdoor Detection')
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='disables CUDA training')
# parser.add_argument('--seed', type=int, default=0, metavar='S',
#                     help='random seed (default: 1)')
# parser.add_argument('--model-path', default='./model-cifar-wideResNet',
#                     help='directory of model for saving checkpoint')
# parser.add_argument('--target', type=int, default=0)
# parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
#                     help='input batch size for testing (default: 128)')
# parser.add_argument('--trigger', action='store_true', default=False)

# args = parser.parse_args()
# print(args, flush=True)


# # specify optimization related parameters
# LR = 0.05  # learning rate
# EPOCHS = 20  # total optimization epochs
# NB_SAMPLE = 1000  # number of samples for adjusting lambda
# MINI_BATCH = NB_SAMPLE // args.test_batch_size  # number of batches
# INIT_COST = 1e-3  # initial weight of lambda

# ATTACK_SUCC_THRESHOLD = 0.99  # attack success threshold
# PATIENCE = 5  # patience for adjusting lambda, number of mini batches
# COST_MULTIPLIER = 2  # multiplier for auto-control of weight (COST)
# COST_MULTIPLIER_UP = COST_MULTIPLIER
# COST_MULTIPLIER_DOWN = COST_MULTIPLIER ** 1.5

# EARLY_STOP_THRESHOLD = 1.0  # loss threshold for early stop
# EARLY_STOP_PATIENCE = 5 * PATIENCE  # patience for early stop
# EPSILON = 1e-8

# # settings
# use_cuda = not args.no_cuda and torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
# kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
# ])

# testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

# if args.trigger:
#     testset_trigger = add_trigger_test(testset, args.target)
#     test_loader_trigger = torch.utils.data.DataLoader(testset_trigger, batch_size=args.test_batch_size, shuffle=False, **kwargs)


# model = ResNet18()
# #model = VGG('VGG16')
# model = nn.DataParallel(model).cuda()
# print('======Load Model======', flush=True)
# model.load_state_dict(torch.load(args.model_path))
# evaluate(model, device, test_loader)
# if args.trigger:
#     print('======Test Backdoor Attack======', flush=True)
#     evaluate(model, device, test_loader_trigger)

# mask_list = []
# pattern_list = []
# l1_norm_list = []
# acc_list = []

# for i in range(10):
#     print('processing label %d' % i, flush=True)
#     target_label = torch.tensor([i]).to(device)
#     mask, pattern = optimize_blackbox(model, test_loader, target_label)
#     if mask is None:
#         continue

#     print('======Evaluate Reversed Triggers======', flush=True)
#     _, acc = evaluate_backdoor(model, device, test_loader, mask, pattern, target_label)

#     mask = mask.detach().permute(1,2,0).squeeze_().cpu().numpy()
#     pattern = pattern.detach().permute(1,2,0).cpu().numpy()
#     print('mask:', mask.shape, np.min(mask), np.max(mask), flush=True)
#     print('pattern:', pattern.shape, np.min(pattern), np.max(pattern), flush=True)

#     mask_list.append(mask)
#     pattern_list.append(pattern)
#     l1_norm_list.append(np.sum(np.abs(mask)))
#     acc_list.append(acc)

#     im = Image.fromarray((mask * 255).astype(np.uint8))
#     im.save(os.path.join(os.path.dirname(args.model_path), 'b3d_mask_' +  str(i) + '.png'))
#     im = Image.fromarray((pattern * 255).astype(np.uint8))
#     im.save(os.path.join(os.path.dirname(args.model_path), 'b3d_pattern_' + str(i) + '.png'))
#     im = Image.fromarray((mask.reshape([32,32,1]) * pattern * 255).astype(np.uint8))
#     im.save(os.path.join(os.path.dirname(args.model_path), 'b3d_fusion_' + str(i) + '.png'))

# print('l1 norm:', l1_norm_list, flush=True)

# abnormal_classes = outlier_detection(l1_norm_list)
# for c in abnormal_classes:
#     print('Infected class:', c, 'l1 norm:', l1_norm_list[c], flush=True)

# np.savez(os.path.join(os.path.dirname(args.model_path), 'b3d_results.npz'), 
#             mask=mask_list, pattern=pattern_list,
#             l1_norm=l1_norm_list, acc=acc_list)
