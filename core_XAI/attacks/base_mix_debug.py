from copy import deepcopy
import math
import os
import os.path as osp
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, MNIST, CIFAR10

from ..utils import Log


support_list = (
    DatasetFolder,
    MNIST,
    CIFAR10
)


def check(dataset):
    return isinstance(dataset, support_list)


def cal_adv_loss(loss, digits, labels, poisoned_labels, probabilities):
    """Calculate adv loss.

    Args:
        loss (function): Loss.
        digits (torch.Tensor): (B, C).
        labels (torch.Tensor): (B).
        poisoned_labels (torch.Tensor): (B).
        probabilities (torch.Tensor): (B) or (1).

    Returns:
        torch.Tensor: The calculated adv loss.
    """
    return loss(digits, labels) * probabilities + loss(digits, poisoned_labels) * (1.0 - probabilities)


def tanh_func(x):
    return x.tanh().add(1).mul(0.5)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AddTrigger:
    """Add watermarked trigger to images.

    Args:
        num_classes (int): Number of total classes.
        y_target (int): N-to-1 attack target label.
        pattern (None | torch.Tensor): Trigger pattern, shape (C, H, W) or (H, W), dtype torch.float32 (between 0.0 and 1.0).
        weight (None | torch.Tensor): Trigger pattern weight, shape (C, H, W) or (H, W), dtype torch.float32 (between 0.0 and 1.0).
        device (torch.device): Device for computing.
    """

    def __init__(self, num_classes, y_target, pattern, weight, device):
        self.num_classes = num_classes
        self.y_target = y_target

        self.res = pattern.to(device)
        if self.res.dim() == 2:
            self.res = self.res.unsqueeze(0)
        self.res = self.res.unsqueeze(0) # (1, C, H, W)

        self.weight = weight.to(device)
        if self.weight.dim() == 2:
            self.weight = self.weight.unsqueeze(0)
        self.weight = self.weight.unsqueeze(0) # (1, C, H, W)

        self.res = self.weight * self.res
        self.weight = 1.0 - self.weight

        self.device = device

    def __call__(self, imgs):
        """Add watermarked trigger to images.

        Args:
            img (torch.Tensor): shape (B, C, H, W), dtype torch.float32 (between 0.0 and 1.0).

        Returns:
            torch.Tensor: Poisoned images, shape (B, C, H, W).
            torch.Tensor: Poisoned labels, shape (B).
        """
        poisoned_imgs = self.weight * imgs + self.res

        labels = torch.zeros((imgs.size(0)), dtype=torch.long, device=self.device)
        labels[:] = self.y_target

        return poisoned_imgs, labels


class BaseMix(object):
    """Base class for backdoor mixture training and testing.

    Args:
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        model (torch.nn.Module): Network.
        loss (torch.nn.Module): Loss.
        num_classes (int): Number of total classes.
        y_target (int): N-to-1 attack target label.
        pattern (None | torch.Tensor): Trigger pattern, shape (C, H, W) or (H, W), dtype torch.float32 (between 0.0 and 1.0).
        weight (None | torch.Tensor): Trigger pattern weight, shape (C, H, W) or (H, W), dtype torch.float32 (between 0.0 and 1.0).
        schedule (dict): Training or testing global schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(self, train_dataset, test_dataset, model, loss, num_classes, y_target, pattern, weight, schedule=None, seed=0, deterministic=False):
        assert isinstance(train_dataset, support_list), 'train_dataset is an unsupported dataset type, train_dataset should be a subclass of our support list.'
        self.train_dataset = train_dataset

        assert isinstance(test_dataset, support_list), 'test_dataset is an unsupported dataset type, test_dataset should be a subclass of our support list.'
        self.test_dataset = test_dataset
        self.model = model
        self.loss = loss
        self.num_classes = num_classes
        self.y_target = y_target
        self.pattern = pattern
        self.weight = weight
        self.global_schedule = deepcopy(schedule)
        self.current_schedule = None
        self._set_seed(seed, deterministic)

    def _set_seed(self, seed, deterministic):
        # Use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA).
        torch.manual_seed(seed)

        # Set python seed
        random.seed(seed)

        # Set numpy seed (However, some applications and libraries may use NumPy Random Generator objects,
        # not the global RNG (https://numpy.org/doc/stable/reference/random/generator.html), and those will
        # need to be seeded consistently as well.)
        np.random.seed(seed)

        os.environ['PYTHONHASHSEED'] = str(seed)

        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            # Hint: In some versions of CUDA, RNNs and LSTM networks may have non-deterministic behavior.
            # If you want to set them deterministic, see torch.nn.RNN() and torch.nn.LSTM() for details and workarounds.

    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_model(self):
        return self.model

    def adjust_learning_rate(self, optimizer, epoch, step, len_epoch):
        factor = (torch.tensor(self.current_schedule['schedule']) <= epoch).sum()

        lr = self.current_schedule['lr']*(self.current_schedule['gamma']**factor)

        """Warmup"""
        if 'warmup_epoch' in self.current_schedule and epoch < self.current_schedule['warmup_epoch']:
            lr = lr*float(1 + step + epoch*len_epoch)/(self.current_schedule['warmup_epoch']*len_epoch)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def data2img(self, data):
        return (self.current_schedule['inv_normalize'](data) * 255).clip(0.0, 255.0).round().to(dtype=torch.uint8).permute(1, 2, 0).cpu().numpy()

    def get_adv_mask(self, data_loader, device, log, adv_id, adv_lambda, init=False):
        delta = self.pattern.to(device)
        if delta.dim() == 2:
            delta = delta.unsqueeze(0)
        delta = delta.unsqueeze(0) # (1, C, H, W)

        _, C, H, W = delta.size()

        L = torch.nn.CrossEntropyLoss(reduction='none')

        f = self.model

        torch.manual_seed(int(time.time()))
        atanh_m_ = torch.randn((1, 1, H, W), device=device) # (1, 1, H, W)
        torch.save(atanh_m_, '/HOME/scz0bof/run/Backdoor/XAI/atanh_m_.pth')
        atanh_m_[0,0,-3:,-3:] = -1000000000000.0
        atanh_m_ = torch.nn.Parameter(atanh_m_)

        atanh_delta_ = torch.randn((1, C, H, W), device=device) # (1, C, H, W)
        torch.save(atanh_delta_, '/HOME/scz0bof/run/Backdoor/XAI/atanh_delta_.pth')
        atanh_delta_ = torch.nn.Parameter(atanh_delta_)

        optimizer = torch.optim.Adam([atanh_delta_, atanh_m_], lr=0.1, betas=(0.5, 0.9))

        best_mean_total_loss = float("inf")
        best_mean_ce_loss = float("inf")
        best_mean_norm_loss = float("inf")
        best_m = None
        best_delta = None
        for epoch in range(1, 11):
            sum_ce_loss = 0.0
            sum_norm_loss = 0.0
            for batch_id, batch in enumerate(data_loader):
                # Map to [0, 1] (differentiable).
                m_ = tanh_func(atanh_m_) # (1, 1, H, W), [0.0, 1.0]
                delta_ = tanh_func(atanh_delta_) # (1, C, H, W), [0.0, 1.0]

                B, C, H, W = batch[0].size()
                x = batch[0].to(device) # (B, C, H, W)
                y = batch[1].to(device) # (B)

                y_t = torch.zeros((B), dtype=torch.long, device=device) # (B)
                y_t[:] = self.y_target

                # breakpoint()
                # print(f.linear.weight)
                digits = f(x * (1.0 - m_) + m_ * delta_) # (B, self.num_classes)

                penalty = (m_[:,0,:,:].abs()).sum(dim=(1, 2)) # (1)
                backdoor_losses = L(digits, y_t) # (B), minimum
                penalty_loss = penalty # (1), minimum
                losses = backdoor_losses + adv_lambda * penalty_loss # (B)
                sum_ce_loss += backdoor_losses.sum() # scalar tensor
                sum_norm_loss += penalty_loss[0] * B # scalar tensor

                loss = losses.mean() # scalar tensor

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            print(f'==========epoch: {epoch}==========\n')
            mean_total_loss = (sum_ce_loss + adv_lambda * sum_norm_loss) / len(data_loader.dataset) # scalar tensor
            mean_ce_loss = sum_ce_loss / len(data_loader.dataset) # scalar tensor
            mean_norm_loss = sum_norm_loss / len(data_loader.dataset) # scalar tensor
            print(time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"mean_total_loss:{mean_total_loss.cpu().item()}, mean_ce_loss:{mean_ce_loss.cpu().item()}, mean_norm_loss:{mean_norm_loss.cpu().item()}, adv_lambda: {adv_lambda}\n")

            # if self.reversed_trigger_dir is not None:
            #     trigger_dir = os.path.join(self.reversed_trigger_dir, str(label))
            #     if not os.path.exists(trigger_dir):
            #         os.makedirs(trigger_dir, exist_ok=True)
            #     self.save_trigger(trigger_dir, delta_[0], m_[0,0], suffix=f'_{epoch:03d}_latest')


            if best_mean_total_loss >= mean_total_loss:
                best_mean_total_loss = mean_total_loss # best epoch, scalar tensor
                best_mean_ce_loss = mean_ce_loss # best epoch, scalar tensor
                best_mean_norm_loss = mean_norm_loss # best epoch, scalar tensor
                best_m = m_.detach() # best mask
                best_delta = delta_.detach() # best mask

        print(f'==========result==========\n')
        print(time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"best_mean_total_loss:{best_mean_total_loss.cpu().item()}, best_mean_ce_loss:{best_mean_ce_loss.cpu().item()}, best_mean_norm_loss:{best_mean_norm_loss.cpu().item()}\n")


        return None

    def train(self, schedule=None):
        if schedule is None and self.global_schedule is None:
            raise AttributeError("Training schedule is None, please check your schedule setting.")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)


        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

        # log and output:
        # 1. experiment config
        # 2. ouput loss and time
        # 3. test and output statistics
        # 4. save checkpoint

        log('==========Schedule parameters==========\n')
        log(str(self.current_schedule)+'\n')


        if 'pretrain' in self.current_schedule:
            self.model.load_state_dict(torch.load(self.current_schedule['pretrain'], map_location='cpu'), strict=False)
            log(f"Load pretrained parameters: {self.current_schedule['pretrain']}\n")


        # Use GPU
        if 'device' in self.current_schedule and self.current_schedule['device'] == 'GPU':
            log('==========Use GPUs to train==========\n')

            CUDA_VISIBLE_DEVICES = ''
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
            else:
                CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in range(torch.cuda.device_count())])
            log(f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}\n')

            if CUDA_VISIBLE_DEVICES == '':
                raise ValueError(f'This machine has no visible cuda devices!')

            CUDA_SELECTED_DEVICES = ''
            if 'CUDA_SELECTED_DEVICES' in self.current_schedule:
                CUDA_SELECTED_DEVICES = self.current_schedule['CUDA_SELECTED_DEVICES']
            else:
                CUDA_SELECTED_DEVICES = CUDA_VISIBLE_DEVICES
            log(f'CUDA_SELECTED_DEVICES={CUDA_SELECTED_DEVICES}\n')

            CUDA_VISIBLE_DEVICES_LIST = sorted(CUDA_VISIBLE_DEVICES.split(','))
            CUDA_SELECTED_DEVICES_LIST = sorted(CUDA_SELECTED_DEVICES.split(','))

            CUDA_VISIBLE_DEVICES_SET = set(CUDA_VISIBLE_DEVICES_LIST)
            CUDA_SELECTED_DEVICES_SET = set(CUDA_SELECTED_DEVICES_LIST)
            if not (CUDA_SELECTED_DEVICES_SET <= CUDA_VISIBLE_DEVICES_SET):
                raise ValueError(f'CUDA_VISIBLE_DEVICES should be a subset of CUDA_VISIBLE_DEVICES!')

            GPU_num = len(CUDA_SELECTED_DEVICES_SET)
            device_ids = [CUDA_VISIBLE_DEVICES_LIST.index(CUDA_SELECTED_DEVICE) for CUDA_SELECTED_DEVICE in CUDA_SELECTED_DEVICES_LIST]
            device = torch.device(f'cuda:{device_ids[0]}')
            self.model = self.model.to(device)

            if GPU_num > 1:
                self.model = nn.DataParallel(self.model, device_ids=device_ids, output_device=device_ids[0])
        # Use CPU
        else:
            device = torch.device("cpu")

        assert self.current_schedule['batch_size'] % 3 == 0, 'batch_size should be divisible by 3!'

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.current_schedule['batch_size'] // 3,
            shuffle=False,
            num_workers=self.current_schedule['num_workers'],
            drop_last=False,
            pin_memory=True,
            # worker_init_fn=self._seed_worker
        )

        # self.model.eval()
        # for i in range(100):
        #     self.get_adv_mask(train_loader, device, log, 999, self.current_schedule['adv_lambda'])
        # self.model.train()
        # breakpoint()

        # self.model.train()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.current_schedule['lr'], momentum=self.current_schedule['momentum'], weight_decay=self.current_schedule['weight_decay'])

        iteration = 0
        last_time = time.time()

        msg = f"Total train samples: {len(self.train_dataset)}\nTotal test samples: {len(self.test_dataset)}\nBatch size: {self.current_schedule['batch_size']}\niteration every epoch: {len(self.train_dataset)//(self.current_schedule['batch_size'] // 3)}\nInitial learning rate: {self.current_schedule['lr']}\n"
        log(msg)

        add_trigger = AddTrigger(self.num_classes, self.y_target, self.pattern, self.weight, device)

        probability_activation_function = self.current_schedule['probability_activation_function']
        lambda_1 = self.current_schedule['lambda_1']
        lambda_2 = self.current_schedule['lambda_2']
        for i in range(self.current_schedule['epochs']):
            if i == 0: # First test!
                # test on train dataset
                preds, labels, mean_loss, poisoned_preds, poisoned_labels, poisoned_mean_loss, adv_preds, adv_labels, adv_mean_loss = \
                    self._test(self.train_dataset, device, self.current_schedule['batch_size'] // 3, self.current_schedule['num_workers'], log=log, test_id=1000+i)

                total_num = labels.size(0)
                prec1, prec5 = accuracy(preds, labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                msg = "==========Test result on benign train dataset==========\n" + \
                      time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                      f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
                log(msg)

                total_num = poisoned_labels.size(0)
                prec1, prec5 = accuracy(poisoned_preds, poisoned_labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                msg = "==========Test result on poisoned train dataset==========\n" + \
                      time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                      f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {lambda_1 * poisoned_mean_loss}, time: {time.time()-last_time}\n"
                log(msg)

                msg = "==========Mean loss==========\n" + \
                      f"Mean loss: {mean_loss + lambda_1 * poisoned_mean_loss}\n"
                log(msg)

                last_time = time.time()


                # test on test dataset
                preds, labels, mean_loss, poisoned_preds, poisoned_labels, poisoned_mean_loss, adv_preds, adv_labels, adv_mean_loss = \
                    self._test(self.test_dataset, device, self.current_schedule['batch_size'] // 3, self.current_schedule['num_workers'], log=log, test_id=2000+i)

                total_num = labels.size(0)
                prec1, prec5 = accuracy(preds, labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                msg = "==========Test result on benign test dataset==========\n" + \
                      time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                      f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
                log(msg)

                total_num = poisoned_labels.size(0)
                prec1, prec5 = accuracy(poisoned_preds, poisoned_labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                msg = "==========Test result on poisoned test dataset==========\n" + \
                      time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                      f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {lambda_1 * poisoned_mean_loss}, time: {time.time()-last_time}\n"
                log(msg)
                poisoned_test_dataset_accuracy = top1_correct/total_num

                msg = "==========Mean loss==========\n" + \
                      f"Mean loss: {mean_loss + lambda_1 * poisoned_mean_loss}\n"
                log(msg)

                self.current_schedule['poisoned_mean_loss'] = poisoned_mean_loss

                last_time = time.time()

            if poisoned_test_dataset_accuracy > -1.0:
                self.model.eval()
                _, adv_mask, adv_delta, distance = self.get_adv_mask(train_loader, device, log, i+1, self.current_schedule['adv_lambda'])
                self.model.train()
                probability = probability_activation_function(distance) # (1)
                for batch_id, batch in enumerate(train_loader):
                    self.adjust_learning_rate(optimizer, i, batch_id, int(math.ceil(len(self.train_dataset) / self.current_schedule['batch_size'])))
                    B, C, H, W = batch[0].size()
                    imgs = batch[0].to(device) # (B, C, H, W), B=self.current_schedule['batch_size'] // 3
                    labels = batch[1].to(device) # (B)

                    poisoned_imgs, poisoned_labels = add_trigger(imgs) # (B, C, H, W), (B)

                    adv_imgs = imgs * (1.0 - adv_mask) + adv_delta * adv_mask # (B, C, H, W)

                    benign_digits = self.model(imgs)
                    poisoned_digits = self.model(poisoned_imgs)
                    adv_digits = self.model(adv_imgs)
                    losses = self.loss(benign_digits, labels) + lambda_1 * self.loss(poisoned_digits, poisoned_labels) \
                        + lambda_2 * cal_adv_loss(self.loss, adv_digits, labels, poisoned_labels, probability)
                    loss = losses.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    iteration += 1

                    if iteration % self.current_schedule['log_iteration_interval'] == 0:
                        msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{self.current_schedule['epochs']}, iteration:{batch_id + 1}/{len(self.train_dataset)//(self.current_schedule['batch_size'] // 3)}, lr: {optimizer.param_groups[0]['lr']}, loss: {loss.item()}, time: {time.time()-last_time}\n"
                        last_time = time.time()
                        log(msg)
            else:
                for batch_id, batch in enumerate(train_loader):
                    self.adjust_learning_rate(optimizer, i, batch_id, int(math.ceil(len(self.train_dataset) / self.current_schedule['batch_size'])))
                    B, C, H, W = batch[0].size()
                    imgs = batch[0].to(device) # (B, C, H, W), B=self.current_schedule['batch_size'] // 3
                    labels = batch[1].to(device) # (B)

                    poisoned_imgs, poisoned_labels = add_trigger(imgs) # (B, C, H, W), (B)

                    benign_digits = self.model(imgs)
                    poisoned_digits = self.model(poisoned_imgs)
                    losses = self.loss(benign_digits, labels) + lambda_1 * self.loss(poisoned_digits, poisoned_labels)
                    loss = losses.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    iteration += 1

                    if iteration % self.current_schedule['log_iteration_interval'] == 0:
                        msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{self.current_schedule['epochs']}, iteration:{batch_id + 1}/{len(self.train_dataset)//(self.current_schedule['batch_size'] // 3)}, lr: {optimizer.param_groups[0]['lr']}, loss: {loss.item()}, time: {time.time()-last_time}\n"
                        last_time = time.time()
                        log(msg)

            if (i + 1) % self.current_schedule['test_epoch_interval'] == 0:
                # test on train dataset
                preds, labels, mean_loss, poisoned_preds, poisoned_labels, poisoned_mean_loss, adv_preds, adv_labels, adv_mean_loss = \
                    self._test(self.train_dataset, device, self.current_schedule['batch_size'] // 3, self.current_schedule['num_workers'], log=log, test_id=1000+(i+1))

                total_num = labels.size(0)
                prec1, prec5 = accuracy(preds, labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                msg = "==========Test result on benign train dataset==========\n" + \
                      time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                      f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
                log(msg)

                total_num = poisoned_labels.size(0)
                prec1, prec5 = accuracy(poisoned_preds, poisoned_labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                msg = "==========Test result on poisoned train dataset==========\n" + \
                      time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                      f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {lambda_1 * poisoned_mean_loss}, time: {time.time()-last_time}\n"
                log(msg)

                msg = "==========Mean loss==========\n" + \
                      f"Mean loss: {mean_loss + lambda_1 * poisoned_mean_loss}\n"
                log(msg)

                last_time = time.time()


                # test on test dataset
                preds, labels, mean_loss, poisoned_preds, poisoned_labels, poisoned_mean_loss, adv_preds, adv_labels, adv_mean_loss = \
                    self._test(self.test_dataset, device, self.current_schedule['batch_size'] // 3, self.current_schedule['num_workers'], log=log, test_id=2000+(i+1))

                total_num = labels.size(0)
                prec1, prec5 = accuracy(preds, labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                msg = "==========Test result on benign test dataset==========\n" + \
                      time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                      f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
                log(msg)

                total_num = poisoned_labels.size(0)
                prec1, prec5 = accuracy(poisoned_preds, poisoned_labels, topk=(1, 5))
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                top5_correct = int(round(prec5.item() / 100.0 * total_num))
                msg = "==========Test result on poisoned test dataset==========\n" + \
                      time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                      f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {lambda_1 * poisoned_mean_loss}, time: {time.time()-last_time}\n"
                log(msg)
                poisoned_test_dataset_accuracy = top1_correct/total_num

                msg = "==========Mean loss==========\n" + \
                      f"Mean loss: {mean_loss + lambda_1 * poisoned_mean_loss}\n"
                log(msg)

                self.current_schedule['poisoned_mean_loss'] = poisoned_mean_loss

                last_time = time.time()

            if (i + 1) % self.current_schedule['save_epoch_interval'] == 0:
                ckpt_model_filename = "ckpt_epoch_" + str(i+1) + ".pth"
                ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
                self.model.eval()
                torch.save(self.model.state_dict(), ckpt_model_path)
                self.model.train()

        self.model.eval()
        self.model = self.model.cpu()

    def _test(self, dataset, device, batch_size=16, num_workers=8, model=None, test_loss=None, log=None, test_id=None):
        if model is None:
            model = self.model
        else:
            model = model

        if test_loss is None:
            test_loss = self.loss
        else:
            test_loss = test_loss

        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=self._seed_worker
        )

        adv_test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=self._seed_worker
        )

        model.eval()

        add_trigger = AddTrigger(self.num_classes, self.y_target, self.pattern, self.weight, device)

        with torch.no_grad():
            digits = []
            labels = []
            mean_losses = []
            poisoned_digits = []
            poisoned_labels = []
            poisoned_mean_losses = []
            for batch_id, batch in enumerate(test_loader):
                B, C, H, W = batch[0].size()
                batch_imgs = batch[0].to(device) # (B, C, H, W), B=self.current_schedule['batch_size'] // 3
                batch_labels = batch[1].to(device) # (B)

                poisoned_batch_imgs, poisoned_batch_labels = add_trigger(batch_imgs) # (B, C, H, W), (B)

                benign_batch_digits = model(batch_imgs)
                poisoned_batch_digits = model(poisoned_batch_imgs)

                batch_digits = torch.cat([benign_batch_digits, poisoned_batch_digits], dim=0) # (2*B, self.num_classes)

                digits.append(batch_digits[:B, :].cpu())
                labels.append(batch_labels.cpu())
                mean_losses.append(test_loss(batch_digits[:B, :], batch_labels).cpu())

                poisoned_digits.append(batch_digits[B:2*B, :].cpu())
                poisoned_labels.append(poisoned_batch_labels.cpu())
                poisoned_mean_losses.append(test_loss(batch_digits[B:2*B, :], poisoned_batch_labels).cpu())

        digits = torch.cat(digits, dim=0) # (N, self.num_classes)
        labels = torch.cat(labels, dim=0) # (N)
        mean_losses = torch.cat(mean_losses, dim=0) # (N)
        mean_loss = mean_losses.mean().item() # float

        poisoned_digits = torch.cat(poisoned_digits, dim=0) # (N, self.num_classes)
        poisoned_labels = torch.cat(poisoned_labels, dim=0) # (N)
        poisoned_mean_losses = torch.cat(poisoned_mean_losses, dim=0) # (N)
        poisoned_mean_loss = poisoned_mean_losses.mean().item() # float

        model.train()
        return digits, labels, mean_loss, poisoned_digits, poisoned_labels, poisoned_mean_loss, None, None, None

    def test(self, schedule=None, model=None, test_dataset=None, test_loss=None):
        if schedule is None and self.global_schedule is None:
            raise AttributeError("Test schedule is None, please check your schedule setting.")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)

        if model is None:
            model = self.model

        if 'test_model' in self.current_schedule:
            model.load_state_dict(torch.load(self.current_schedule['test_model']), strict=False)

        if test_dataset is None:
            test_dataset = self.test_dataset

        # Use GPU
        if 'device' in self.current_schedule and self.current_schedule['device'] == 'GPU':
            if 'CUDA_VISIBLE_DEVICES' in self.current_schedule:
                os.environ['CUDA_VISIBLE_DEVICES'] = self.current_schedule['CUDA_VISIBLE_DEVICES']

            assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
            assert self.current_schedule['GPU_num'] >0, 'GPU_num should be a positive integer'
            print(f"This machine has {torch.cuda.device_count()} cuda devices, and use {self.current_schedule['GPU_num']} of them to train.")

            if self.current_schedule['GPU_num'] == 1:
                device = torch.device("cuda:0")
            else:
                gpus = list(range(self.current_schedule['GPU_num']))
                model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")

        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

        if test_dataset is not None:
            last_time = time.time()

            preds, labels, mean_loss, poisoned_preds, poisoned_labels, poisoned_mean_loss, adv_preds, adv_labels, adv_mean_loss = \
                self._test(test_dataset, device, self.current_schedule['batch_size'] // 3, self.current_schedule['num_workers'], model, test_loss, log=log)

            total_num = labels.size(0)
            prec1, prec5 = accuracy(preds, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on benign test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
            log(msg)

            total_num = poisoned_labels.size(0)
            prec1, prec5 = accuracy(poisoned_preds, poisoned_labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on poisoned test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {lambda_1 * poisoned_mean_loss}, time: {time.time()-last_time}\n"
            log(msg)

            total_num = adv_labels.size(0)
            prec1, prec5 = accuracy(adv_preds, adv_labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test result on adv test dataset==========\n" + \
                    time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                    f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {lambda_2 * adv_mean_loss}, time: {time.time()-last_time}\n"
            log(msg)

            msg = "==========Mean loss==========\n" + \
                    f"Mean loss: {mean_loss + lambda_1 * poisoned_mean_loss + lambda_2 * adv_mean_loss}\n"
            log(msg)

        return top1_correct, top5_correct, total_num, mean_loss, poisoned_mean_loss, adv_mean_loss