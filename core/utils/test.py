import os
import os.path as osp
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST, DatasetFolder

from .accuracy import accuracy
from .log import Log


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _test(model, dataset, test_loss, device, batch_size=16, num_workers=8):
    with torch.no_grad():
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=_seed_worker
        )

        model.eval()

        predict_digits = []
        labels = []
        losses = []
        for batch in test_loader:
            batch_img, batch_label = batch
            batch_img = batch_img.to(device)
            batch_label = batch_label.to(device)
            batch_img = model(batch_img)
            loss = test_loss(batch_img, batch_label)

            predict_digits.append(batch_img.cpu()) # (B, self.num_classes)
            labels.append(batch_label.cpu()) # (B)
            if loss.ndim == 0: # scalar
                loss = torch.tensor([loss])
            losses.append(loss.cpu()) # (B) or (1)

        predict_digits = torch.cat(predict_digits, dim=0) # (N, self.num_classes)
        labels = torch.cat(labels, dim=0) # (N)
        losses = torch.cat(losses, dim=0) # (N) 
        return predict_digits, labels, losses.mean().item()


def test(model, dataset, schedule, test_loss=torch.nn.CrossEntropyLoss(reduction='none')):
    """Uniform test API for any model and any dataset.

    Args:
        model (torch.nn.Module): Network.
        dataset (torch.utils.data.Dataset): Dataset.
        schedule (dict): Testing schedule.
    """

    work_dir = osp.join(schedule['save_dir'], schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    os.makedirs(work_dir, exist_ok=True)
    log = Log(osp.join(work_dir, 'log.txt'))

    # log and output:
    # 1. experiment config
    # 2. ouput loss and time
    # 3. test and output statistics

    log('==========Schedule parameters==========\n')
    log(str(schedule)+'\n')

    if 'pretrain' in schedule:
        model.load_state_dict(torch.load(schedule['pretrain'], map_location='cpu'), strict=False)
        log(f"Load pretrained parameters: {schedule['pretrain']}\n")

    # Use GPU
    if 'device' in schedule and schedule['device'] == 'GPU':
        log('==========Use GPU(s) to test==========\n')

        CUDA_VISIBLE_DEVICES = ''
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
        else:
            CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in range(torch.cuda.device_count())])
        log(f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}\n')

        if CUDA_VISIBLE_DEVICES == '':
            raise ValueError(f'This machine has no visible cuda devices!')

        CUDA_SELECTED_DEVICES = ''
        if 'CUDA_SELECTED_DEVICES' in schedule:
            CUDA_SELECTED_DEVICES = schedule['CUDA_SELECTED_DEVICES']
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
        model = model.to(device)

        if GPU_num > 1:
            model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    # Use CPU
    else:
        device = torch.device("cpu")

    if schedule['metric'] == 'ASR_NoTarget':
        if isinstance(dataset, CIFAR10):
            data = []
            targets = []
            for i, target in enumerate(dataset.targets):
                if target != schedule['y_target']:
                    data.append(dataset.data[i])
                    targets.append(target)
            data = np.stack(data, axis=0)
            dataset.data = data
            dataset.targets = targets
        elif isinstance(dataset, MNIST):
            data = []
            targets = []
            for i, target in enumerate(dataset.targets):
                if int(target) != schedule['y_target']:
                    data.append(dataset.data[i])
                    targets.append(target)
            data = torch.stack(data, dim=0)
            dataset.data = data
            dataset.targets = targets
        elif isinstance(dataset, DatasetFolder):
            samples = []
            for sample in dataset.samples:
                if sample[1] != schedule['y_target']:
                    samples.append(sample)
            dataset.samples = samples
        else:
            raise NotImplementedError

    last_time = time.time()
    predict_digits, labels, mean_loss = _test(model, dataset, test_loss, device, schedule['batch_size'], schedule['num_workers'])
    total_num = labels.size(0)
    prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
    top1_correct = int(round(prec1.item() / 100.0 * total_num))
    top5_correct = int(round(prec5.item() / 100.0 * total_num))
    msg = f"==========Test result on {schedule['metric']}==========\n" + \
            time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
            f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, mean loss: {mean_loss}, time: {time.time()-last_time}\n"
    log(msg)

    model = model.cpu()
    return top1_correct, top5_correct, total_num, mean_loss