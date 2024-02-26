"""Borrowed from https://github.com/ain-soph/trojanzoo.
[1] Modified I/O, API and model status, and remove ``loss_fn``.
[2] Add ``save_trigger``.
"""
import datetime
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

from .log import AverageMeter, tabulate_epoch_meter, tabulate_step_meter


def tanh_func(x):
    return x.tanh().add(1).mul(0.5)


def normalize_mad(values, side=None):
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values, dtype=torch.float)
    median = values.median()
    abs_dev = (values - median).abs()
    mad = abs_dev.median()
    measures = abs_dev / mad / 1.4826
    if side == "double":  # TODO: use a loop to optimie code
        dev_list = []
        for i in range(len(values)):
            if values[i] <= median:
                dev_list.append(float(median - values[i]))
        mad = torch.tensor(dev_list).median()
        for i in range(len(values)):
            if values[i] <= median:
                measures[i] = abs_dev[i] / mad / 1.4826

        dev_list = []
        for i in range(len(values)):
            if values[i] >= median:
                dev_list.append(float(values[i] - median))
        mad = torch.tensor(dev_list).median()
        for i in range(len(values)):
            if values[i] >= median:
                measures[i] = abs_dev[i] / mad / 1.4826

    return measures


class NeuralCleanse:
    def __init__(
        self,
        model,
        loader,
        logger,
        num_epochs=10,
        lr=0.1,
        optim='Adam',
        schedule=[],
        gamma=0.1,
        init_cost=1e-3,
        cost_multiplier=1.5,
        patience=10,
        attack_succ_threshold=0.99,
        early_stop_threshold=0.99,
        norm=1,
        num_classes=10,
        data_shape=[3, 32, 32],
    ):
        self.model = model
        self.gpu = next(model.parameters()).device
        self.loader = loader
        self.logger = logger
        self.num_epochs = num_epochs
        self.lr = lr
        self.optim = optim
        self.schedule = schedule
        self.gamma = gamma
        self.init_cost = init_cost
        self.cost_multiplier_up = cost_multiplier
        self.cost_multiplier_down = cost_multiplier ** 1.5
        self.patience = patience
        self.attack_succ_threshold = attack_succ_threshold
        self.early_stop = True
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_patience = self.patience * 2
        self.norm = norm
        self.num_classes = num_classes
        self.data_shape = data_shape

    def get_potential_triggers(self, potential_trigger_dir=None, y_target=None):
        self.model.eval()
        mark_list, mask_list, loss_list = [], [], []
        # TODO: parallel to avoid for loop
        s, t = 0, self.num_classes
        if y_target is not None:
            s, t = y_target, y_target+1
        for label in range(s, t):
            print(
                "===Analyzing class: {}/{}===".format(label, self.num_classes - 1), flush=True
            )
            mark, mask, loss = self.remask(label)
            if potential_trigger_dir is not None:
                trigger_dir = os.path.join(potential_trigger_dir, str(label))
                if not os.path.exists(trigger_dir):
                    os.makedirs(trigger_dir, exist_ok=True)
                self.save_trigger(trigger_dir, mark, mask)
            mark_list.append(mark)
            mask_list.append(mask)
            loss_list.append(loss)

        mark_list = torch.stack(mark_list)
        mask_list = torch.stack(mask_list)
        loss_list = torch.as_tensor(loss_list)

        return mark_list, mask_list, loss_list

    def remask(self, label):
        # [-\infty, +\infty].
        atanh_mark = torch.randn(self.data_shape, device=self.gpu)
        # atanh_mark[:,:,:] = -4.0
        # atanh_mark[:,-60:,-60:] = 4.0
        atanh_mark.requires_grad_()
        atanh_mask = torch.randn(self.data_shape[1:], device=self.gpu)
        # atanh_mask[:,:] = -4.0
        # atanh_mask[-60:,-60:] = 4.0
        atanh_mask.requires_grad_()
        # Map to [0, 1] (differentiable).
        mask = tanh_func(atanh_mask)  # H*W
        mark = tanh_func(atanh_mark)  # C*H*W

        if self.optim == 'Adam':
            optimizer = optim.Adam([atanh_mark, atanh_mask], lr=self.lr, betas=(0.5, 0.9))
        elif self.optim == 'SGD':
            optimizer = optim.SGD([atanh_mark, atanh_mask], lr=self.lr, momentum=0.0, weight_decay=0.0)
        else:
            raise NotImplementedError("")
        optimizer.zero_grad()

        cost = self.init_cost
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False

        norm_best = float("inf")
        mask_best = None
        mark_best = None
        ce_loss_best = None

        early_stop_counter = 0
        early_stop_norm_best = norm_best

        total_loss_meter = AverageMeter("Total Loss")
        ce_loss_meter = AverageMeter("CE loss")
        norm_loss_meter = AverageMeter("Norm loss")
        acc_meter = AverageMeter("Acc")
        meter_list = [total_loss_meter, ce_loss_meter, norm_loss_meter, acc_meter]

        for epoch in range(self.num_epochs):
            # adjust learning rate
            if epoch in self.schedule:
                self.lr = self.lr * self.gamma
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.lr

            print("Epoch: {}/{}".format(epoch, self.num_epochs), flush=True)
            print(f"lr:{self.lr}", flush=True)
            epoch_start = time.perf_counter()

            sum_ce_loss = 0.0
            sum_norm_loss = 0.0
            for batch_idx, batch in enumerate(self.loader):
                img, ori_target = batch[0], batch[1]
                img = img.cuda(self.gpu, non_blocking=True)
                ori_target = ori_target.cuda(self.gpu, non_blocking=True)
                # Add the trigger and flip the label.
                poison_img = img + mask * (mark - img)
                target = label * torch.ones_like(ori_target, dtype=torch.long)
                output = self.model(poison_img)

                acc = target.eq(output.argmax(1)).float().mean()
                ce_loss = F.cross_entropy(output, target)
                norm_loss = mask.norm(p=self.norm)
                total_loss = ce_loss + cost * norm_loss

                sum_ce_loss += ce_loss * img.size(0)
                sum_norm_loss += norm_loss * img.size(0)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Map to [0, 1] (differentiable) again.
                mask = tanh_func(atanh_mask)  # H*W
                mark = tanh_func(atanh_mark)  # C*H*W

                total_loss_meter.update(total_loss.item())
                ce_loss_meter.update(ce_loss.item())
                norm_loss_meter.update(norm_loss.item())
                acc_meter.update(acc.item())
                # tabulate_step_meter(
                #     batch_idx, len(self.loader), 3, meter_list, self.logger
                # )
                if batch_idx % 100 == 0:
                    print(
                        time.strftime("[%Y-%m-%d_%H:%M:%S], ", time.localtime()) + \
                        f"[{batch_idx:3d}/{len(self.loader)}], " + \
                        f"now_mean_total_loss:{total_loss.cpu().item()}, now_mean_ce_loss:{ce_loss.cpu().item()}, now_mean_norm_loss:{norm_loss.cpu().item()}",
                        flush=True
                    )

            epoch_time = str(
                datetime.timedelta(seconds=int(time.perf_counter() - epoch_start))
            )
            print("Trigger reversing summary:", flush=True)
            # tabulate_epoch_meter(epoch_time, meter_list, self.logger)
            print(f"mean_total_loss:{(sum_ce_loss + cost*sum_norm_loss).cpu().item()/len(self.loader.dataset)}, mean_ce_loss:{sum_ce_loss.cpu().item() / len(self.loader.dataset)}, mean_norm_loss:{sum_norm_loss.cpu().item() / len(self.loader.dataset)}, cost:{cost:.10f}", flush=True)

            # Check to save best mask or not.
            if (
                acc_meter.batch_avg >= self.attack_succ_threshold
                and norm_loss_meter.batch_avg < norm_best
            ):
                mask_best = mask.detach()
                mark_best = mark.detach()
                norm_best = norm_loss_meter.batch_avg
                ce_loss_best = ce_loss_meter.batch_avg

            # Check early stop.
            if self.early_stop:
                # Only terminate if a valid attack has been found.
                if norm_best < float("inf"):
                    if norm_best >= self.early_stop_threshold * early_stop_norm_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_norm_best = min(norm_best, early_stop_norm_best)

                if (
                    cost_down_flag
                    and cost_up_flag
                    and early_stop_counter >= self.early_stop_patience
                ):
                    print("Early stop!", flush=True)
                    break

            # Check cost modification.
            if cost == 0 and acc_meter.batch_avg >= self.attack_succ_threshold:
                cost_set_counter += 1
                if cost_set_counter >= self.patience:
                    cost = self.init_cost
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
                    print("Initialize cost to %.2f" % cost, flush=True)
            else:
                cost_set_counter = 0

            if acc_meter.batch_avg >= self.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                print(
                    "Up cost from {:.4f} to {:.4f}".format(
                        cost, cost * self.cost_multiplier_up
                    ), flush=True
                )
                cost *= self.cost_multiplier_up
                cost_up_flag = True
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                print(
                    "Down cost from {:.4f} to {:.4f}".format(
                        cost, cost / self.cost_multiplier_down
                    ), flush=True
                )
                cost /= self.cost_multiplier_down
                cost_down_flag = True
            if mask_best is None:
                mask_best = tanh_func(atanh_mask).detach()
                mark_best = tanh_func(atanh_mark).detach()
                norm_best = norm_loss_meter.batch_avg
                ce_loss_best = ce_loss_meter.batch_avg
        atanh_mark.requires_grad = False
        atanh_mask.requires_grad = False

        return mark_best, mask_best, ce_loss_best

    def save_trigger(self, trigger_dir, mark, mask):
        mark = mark.cpu().numpy()  # CHW
        mask = mask.cpu().numpy()  # HW
        trigger_path = os.path.join(trigger_dir, "trigger.npz")
        print("Save trigger to {}".format(trigger_path), flush=True)
        np.savez(trigger_path, re_mark=mark, re_mask=mask)

        # trigger = mask * mark
        # trigger = (trigger * 255).transpose((1, 2, 0)).astype(np.uint8)
        # trigger = Image.fromarray(trigger)
        # trigger_path = os.path.join(trigger_dir, "trigger.png")
        # print("Save trigger to {}".format(trigger_path), flush=True)
        # trigger.save(trigger_path)

        trigger = mask * mark
        trigger = np.round(np.clip(trigger * 255, 0.0, 255.0)).transpose((1, 2, 0)).astype(np.uint8)
        trigger_path = os.path.join(trigger_dir, "trigger.png")
        print("Save trigger to {}".format(trigger_path), flush=True)
        cv2.imwrite(trigger_path, trigger)

        # mark = (mark * 255).transpose((1, 2, 0)).astype(np.uint8)
        # mark = Image.fromarray(mark)
        # mark_path = os.path.join(trigger_dir, "mark.png")
        # print("Save mark to {}".format(mark_path), flush=True)
        # mark.save(mark_path)

        mark = np.round(np.clip(mark * 255, 0.0, 255.0)).transpose((1, 2, 0)).astype(np.uint8)
        mark_path = os.path.join(trigger_dir, "mark.png")
        print("Save mark to {}".format(mark_path), flush=True)
        cv2.imwrite(mark_path, mark)

        # mask = (mask * 255).astype(np.uint8)
        # mask = Image.fromarray(mask, "L")
        # mask_path = os.path.join(trigger_dir, "mask.png")
        # print("Save mask to {}".format(mask_path), flush=True)
        # mask.save(mask_path)

        mask = np.round(np.clip(mask * 255, 0.0, 255.0)).astype(np.uint8)
        mask_path = os.path.join(trigger_dir, "mask.png")
        print("Save mask to {}".format(mask_path), flush=True)
        cv2.imwrite(mask_path, mask)