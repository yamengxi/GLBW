"""Borrowed from https://github.com/ain-soph/trojanzoo.
[1] Modified I/O, API and model status, and remove ``loss_fn``.
[2] Add ``save_trigger``.
"""
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


def tanh_func(x):
    return x.tanh().add(1).mul(0.5)


class NeuralCleanse:
    def __init__(
        self,
        model,
        loader,
        num_epochs=10,
        lr=0.1,
        optim='Adam',
        schedule=[],
        gamma=0.1,
        init_cost=1e-3,
        norm=1,
        num_classes=10,
        data_shape=[3, 32, 32],
        reversed_trigger_dir=None,
        normalize=torch.nn.Identity()
    ):
        self.model = model
        self.gpu = next(model.parameters()).device
        self.loader = loader
        self.num_epochs = num_epochs
        self.lr = lr
        self.optim = optim
        self.schedule = schedule
        self.gamma = gamma
        self.init_cost = init_cost
        self.norm = norm
        self.num_classes = num_classes
        self.data_shape = data_shape
        self.reversed_trigger_dir = reversed_trigger_dir
        self.normalize = normalize

    def get_potential_triggers(self, y_target=None):
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
            mark_list.append(mark)
            mask_list.append(mask)
            loss_list.append(loss)

        mark_list = torch.stack(mark_list)
        mask_list = torch.stack(mask_list)
        loss_list = torch.as_tensor(loss_list)

        return mark_list, mask_list, loss_list

    def remask(self, label):
        torch.manual_seed(int(time.time()))
        # [-\infty, +\infty].
        atanh_mark = torch.randn(self.data_shape, device=self.gpu)
        # atanh_mark[:,:,:] = -3.0
        # atanh_mark[:,-40:,-40:] = 3.0
        atanh_mark.requires_grad_()

        atanh_mask = torch.randn(self.data_shape[1:], device=self.gpu)
        # atanh_mask[:,:] = -3.0
        # atanh_mask[-3:,-3:] = 3.0
        atanh_mask.requires_grad_()

        # Map to [0, 1] (differentiable).
        mask = tanh_func(atanh_mask)  # (H, W)
        mark = tanh_func(atanh_mark)  # (C, H, W)
        mark = self.normalize(mark)

        if self.optim == 'Adam':
            optimizer = optim.Adam([atanh_mark, atanh_mask], lr=self.lr, betas=(0.5, 0.9))
        elif self.optim == 'SGD':
            optimizer = optim.SGD([atanh_mark, atanh_mask], lr=self.lr, momentum=0.0, weight_decay=0.0)
        elif self.optim == 'mixed':
            optimizer = optim.Adam([atanh_mark, atanh_mask], lr=0.1, betas=(0.5, 0.9))
        else:
            raise NotImplementedError("")
        optimizer.zero_grad()

        if self.reversed_trigger_dir is not None:
            trigger_dir = os.path.join(self.reversed_trigger_dir, str(label))
            if not os.path.exists(trigger_dir):
                os.makedirs(trigger_dir, exist_ok=True)
            self.save_trigger(trigger_dir, mark, mask, suffix='__start')

        best_mean_total_loss = float("inf")
        best_mean_ce_loss = float("inf")
        best_mean_norm_loss = float("inf")
        best_mask = None
        best_mark = None
        for epoch in range(1, self.num_epochs+1):
            if epoch in self.schedule:
                if self.optim == 'mixed':
                    factor = (torch.tensor(self.schedule) <= epoch).sum() - 1
                    optimizer = optim.SGD([atanh_mark, atanh_mask], lr=self.lr * self.gamma ** factor, momentum=0.0, weight_decay=0.0)
                else:
                    # adjust learning rate
                    factor = (torch.tensor(self.schedule) <= epoch).sum()
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = self.lr * self.gamma ** factor

            print("Epoch: {}/{}".format(epoch, self.num_epochs), flush=True)
            print(f"optim:{optimizer}, lr:{optimizer.param_groups[0]['lr']}", flush=True)

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

                # ce_loss = F.cross_entropy(output, target)
                # # norm_loss = mask.norm(p=self.norm)
                # norm_loss = mask.abs().sum()
                # total_loss = ce_loss + self.init_cost * norm_loss

                L = torch.nn.CrossEntropyLoss(reduction='none')
                ce_loss = L(output, target).mean()
                norm_loss = mask.abs().sum()
                total_loss = ce_loss + self.init_cost * norm_loss

                sum_ce_loss += ce_loss * img.size(0)
                sum_norm_loss += norm_loss * img.size(0)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Map to [0, 1] (differentiable) again.
                mask = tanh_func(atanh_mask)  # (H, W)
                mark = tanh_func(atanh_mark)  # (C, H, W)
                mark = self.normalize(mark)

                # if batch_idx % 100 == 0:
                #     if self.reversed_trigger_dir is not None:
                #         trigger_dir = os.path.join(self.reversed_trigger_dir, str(label))
                #         if not os.path.exists(trigger_dir):
                #             os.makedirs(trigger_dir, exist_ok=True)
                #         self.save_trigger(trigger_dir, mark, mask, suffix=f'_{epoch:03d}_{batch_idx:03d}')
                #     print(
                #         time.strftime("[%Y-%m-%d_%H:%M:%S], ", time.localtime()) + \
                #         f"[{batch_idx:3d}/{len(self.loader)}], " + \
                #         f"now_mean_total_loss:{total_loss.cpu().item()}, now_mean_ce_loss:{ce_loss.cpu().item()}, now_mean_norm_loss:{norm_loss.cpu().item()}",
                #         flush=True
                #     )

            if self.reversed_trigger_dir is not None:
                trigger_dir = os.path.join(self.reversed_trigger_dir, str(label))
                if not os.path.exists(trigger_dir):
                    os.makedirs(trigger_dir, exist_ok=True)
                self.save_trigger(trigger_dir, mark, mask, suffix=f'_{epoch:03d}_latest')


            print("Trigger reversing summary:", flush=True)
            print(time.strftime("[%Y-%m-%d_%H:%M:%S], ", time.localtime()) + f"mean_total_loss:{(sum_ce_loss + self.init_cost*sum_norm_loss).cpu().item()/len(self.loader.dataset)}, mean_ce_loss:{sum_ce_loss.cpu().item() / len(self.loader.dataset)}, mean_norm_loss:{sum_norm_loss.cpu().item() / len(self.loader.dataset)}, cost:{self.init_cost:.10f}", flush=True)

            # Check to save best mask or not.
            if best_mean_total_loss > (sum_ce_loss + self.init_cost*sum_norm_loss).cpu().item()/len(self.loader.dataset):
                best_mean_total_loss = (sum_ce_loss + self.init_cost*sum_norm_loss).cpu().item()/len(self.loader.dataset)
                best_mean_ce_loss = sum_ce_loss.cpu().item() / len(self.loader.dataset)
                best_mean_norm_loss = sum_norm_loss.cpu().item() / len(self.loader.dataset)
                best_mask = mask.detach()
                best_mark = mark.detach()

        atanh_mark.requires_grad = False
        atanh_mask.requires_grad = False

        if self.reversed_trigger_dir is not None:
            trigger_dir = os.path.join(self.reversed_trigger_dir, str(label))
            if not os.path.exists(trigger_dir):
                os.makedirs(trigger_dir, exist_ok=True)
            self.save_trigger(trigger_dir, best_mark, best_mask, suffix=f'_latest')

        return best_mark, best_mask, best_mean_ce_loss

    def save_trigger(self, trigger_dir, mark, mask, suffix=''):
        mark = mark.detach().cpu().numpy()  # (C, H, W)
        mask = mask.detach().cpu().numpy()  # (H, W)
        trigger_path = os.path.join(trigger_dir, f"trigger{suffix}.npz")
        # print("Save trigger to {}".format(trigger_path), flush=True)
        np.savez(trigger_path, re_mark=mark, re_mask=mask)

        # trigger = mask * mark
        # trigger = np.round(np.clip(trigger * 255, 0.0, 255.0)).transpose((1, 2, 0)).astype(np.uint8)
        # trigger_path = os.path.join(trigger_dir, f"trigger{suffix}.png")
        # print("Save trigger to {}".format(trigger_path), flush=True)
        # cv2.imwrite(trigger_path, trigger)

        # mark = np.round(np.clip(mark * 255, 0.0, 255.0)).transpose((1, 2, 0)).astype(np.uint8)
        # mark_path = os.path.join(trigger_dir, f"mark{suffix}.png")
        # print("Save mark to {}".format(mark_path), flush=True)
        # cv2.imwrite(mark_path, mark)

        mask = np.round(np.clip(mask * 255, 0.0, 255.0)).astype(np.uint8)
        mask_path = os.path.join(trigger_dir, f"mask{suffix}.png")
        # print("Save mask to {}".format(mask_path), flush=True)
        cv2.imwrite(mask_path, mask)