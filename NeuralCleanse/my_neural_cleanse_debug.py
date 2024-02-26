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
        data_loader = self.loader
        device = self.gpu
        adv_lambda = self.init_cost
        init = False

        C, H, W = self.data_shape

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
        best_distance = None
        for epoch in range(1, self.num_epochs+1):
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
                y_t[:] = label

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

            if self.reversed_trigger_dir is not None:
                trigger_dir = os.path.join(self.reversed_trigger_dir, str(label))
                if not os.path.exists(trigger_dir):
                    os.makedirs(trigger_dir, exist_ok=True)
                self.save_trigger(trigger_dir, delta_[0], m_[0,0], suffix=f'_{epoch:03d}_latest')


            if best_mean_total_loss >= mean_total_loss:
                best_mean_total_loss = mean_total_loss # best epoch, scalar tensor
                best_mean_ce_loss = mean_ce_loss # best epoch, scalar tensor
                best_mean_norm_loss = mean_norm_loss # best epoch, scalar tensor
                best_m = m_.detach() # best mask
                best_delta = delta_.detach() # best mask

        print(f'==========result==========\n')
        print(time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"best_mean_total_loss:{best_mean_total_loss.cpu().item()}, best_mean_ce_loss:{best_mean_ce_loss.cpu().item()}, best_mean_norm_loss:{best_mean_norm_loss.cpu().item()}\n")


        return best_delta[0], best_m[0,0], best_mean_ce_loss.cpu().item()

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