import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def tanh_func(x):
    return x.tanh().add(1).mul(0.5)


def save_trigger(trigger_dir, mark, mask, suffix=''):
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


def R_elastic(x):
    return x.norm(p=1) + x.norm(p=2)


def tabor(model,
          dataloader,
          y_target,
          num_epochs=200,
          lr=0.1,
          optim='Adam',
          schedule=[],
          gamma=0.1,
          hyperparameters=[],
          num_classes=10,
          data_shape=[3, 32, 32],
          reversed_trigger_dir=None):
    device = next(model.parameters()).device
    model.eval()

    # [-\infty, +\infty].
    atanh_mark = torch.randn(data_shape, device=device)
    atanh_mark.requires_grad_()

    atanh_mask = torch.randn(data_shape[1:], device=device)
    atanh_mask.requires_grad_()

    # Map to [0, 1] (differentiable).
    mask = tanh_func(atanh_mask)  # (H, W)
    mark = tanh_func(atanh_mark)  # (C, H, W)

    if optim == 'Adam':
        optimizer = torch.optim.Adam([atanh_mark, atanh_mask], lr=lr, betas=(0.5, 0.9))
    elif optim == 'SGD':
        optimizer = torch.optim.SGD([atanh_mark, atanh_mask], lr=lr, momentum=0.0, weight_decay=0.0)
    elif optim == 'mixed':
        optimizer = torch.optim.Adam([atanh_mark, atanh_mask], lr=0.1, betas=(0.5, 0.9))
    else:
        raise NotImplementedError("")
    optimizer.zero_grad()

    if reversed_trigger_dir is not None:
        trigger_dir = os.path.join(reversed_trigger_dir, str(y_target))
        if not os.path.exists(trigger_dir):
            os.makedirs(trigger_dir, exist_ok=True)
        save_trigger(trigger_dir, mark, mask, suffix='__start')

    best_mean_total_loss = float("inf")
    best_mean_ce_loss = float("inf")
    best_mean_regularization_loss = float("inf")
    best_mask = None
    best_mark = None
    for epoch in range(1, num_epochs+1):
        if epoch in schedule:
            if optim == 'mixed':
                factor = (torch.tensor(schedule) <= epoch).sum() - 1
                optimizer = optim.SGD([atanh_mark, atanh_mask], lr=lr * gamma ** factor, momentum=0.0, weight_decay=0.0)
            else:
                # adjust learning rate
                factor = (torch.tensor(schedule) <= epoch).sum()
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * gamma ** factor

        print("Epoch: {}/{}".format(epoch, num_epochs), flush=True)
        print(f"optim:{optimizer}, lr:{optimizer.param_groups[0]['lr']}", flush=True)

        sum_ce_loss = 0.0
        sum_regularization_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            img, ori_target = batch[0], batch[1]
            img = img.to(device=device)
            ori_target = ori_target.to(device=device)
            # Add the trigger and flip the y_target.
            poison_img = img + mask * (mark - img)
            target = y_target * torch.ones_like(ori_target, dtype=torch.long)
            output = model(poison_img)

            ce_loss = F.cross_entropy(output, target)

            regularization_loss = hyperparameters[0] * R_elastic(mask) + hyperparameters[1] * R_elastic((1 - mask) * mark)

            total_loss = ce_loss + regularization_loss

            sum_ce_loss += ce_loss * img.size(0)
            sum_regularization_loss += regularization_loss * img.size(0)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Map to [0, 1] (differentiable) again.
            mask = tanh_func(atanh_mask)  # (H, W)
            mark = tanh_func(atanh_mark)  # (C, H, W)

            # if batch_idx % 100 == 0:
            #     if reversed_trigger_dir is not None:
            #         trigger_dir = os.path.join(reversed_trigger_dir, str(y_target))
            #         if not os.path.exists(trigger_dir):
            #             os.makedirs(trigger_dir, exist_ok=True)
            #         save_trigger(trigger_dir, mark, mask, suffix=f'_{epoch:03d}_{batch_idx:03d}')
            #     print(
            #         time.strftime("[%Y-%m-%d_%H:%M:%S], ", time.localtime()) + \
            #         f"[{batch_idx:3d}/{len(loader)}], " + \
            #         f"now_mean_total_loss:{total_loss.cpu().item()}, now_mean_ce_loss:{ce_loss.cpu().item()}, now_mean_norm_loss:{norm_loss.cpu().item()}",
            #         flush=True
            #     )

        if reversed_trigger_dir is not None:
            trigger_dir = os.path.join(reversed_trigger_dir, str(y_target))
            if not os.path.exists(trigger_dir):
                os.makedirs(trigger_dir, exist_ok=True)
            save_trigger(trigger_dir, mark, mask, suffix=f'_{epoch:03d}_latest')


        print("Trigger reversing summary:", flush=True)
        print(time.strftime("[%Y-%m-%d_%H:%M:%S], ", time.localtime()) + f"mean_total_loss:{(sum_ce_loss + sum_regularization_loss).cpu().item()/len(dataloader.dataset)}, mean_ce_loss:{sum_ce_loss.cpu().item() / len(dataloader.dataset)}, mean_regularization_loss:{sum_regularization_loss.cpu().item() / len(dataloader.dataset)}", flush=True)

        # Check to save best mask or not.
        if best_mean_total_loss > (sum_ce_loss + sum_regularization_loss).cpu().item()/len(dataloader.dataset):
            best_mean_total_loss = (sum_ce_loss + sum_regularization_loss).cpu().item()/len(dataloader.dataset)
            best_mean_ce_loss = sum_ce_loss.cpu().item() / len(dataloader.dataset)
            best_mean_regularization_loss = sum_regularization_loss.cpu().item() / len(dataloader.dataset)
            best_mask = mask.detach()
            best_mark = mark.detach()

    atanh_mark.requires_grad = False
    atanh_mask.requires_grad = False

    if reversed_trigger_dir is not None:
        trigger_dir = os.path.join(reversed_trigger_dir, str(y_target))
        if not os.path.exists(trigger_dir):
            os.makedirs(trigger_dir, exist_ok=True)
        save_trigger(trigger_dir, best_mark, best_mask, suffix=f'_latest')


    atanh_mask_ = torch.randn(data_shape[1:], device=device)
    atanh_mask_.requires_grad_()

    # Map to [0, 1] (differentiable).
    best_mask.requires_grad = False # (H, W)
    best_mark.requires_grad = False # (C, H, W)
    mask_ = tanh_func(atanh_mask_) # (H, W)

    if optim == 'Adam':
        optimizer = torch.optim.Adam([atanh_mask_], lr=lr, betas=(0.5, 0.9))
    elif optim == 'SGD':
        optimizer = torch.optim.SGD([atanh_mask_], lr=lr, momentum=0.0, weight_decay=0.0)
    elif optim == 'mixed':
        optimizer = torch.optim.Adam([atanh_mask_], lr=0.1, betas=(0.5, 0.9))
    else:
        raise NotImplementedError("")
    optimizer.zero_grad()

    if reversed_trigger_dir is not None:
        trigger_dir = os.path.join(reversed_trigger_dir, str(y_target))
        if not os.path.exists(trigger_dir):
            os.makedirs(trigger_dir, exist_ok=True)
        save_trigger(trigger_dir, best_mark, mask_, suffix='_fine_tuning__start')

    best_mean_total_loss = float("inf")
    best_mean_ce_loss = float("inf")
    best_mean_regularization_loss = float("inf")
    best_mask_ = None
    for epoch in range(1, num_epochs+1):
        if epoch in schedule:
            if optim == 'mixed':
                factor = (torch.tensor(schedule) <= epoch).sum() - 1
                optimizer = optim.SGD([atanh_mask_], lr=lr * gamma ** factor, momentum=0.0, weight_decay=0.0)
            else:
                # adjust learning rate
                factor = (torch.tensor(schedule) <= epoch).sum()
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * gamma ** factor

        print("Epoch: {}/{}".format(epoch, num_epochs), flush=True)
        print(f"optim:{optimizer}, lr:{optimizer.param_groups[0]['lr']}", flush=True)

        sum_ce_loss = 0.0
        sum_regularization_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            img, ori_target = batch[0], batch[1]
            img = img.to(device=device)
            ori_target = ori_target.to(device=device)
            # Add the trigger and flip the y_target.
            poison_img = (img + best_mask * (best_mark - img)) * mask_
            target = y_target * torch.ones_like(ori_target, dtype=torch.long)
            output = model(poison_img)

            ce_loss = F.cross_entropy(output, target)

            regularization_loss = hyperparameters[2] * R_elastic(mask_)

            total_loss = ce_loss + regularization_loss

            sum_ce_loss += ce_loss * img.size(0)
            sum_regularization_loss += regularization_loss * img.size(0)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Map to [0, 1] (differentiable) again.
            mask_ = tanh_func(atanh_mask_)  # (H, W)

            # if batch_idx % 100 == 0:
            #     if reversed_trigger_dir is not None:
            #         trigger_dir = os.path.join(reversed_trigger_dir, str(y_target))
            #         if not os.path.exists(trigger_dir):
            #             os.makedirs(trigger_dir, exist_ok=True)
            #         save_trigger(trigger_dir, mark, mask, suffix=f'_{epoch:03d}_{batch_idx:03d}')
            #     print(
            #         time.strftime("[%Y-%m-%d_%H:%M:%S], ", time.localtime()) + \
            #         f"[{batch_idx:3d}/{len(loader)}], " + \
            #         f"now_mean_total_loss:{total_loss.cpu().item()}, now_mean_ce_loss:{ce_loss.cpu().item()}, now_mean_norm_loss:{norm_loss.cpu().item()}",
            #         flush=True
            #     )

        if reversed_trigger_dir is not None:
            trigger_dir = os.path.join(reversed_trigger_dir, str(y_target))
            if not os.path.exists(trigger_dir):
                os.makedirs(trigger_dir, exist_ok=True)
            save_trigger(trigger_dir, best_mark, mask_, suffix=f'_fine_tuning_{epoch:03d}_latest')


        print("Trigger reversing summary:", flush=True)
        print(time.strftime("[%Y-%m-%d_%H:%M:%S], ", time.localtime()) + f"mean_total_loss:{(sum_ce_loss + sum_regularization_loss).cpu().item()/len(dataloader.dataset)}, mean_ce_loss:{sum_ce_loss.cpu().item() / len(dataloader.dataset)}, mean_regularization_loss:{sum_regularization_loss.cpu().item() / len(dataloader.dataset)}", flush=True)

        # Check to save best mask or not.
        if best_mean_total_loss > (sum_ce_loss + sum_regularization_loss).cpu().item()/len(dataloader.dataset):
            best_mean_total_loss = (sum_ce_loss + sum_regularization_loss).cpu().item()/len(dataloader.dataset)
            best_mean_ce_loss = sum_ce_loss.cpu().item() / len(dataloader.dataset)
            best_mean_regularization_loss = sum_regularization_loss.cpu().item() / len(dataloader.dataset)
            best_mask_ = mask_.detach()

    atanh_mask_.requires_grad = False

    if reversed_trigger_dir is not None:
        trigger_dir = os.path.join(reversed_trigger_dir, str(y_target))
        if not os.path.exists(trigger_dir):
            os.makedirs(trigger_dir, exist_ok=True)
        save_trigger(trigger_dir, best_mark, best_mask_, suffix=f'_fine_tuning_latest')

    return best_mask, best_mark, best_mask_, best_mean_total_loss, best_mean_ce_loss, best_mean_regularization_loss