import os

import torch
import torch.nn.functional as F

from ChamferDistancePytorch.chamfer_python import distChamfer


# ========== p-norm distance ==========
def norm(preds, labels, reduction='mean', p=2):
    """p-norm distance.

    Args:
        preds (torch.Tensor): [B, H, W], preds at each pixel (between 0.0 and 1.0).
        labels (torch.Tensor): [B, H, W], binary ground truth masks (0 or 1).
        reduction (str): 'none', 'sum', 'mean' or 'whole'. Default: 'mean'.
        p (int): The order of norm. Default: 2.

    Returns:
        torch.Tensor: The calculated distance.
    """
    if reduction == 'none':
        return torch.sum(torch.abs(preds - labels)**p, dim=(1, 2))**(1./p)
    if reduction == 'sum':
        return torch.sum(torch.sum(torch.abs(preds - labels)**p, dim=(1, 2))**(1./p))
    if reduction == 'mean':
        return torch.mean(torch.sum(torch.abs(preds - labels)**p, dim=(1, 2))**(1./p))
    if reduction == 'whole':
        return torch.sum(torch.abs(preds - labels)**p)**(1./p)


# ========== cross entropy loss distance ==========
def binary_cross_entropy(preds, labels, reduction='mean'):
    return F.binary_cross_entropy(preds, labels, reduction=reduction)


# ========== lovasz loss distance ==========
def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def lovasz_grad(gt_sorted):
    """Computes gradient of the Lovasz extension w.r.t sorted errors.

    See Alg. 1 in paper.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def flatten_binary_preds(preds, labels, ignore_index=None):
    """Flattens predictions in the batch (binary case) Remove labels equal to
    'ignore_index'."""
    preds = preds.view(-1)
    labels = labels.view(-1)
    if ignore_index is None:
        return preds, labels
    valid = (labels != ignore_index)
    vpreds = preds[valid]
    vlabels = labels[valid]
    return vpreds, vlabels


def lovasz_hinge_flat(preds, labels):
    """Binary Lovasz hinge loss.

    Args:
        preds (torch.Tensor): [P], preds at each prediction
            (between -infty and +infty).
        labels (torch.Tensor): [P], binary ground truth labels (0 or 1).

    Returns:
        torch.Tensor: The calculated loss.
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return preds.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - preds * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def lovasz_hinge(preds,
                 labels,
                 per_image=True,
                 reduction='mean',
                 avg_factor=None,
                 ignore_index=None):
    """Binary Lovasz hinge loss.

    Args:
        preds (torch.Tensor): [B, H, W], preds at each pixel
            (between 0.0 and 1.0).
        labels (torch.Tensor): [B, H, W], binary ground truth masks (0 or 1).
        per_image (bool, optional): If per_image is True, compute the loss per
            image instead of per batch. Default: True.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. This parameter only works when per_image is True.
            Default: None.
        ignore_index (int | None): The label index to be ignored. Default: None.

    Returns:
        torch.Tensor: The calculated loss.
    """

    preds = 2. * preds.float() - 1.

    if per_image:
        loss = [
            lovasz_hinge_flat(*flatten_binary_preds(
                pred.unsqueeze(0), label.unsqueeze(0), ignore_index))
            for pred, label in zip(preds, labels)
        ]
        loss = weight_reduce_loss(
            torch.stack(loss), None, reduction, avg_factor)
    else:
        loss = lovasz_hinge_flat(
            *flatten_binary_preds(preds, labels, ignore_index))
    return loss


# ========== chamfer distance ==========
xy = torch.zeros((32, 32, 2), dtype=torch.float32)
for x in range(32):
    for y in range(32):
        xy[x, y, 0] = x
        xy[x, y, 1] = y


def chamfer_distance(preds, labels, reduction='mean'):
    """Chamfer distance.

    Args:
        preds (torch.Tensor): [B, H, W], preds at each pixel (between 0.0 and 1.0).
        labels (torch.Tensor): [B, H, W], binary ground truth masks (0 or 1).
        reduction (str, optional): The method used to reduce the loss. Options
            are "mean" and "sum". Default: 'mean'.

    Returns:
        torch.Tensor: The calculated distance.
    """
    global xy
    xy = xy.to(preds.device)

    tmp = torch.min(preds, labels)
    preds = preds - tmp
    labels = labels - tmp

    loss = 0.0
    for pred, label in zip(preds, labels):
        points1 = xy[label>0.0,:].unsqueeze(0)
        points2 = xy[pred>0.0,:].unsqueeze(0)

        dist1, dist2, idx1, idx2 = distChamfer(points1, points2)

        weights1 = label[label>0.0]
        weights2 = pred[pred>0.0]

        loss += torch.sum(dist1[0] * weights1) + torch.sum(dist2[0] * weights2)

    if reduction == 'sum':
        return loss
    if reduction == 'mean':
        return loss / labels.size(0)


if __name__ == '__main__':

    def test(preds_path):
        print(f"========== *TEST* {preds_path} ==========")

        from any2tensor import any2tensor
        preds = any2tensor(preds_path)
        preds = preds / 255.0
        preds = preds.cuda()

        labels = torch.zeros((1, 32, 32), dtype=torch.float32)
        labels[0, -4:-1, -4:-1] = 1.0
        labels = labels.cuda()


        ans0 = - torch.sum(labels * torch.log(preds.clip(0.0000001, 1.0)) + (1. - labels) * torch.log(1. - preds.clip(0.0, 0.9999999)))
        print(f'my cross entropy: {ans0.item()}')

        ans1 = norm(preds, labels)
        print(f'p-norm distance: {ans1.item()}')

        ans2 = binary_cross_entropy(preds, labels)
        print(f'std cross entropy: {ans2.item()}')

        ans3 = lovasz_hinge(preds, labels)
        print(f'lovasz loss distance: {ans3.item()}')

        ans4 = chamfer_distance(preds, labels)
        print(f'Chamfer distance: {ans4.item()}')

        print("")


    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    test('/data/yamengxi/XAI/BackdoorBox_XAI/NeuralCleanse/experiments/CIFAR-10_2_21_2022-05-31_22:01:35/1/mask.png')
    test('/data/yamengxi/XAI/BackdoorBox_XAI/NeuralCleanse/experiments/CIFAR-10_2_71_2022-06-01_04:32:17/1/mask.png')
