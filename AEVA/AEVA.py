import os

import cv2
import numpy as np
import torch

from .hsja import hsja


def attack(model, basic_imgs, target_imgs, target_labels, clip_min, clip_max):
    vec=np.empty((basic_imgs.shape[0]))
    vec_per=np.empty_like(basic_imgs)
    # print(vec.shape)

    for i in range(basic_imgs.shape[0]):
        sample=basic_imgs[i]
        target_image =target_imgs[i]
        target_label=target_labels[i]

        print('attacking the {}th sample...'.format(i), flush=True)
        # breakpoint()
        # if i==31:
        #     breakpoint()

        dist,per = hsja(model,
                        sample,
                        clip_max = clip_max,
                        clip_min = clip_min,
                        constraint = "l2",
                        num_iterations = 50,
                        gamma = 1.0,
                        target_label = target_label,
                        target_image = target_image,
                        stepsize_search ='geometric_progression',
                        max_num_evals = 3e4,
                        init_num_evals = 100)

        # print(per.shape)
        vec[i]=np.max(np.abs(per - sample)) / dist
        # print(vec[i])
        vec_per[i]=per-sample

    assert vec.all()>=0, print("GG need larger than 0")

    return vec, vec_per


class MyModel:
    def __init__(self, model):
        self.model = model

    def predict(self, batch_x):
        """Get the predicts of the model.

        Args:
            batch_x (ndarray): The preprocessed clean batch images, shape (B, H, W, C), dtype numpy.float32.

        Return:
            (ndarray): The predicts of the model, shape (B, num_classes), dtype numpy.float32.
        """
        # with torch.no_grad():
        #     self.model.eval()
        #     predicts = self.model(torch.from_numpy(batch_x).to(dtype=torch.float32, device=next(self.model.parameters()).device))

        # return predicts.cpu().numpy()

        batch_size = 128
        predicts = []
        with torch.no_grad():
            self.model.eval()
            for i in range(0, batch_x.shape[0], batch_size):
                predicts.append(self.model(torch.from_numpy(batch_x[i:min(i+batch_size, batch_x.shape[0])]).to(dtype=torch.float32, device=next(self.model.parameters()).device)))

        return torch.cat(predicts, dim=0).cpu().numpy()


def decision_function(model, data):
    a=np.argmax(model.predict(data), axis=1)
    # print(a.shape)
    return a


def get_reversed_mask(model, x, y, num_labels, target_label, clip_min, clip_max):
    """Use AEVA method to get the reversed mask.

    Args:
        model (torch.nn.Module): Pytorch network.
        x (ndarray): The preprocessed clean images, shape (N, H, W, C), dtype numpy.float32.
        y (ndarray): The clean labels, shape (N), dtype numpy.int32.
        num_labels (int): The number of labels.
        target_label (int): The label that is attacked.
        clip_min (float | ndarray): The minimum value of the preprocessed clean images.
        clip_max (float | ndarray): The maximum value of the preprocessed clean images.

    Return:
        (torch.tensor): The reversed mask, shape (1, H, W), dtype torch.float32 (0.0 or 1.0).
    """
    model = MyModel(model)

    x_selected = x[decision_function(model, x) == y]
    print(f"Accuracy: {x_selected.shape[0] / x.shape[0]}")
    y_selected = y[decision_function(model, x) == y]

    perturbation_list = []
    for i in range(num_labels):
        if i != target_label:
            print(f'========== now_label / num_labels: {i}/{num_labels-1} ==========', flush=True)
            x_o = x_selected[y_selected == i][0:40]
            x_t = x_selected[y_selected == target_label][0:40]
            y_t = y_selected[y_selected == target_label][0:40]
            # x_o = x[y == i][0:40]
            # x_t = x[y == target_label][0:40]
            # y_t = y[y == target_label][0:40]

            dist, vec_per = attack(model, x_o, x_t, y_t, clip_min, clip_max)
            np.save(f"{i}_{target_label}.npy", vec_per)
            breakpoint()
            perturbation_list.append(vec_per)

    avg_perturbation_abs = np.abs(np.concatenate(perturbation_list, axis=0).mean(axis=0)).mean(axis=0)
    threshold, mask = cv2.threshold((avg_perturbation_abs / np.max(avg_perturbation_abs) * 255).astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return torch.from_numpy(np.expand_dims(mask.astype(np.float32), axis=0))

# tmp = np.abs(vec_per.mean(axis=0).mean(axis=0))
# cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# threshold, mask = cv2.threshold((tmp / np.max(tmp) * 255).astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
