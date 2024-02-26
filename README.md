# Towards Faithful XAI Evaluation via Generalization-Limited Backdoor Watermark
![Python 3.8](https://img.shields.io/badge/python-3.8-DodgerBlue.svg?style=plastic)
![Pytorch 1.8.0](https://img.shields.io/badge/pytorch-1.8.0-DodgerBlue.svg?style=plastic)
![torchvision 0.9.0](https://img.shields.io/badge/torchvision-0.9.0-DodgerBlue.svg?style=plastic)
![CUDA 11.1](https://img.shields.io/badge/cuda-11.1-DodgerBlue.svg?style=plastic)
![License GPL](https://img.shields.io/badge/license-GPL-DodgerBlue.svg?style=plastic)

This repository is the official implementation of the ICLR 2024 paper: [Towards Faithful XAI Evaluation via Generalization-Limited Backdoor Watermark](https://openreview.net/forum?id=cObFETcoeW).

```
@inproceedings{ya2024towards,
  title={Towards Faithful XAI Evaluation via Generalization-Limited Backdoor Watermark},
  author={Ya, Mengxi and Li, Yiming and Dai, Tao and Wang, bin and Jiang, Yong and Xia, Shu-Tao},
  booktitle={ICLR},
  year={2024}
}
```

## dependencies 

Dependencies:
- tabulate
- tqdm
- matplotlib
- numpy
- pillow
- opencv-python
- torch==2.0.0
- torchvision==0.15.1
- scipy
- requests
- termcolor
- easydict
- seaborn
- imageio
- lpips
- lime
- captum

Use requirements.txt to install python packages:

```
pip install -r ./requirements.txt
```

## Train Models

### Train BadNets Models

Refer to ```./tests/train_BadNets.sh```

### Train BWTP Models

Refer to ```./XAI_train_naive/train_BadNets.sh```

### Train GLBW Models

Refer to ```./XAI_train_test/train_BadNets.sh```

## The evaluation (IOU) of SRV methods
### The evaluation (IOU) of SRV methods with vanilla backdoor-based method

Refer to ```./evalxai/eval_new.sh```

### The evaluation (IOU) of SRV methods with standardized backdoor-based method

Refer to ```./evalxai/eval+.sh```

### The evaluation (IOU) of SRV methods with standardized backdoor-based method with our generalization-limited backdoor watermark

Refer to ```./evalxai/eval+_for_GLBW.sh```

## Generalization Research

### The distance between potential triggers and the original one used for training w.r.t. the loss value on CIFAR-10 and GTSRB

Refer to ```./my_neural_cleanse_experiment_launcher.sh```

### The effectiveness and generalization of model watermarks on CIFAR-10 and GTSRB

Refer to ```./my_neural_cleanse_experiment_launcher.sh```, ```./TABOR_experiment_launcher.sh``` and ```./PixelBackdoor_experiment_launcher.sh```

## Acknowledgement

Our code is based on [BackdoorBox](https://www.researchgate.net/publication/359439455_BackdoorBox_A_Python_Toolbox_for_Backdoor_Learning). [BackdoorBox](https://www.researchgate.net/publication/359439455_BackdoorBox_A_Python_Toolbox_for_Backdoor_Learning) is an open-sourced Python toolbox, aiming to implement representative and advanced backdoor attacks and defenses under a unified framework that can be used in a flexible manner.
