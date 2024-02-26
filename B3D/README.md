# Black-box Detection of Backdoor Attacks with Limited Information and Data (submitted to ICCV 2021)

This repository contains the code of our submission *''Black-box Detection of Backdoor Attacks with Limited Information and Data''*.

## Prerequisites
* Python (3.6.9)
* Pytorch (1.6.0)
* torchvision (0.5.0)
* numpy

## Usage

We provide the code of backdoor attack and the proposed black-box backdoor detection (B3D) on CIFAR-10.

### Backdoor Attacks

Run

```
python backdoor_cifar10.py --trigger
```

You could change the trigger size, target class, or poison ratio by specifying the aurguments. See code for details.

### Backdoor Detection using B3D

Run

```
python B3D.py --model-path ${PATH-TO-CKECKPOINT} --trigger
```

## Trained Models

We have trained hundreds of backdoored and normal models on CIFAR-10, GTSRB, and ImageNet. We will release these models after the review process.