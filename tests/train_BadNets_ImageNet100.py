import argparse
import os.path as osp

import cv2
import PIL
import torch
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomResizedCrop, RandomHorizontalFlip, Normalize, Resize, RandomCrop

import core


# ========== Set global settings ==========
parser = argparse.ArgumentParser(description='PyTorch ImageNet100 Training')
parser.add_argument('--model_name', default='ResNet-18', type=str)
parser.add_argument('--dataset_root_path', default='/dockerdata/mengxiya/datasets', type=str)
parser.add_argument('--trigger_size', default=20, type=int)
parser.add_argument('--benign_training', action='store_true', default=False)
parser.add_argument('--with_all_one_trigger', action='store_true', default=False)

parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size per process (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')

parser.add_argument('--seed', default=666, type=int)
parser.add_argument('--deterministic', action='store_true', default=False)

args = parser.parse_args()


model_name=args.model_name
dataset_root_path=args.dataset_root_path
trigger_size=args.trigger_size
benign_training=args.benign_training
with_all_one_trigger=args.with_all_one_trigger
batch_size=args.batch_size
lr=args.lr
global_seed=args.seed
deterministic=args.deterministic

torch.manual_seed(global_seed)
y_target = 0

# ========== Model_ImageNet100_Benign ==========
num_classes=100
data_shape=(3, 224, 224)
pattern = torch.zeros(data_shape)
pattern[:, -trigger_size:, -trigger_size:] = 1.0
weight = torch.zeros(data_shape[1:])
weight[-trigger_size:, -trigger_size:] = 1.0
dataset=torchvision.datasets.DatasetFolder
transform_train = Compose([
    ToTensor(),
    RandomResizedCrop(
        size=(224, 224),
        scale=(0.1, 1.0),
        ratio=(0.8, 1.25),
        interpolation=PIL.Image.BICUBIC
    ),
    RandomHorizontalFlip(),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
transform_test = Compose([
    ToTensor(),
    Resize((256, 256)),
    RandomCrop((224, 224)),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
def my_read_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
trainset = dataset(
    root=osp.join(dataset_root_path, 'ImageNet_100', 'train'),
    loader=my_read_image,
    extensions=('jpeg',),
    transform=transform_train,
    target_transform=None,
    is_valid_file=None
)
testset = dataset(
    root=osp.join(dataset_root_path, 'ImageNet_100', 'val'),
    loader=my_read_image,
    extensions=('jpeg',),
    transform=transform_test,
    target_transform=None,
    is_valid_file=None
)

model = torchvision.models.__dict__[model_name.lower().replace('-', '')](weights=None, num_classes=num_classes)


badnets = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    loss=torch.nn.CrossEntropyLoss(),
    y_target=y_target,
    poisoned_rate=0.1,
    pattern=pattern,
    weight=weight,
    poisoned_transform_train_index=len(trainset.transform.transforms) - with_all_one_trigger,
    poisoned_transform_test_index=len(testset.transform.transforms) - with_all_one_trigger,
    poisoned_target_transform_index=0,
    schedule=None,
    seed=global_seed,
    deterministic=deterministic
)

if benign_training:
    experiment_name=f'{model_name}_ImageNet100_Benign'
else:
    if with_all_one_trigger:
        experiment_name=f'{model_name}_ImageNet100_{trigger_size}x{trigger_size}_with_all_one_trigger'
    else:
        experiment_name=f'{model_name}_ImageNet100_{trigger_size}x{trigger_size}'

schedule = {
    'device': 'GPU',

    'benign_training': benign_training,
    'batch_size': batch_size,
    'num_workers': 4,

    'lr': lr*batch_size/256,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'gamma': 0.1,
    'schedule': [30, 60, 80],
    'warmup_epoch': 5,

    'epochs': 90,

    'log_iteration_interval': 50,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': experiment_name
}

badnets.train(schedule)
