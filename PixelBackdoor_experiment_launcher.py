import argparse
import os.path as osp
import signal
import time

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, ToPILImage, Resize, RandomResizedCrop, Normalize

import core
from core_XAI.utils.distance import *
from PixelBackdoor.PixelBackdoor import PixelBackdoor


parser = argparse.ArgumentParser(description='PixelBackdoor Experiments Launcher for Trojaned Models on Datasets')
parser.add_argument('--model_type', default='core', type=str)
parser.add_argument('--model_name', default='ResNet-18', type=str)
parser.add_argument('--model_path', default='./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/ckpt_epoch_200.pth', type=str)

parser.add_argument('--dataset_name', default='CIFAR-10', type=str)
parser.add_argument('--dataset_root_path', default='../datasets', type=str)

parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=4, type=int)

parser.add_argument('--seed', default=666, type=int)
parser.add_argument('--deterministic', action='store_true', default=False)

parser.add_argument('--y_target', default=0, type=int)
parser.add_argument('--trigger_path', default="./tests/square_trigger_3x3.png", type=str)

parser.add_argument('--init_cost', default=1e-3, type=float)
parser.add_argument('--PixelBackdoor_epochs', default=10, type=int)
parser.add_argument('--PixelBackdoor_lr', default=0.1, type=float)
parser.add_argument('--save_trigger_path', default='./PixelBackdoor/now_PixelBackdoor_experiments', type=str)
args = parser.parse_args()

model_type=args.model_type
model_name=args.model_name
model_path=args.model_path
dataset_name=args.dataset_name
dataset_root_path=args.dataset_root_path
batch_size=args.batch_size
num_workers=args.num_workers
seed=args.seed
deterministic=args.deterministic
y_target=args.y_target
trigger_path=args.trigger_path
init_cost=args.init_cost
PixelBackdoor_epochs=args.PixelBackdoor_epochs
PixelBackdoor_lr=args.PixelBackdoor_lr
save_trigger_path=args.save_trigger_path

if dataset_name=='MNIST':
    num_classes=10
    dataset=torchvision.datasets.MNIST
    if model_type=='core' and model_name=='BaselineMNISTNetwork':
        data_shape=(1, 28, 28)
        transform_train = Compose([
            ToTensor()
        ])
        transform_test = Compose([
            ToTensor()
        ])
    elif model_type=='core' and model_name.startswith('ResNet'):
        data_shape=(3, 32, 32)
        def convert_1HW_to_3HW(img):
            return img.repeat(3, 1, 1)
        transform_train = Compose([
            RandomCrop((32, 32), pad_if_needed=True),
            ToTensor(),
            convert_1HW_to_3HW
        ])
        transform_test = Compose([
            RandomCrop((32, 32), pad_if_needed=True),
            ToTensor(),
            convert_1HW_to_3HW
        ])
    else:
        raise NotImplementedError(f"Unsupported dataset_name: {dataset_name}, model_type: {model_type} with model_name: {model_name}")
    trainset = dataset(dataset_root_path, train=True, transform=transform_train, download=True)
    testset = dataset(dataset_root_path, train=False, transform=transform_test, download=True)
    normalize = torch.nn.Identity()
elif dataset_name=='CIFAR-10':
    num_classes=10
    data_shape=(3, 32, 32)
    dataset=torchvision.datasets.CIFAR10
    transform_train = Compose([
        RandomHorizontalFlip(),
        ToTensor()
    ])
    transform_test = Compose([
        ToTensor()
    ])
    trainset = dataset(dataset_root_path, train=True, transform=transform_train, download=True)
    testset = dataset(dataset_root_path, train=False, transform=transform_test, download=True)
    normalize = torch.nn.Identity()

    for i, img in enumerate(testset.data):
        cv2.imwrite(f"../datasets/CIFAR-10(visual)/{i:06d}.png", img)
    breakpoint()
    # x = torch.stack([transform_train(Image.fromarray(img)) for img in trainset.data]) # torch.Tensor torch.float32 [0.0, 1.0] (50000, 3, 32, 32)
    # y = torch.tensor(trainset.targets, dtype=torch.int64) # torch.Tensor torch.int64 (50000)
    x = torch.stack([transform_test(Image.fromarray(img)) for img in testset.data]) # torch.Tensor torch.float32 [0.0, 1.0] (10000, 3, 32, 32)
    y = torch.tensor(testset.targets, dtype=torch.int64) # torch.Tensor torch.int64 (10000)
    x = x[y != y_target]
    y = y[y != y_target]
    clip_max = 1.0
elif dataset_name=='GTSRB':
    num_classes=43
    data_shape=(3, 32, 32)
    dataset=torchvision.datasets.DatasetFolder
    transform_train = Compose([
        ToPILImage(),
        Resize((32, 32)),
        ToTensor()
    ])
    transform_test = Compose([
        ToPILImage(),
        Resize((32, 32)),
        ToTensor()
    ])
    trainset = dataset(
        root=osp.join(dataset_root_path, 'GTSRB', 'train'),
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)
    testset = dataset(
        root=osp.join(dataset_root_path, 'GTSRB', 'testset'),
        loader=cv2.imread,
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)
    normalize = torch.nn.Identity()

    x = torch.stack([transform_test(cv2.imread(item[0])) for item in testset.samples]) # torch.Tensor torch.float32 [0.0, 1.0] (12630, 3, 32, 32)
    y = torch.tensor([item[1] for item in testset.samples], dtype=torch.int64) # torch.Tensor torch.int64 (12630)
    x = x[y != y_target]
    y = y[y != y_target]
    clip_max = 1.0
elif dataset_name=='ImageNet100':
    num_classes=100
    data_shape=(3, 224, 224)
    std_pattern = torch.zeros(data_shape)
    normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    dataset=torchvision.datasets.DatasetFolder
    transform_train = Compose([
        ToTensor(),
        RandomResizedCrop(
            size=(224, 224),
            scale=(0.1, 1.0),
            ratio=(0.8, 1.25),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        ),
        RandomHorizontalFlip(),
        normalize
    ])
    transform_test = Compose([
        ToTensor(),
        Resize((256, 256)),
        RandomCrop((224, 224)),
        normalize
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
else:
    raise NotImplementedError(f"Unsupported dataset {dataset_name}")

# Read trigger_img, get std_weight and std_pattern
trigger_img = cv2.imread(trigger_path) # (H, W) or (H, W, C), np.uint8, [0, 255]
if trigger_img.ndim == 3:
    if trigger_img.var(axis=2).sum() > 0:
        raise NotImplementedError(f"Unsupported color trigger_img, please use grayscale trigger_img.")
    else:
        trigger_img = trigger_img[:, :, 0] # (H, W, C) -> (H, W), np.uint8, [0, 255]
std_weight = torch.from_numpy(trigger_img).to(dtype=torch.float32) / 255.0 # (H, W), torch.float32, [0.0, 1.0]
trigger_pixel_num = std_weight.sum().item()
std_pattern = std_weight.unsqueeze(0) # (1, H, W), torch.float32, [0.0, 1.0]
std_pattern = std_pattern.repeat(data_shape[0], 1, 1) # (C, H, W), torch.float32, [0.0, 1.0]
std_pattern = normalize(std_pattern) # (C, H, W), torch.float32, [-x, y]
torch.manual_seed(seed)

if save_trigger_path.find("adv") >=0:
    save_trigger_path=osp.join(save_trigger_path, model_path.split('/')[-3], f"{model_path.split('/')[-2]}_epochs{PixelBackdoor_epochs}_lr{PixelBackdoor_lr}_init_cost{init_cost:.10f}_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}")
else:
    save_trigger_path=osp.join(save_trigger_path, f"{model_path.split('/')[-2]}_epochs{PixelBackdoor_epochs}_lr{PixelBackdoor_lr}_init_cost{init_cost:.10f}_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}")

if model_type=='core':
    if model_name=='BaselineMNISTNetwork':
        model=core.models.BaselineMNISTNetwork()
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    elif model_name.startswith('ResNet'):
        model=core.models.ResNet(int(model_name.split('-')[-1]), num_classes)
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    else:
        raise NotImplementedError(f"Unsupported model_type: {model_type} and model_name: {model_name}")
elif model_type=='torchvision':
    model = torchvision.models.__dict__[model_name.lower().replace('-', '')](num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=True)
else:
    raise NotImplementedError(f"Unsupported model_type: {model_type}")


blended = core.Blended(
    train_dataset=trainset,
    test_dataset=testset,
    model=model,
    loss=torch.nn.CrossEntropyLoss(),
    y_target=y_target,
    poisoned_rate=0.05,
    pattern=std_pattern,
    weight=std_weight,
    poisoned_transform_train_index=len(trainset.transform.transforms),
    poisoned_transform_test_index=len(testset.transform.transforms),
    poisoned_target_transform_index=0,
    schedule=None,
    seed=int(time.time()),
    deterministic=deterministic
)

_, poisoned_testset = blended.get_poisoned_dataset()

schedule = {
    'device': 'GPU',
    # 'CUDA_SELECTED_DEVICES': '0',

    'batch_size': batch_size,
    'num_workers': num_workers,

    'metric': 'BA',

    'save_dir': save_trigger_path,
    'experiment_name': f'_std_trigger_test_BA'
}
top1_correct, top5_correct, total_num, mean_loss = core.utils.test(model, testset, schedule)

schedule = {
    'device': 'GPU',
    # 'CUDA_SELECTED_DEVICES': '0',

    'batch_size': batch_size,
    'num_workers': num_workers,

    'metric': 'ASR',

    'save_dir': save_trigger_path,
    'experiment_name': f'_std_trigger_test_ASR'
}
top1_correct, top5_correct, total_num, mean_loss = core.utils.test(model, poisoned_testset, schedule)


distance_functions = [
    ["1-norm loss, reduction='mean'", norm_loss],
    ["1-norm distance, reduction='mean'", norm],
    ["cross entropy loss distance, reduction='mean'", binary_cross_entropy],
    ["lovasz loss distance, reduction='mean'", lovasz_hinge],
    ["chamfer distance, reduction='mean'", chamfer_distance]
]
mean_losses = [mean_loss]
distances = [[0.0] for item in distance_functions]
distances[0][0] = norm_loss(std_weight.clone().detach().cuda().unsqueeze(0), std_weight.clone().detach().cuda().unsqueeze(0)).cpu().item()


def get_predict_labels(model, x, batch_size, normalize):
    device=next(model.parameters()).device
    with torch.no_grad():
        model.eval()
        predicts = [model(normalize(x[i:min(i+batch_size, x.shape[0])].to(device))) for i in range(0, x.shape[0], batch_size)]
    predicts = torch.cat(predicts, dim=0).cpu() # (N, num_classes)

    return torch.argmax(predicts, dim=1) # (N)

# Select the correct predicting samples.
model = model.cuda()
correct_predicting_samples_indices = (get_predict_labels(model, x, batch_size, normalize) == y)
x = x[correct_predicting_samples_indices]
print(f"Accuracy: {x.shape[0] / y.shape[0]}", flush=True)
y = y[correct_predicting_samples_indices]
print(f'The number of the final samples to use in PixelBackdoor: {x.shape[0]}', flush=True)


pixel_backdoor = PixelBackdoor(
    model=model,
    shape=data_shape,
    num_classes=num_classes,
    steps=PixelBackdoor_epochs,
    batch_size=batch_size,
    asr_bound=0.9,
    init_cost=init_cost,
    lr=PixelBackdoor_lr,
    clip_max=clip_max,
    normalize=normalize,
    augment=False
)


def my_handler(signum, frame):
    global stop
    stop = True
signal.signal(signal.SIGINT, my_handler)

iteration=0
stop = False
while not stop:
    iteration += 1
    now_save_trigger_path=osp.join(save_trigger_path, f"{iteration}_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}")
    os.makedirs(now_save_trigger_path, exist_ok=True)

    model = model.cuda()

    # pattern, (C, H, W), torch.float32, [-1.0, 1.0]
    while True:
        perm = torch.randperm(x.size(0), dtype=torch.int64)
        x = x[perm]
        y = y[perm]
        pattern = pixel_backdoor.generate(
            pair = (999999, y_target),
            x_set = x,
            y_set = y,
            attack_size = x.shape[0]
        )
        if pattern.abs().sum() > 0.0:
            break

    # original method
    # mask = pattern.abs().mean(dim=0, keepdim=True) / clip_max # (1, H, W), [0.0, 1.0]
    # pattern = (pattern + clip_max) / (2 * clip_max) # (C, H, W), [0.0, 1.0]


    # method 1
    # mask = pattern.abs().mean(dim=0, keepdim=True) / clip_max # (1, H, W), [0.0, 1.0]
    # pattern = pattern + pattern.min()
    # pattern = pattern / pattern.max() # (C, H, W), [0.0, 1.0]


    # method 2
    mask = pattern.abs().mean(dim=0, keepdim=True)
    mask = mask / mask.max() # (1, H, W), [0.0, 1.0]
    print(f'L1 mask: {mask.sum()}')

    pattern = pattern - pattern.min()
    pattern = pattern / pattern.max() # (C, H, W), [0.0, 1.0]


    cv2.imwrite(osp.join(now_save_trigger_path, 'mask.png'), (mask[0]*255).clip(0.0, 255.0).round().to(dtype=torch.uint8).cpu().numpy())
    np.savez(osp.join(now_save_trigger_path, "trigger.npz"), re_mark=pattern.cpu().numpy(), re_mask=mask.cpu().numpy())
    pattern = normalize(pattern) # pattern after normalize

    blended = core.Blended(
        train_dataset=trainset,
        test_dataset=testset,
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        y_target=y_target,
        poisoned_rate=0.05,
        pattern=pattern.cpu(),
        weight=mask.cpu(),
        poisoned_transform_train_index=len(trainset.transform.transforms),
        poisoned_transform_test_index=len(testset.transform.transforms),
        poisoned_target_transform_index=0,
        schedule=None,
        seed=int(time.time()),
        deterministic=deterministic
    )

    _, poisoned_testset = blended.get_poisoned_dataset()

    schedule = {
        'device': 'GPU',

        'batch_size': batch_size,
        'num_workers': num_workers,

        'metric': 'ASR',

        'save_dir': now_save_trigger_path,
        'experiment_name': f'PixelBackdoor_trigger_test_ASR'
    }
    top1_correct, top5_correct, total_num, mean_loss = core.utils.test(model, poisoned_testset, schedule)

    mean_losses.append(mean_loss)

    for i in range(len(distances)):
        std_weight_ = std_weight.clone().detach().cuda().unsqueeze(0)
        weight_ = mask.clone().detach().cuda()
        distances[i].append(distance_functions[i][1](weight_, std_weight_).cpu().item())


import matplotlib.pyplot as plt
distances = np.array(distances)
mean_losses = np.array(mean_losses)
os.makedirs(osp.join(save_trigger_path, f'__summary_experiments_results'), exist_ok=True)
np.savez(osp.join(save_trigger_path, f'__summary_experiments_results', f"{dataset_name}_{iteration}.npz"), distances=distances, mean_losses=mean_losses)

for i in range(len(distances)):
    plt.figure(figsize=(16,9), dpi=600)
    plt.scatter(distances[i], mean_losses, s=0.25)
    plt.savefig(osp.join(save_trigger_path, f'__summary_experiments_results', f"{dataset_name}_{iteration}_{distance_functions[i][0]}.png"))
    plt.close()

total_losses = mean_losses + init_cost * distances[0]

with open(osp.join(save_trigger_path, f'__summary_experiments_results', 'trigger_statistics.log'),'w') as f:
    f.write(f'lr:{PixelBackdoor_lr:.10f}\n')
    f.write(f'epochs:{PixelBackdoor_epochs}\n')
    f.write(f'init_cost:{init_cost:.10f}\n')

    f.write(f'PixelBackdoor样本数:{len(mean_losses) - 1}\n')

    f.write(f'total Loss (min):{total_losses[1:].min()}\n')
    f.write(f'total Loss (mean):{total_losses[1:].mean()}\n')
    f.write(f'total Loss (max):{total_losses[1:].max()}\n')
    f.write(f'total Loss (std):{total_losses[1:].std()}\n')

    # f.write(f'Poisoned Loss:{mean_losses[0]}\n')
    f.write(f'Backdoor Loss (min):{mean_losses[1:].min()}\n')
    f.write(f'Backdoor Loss (mean):{mean_losses[1:].mean()}\n')
    f.write(f'Backdoor Loss (max):{mean_losses[1:].max()}\n')
    f.write(f'Backdoor Loss (std):{mean_losses[1:].std()}\n')

    norm_losses = distances[0]
    f.write(f'L1 norm loss (min):{norm_losses[1:].min()}\n')
    f.write(f'L1 norm loss (mean):{norm_losses[1:].mean()}\n')
    f.write(f'L1 norm loss (max):{norm_losses[1:].max()}\n')
    f.write(f'L1 norm loss (std):{norm_losses[1:].std()}\n')

    f.write(f'{PixelBackdoor_lr:.10f}\n')
    f.write(f'{PixelBackdoor_epochs}\n')
    f.write(f'{init_cost:.10f}\n')
    f.write(f'{len(mean_losses) - 1}\n')
    f.write(f'{total_losses[1:].min()}\n')
    f.write(f'{total_losses[1:].mean()}\n')
    f.write(f'{total_losses[1:].max()}\n')
    f.write(f'{total_losses[1:].std()}\n')
    f.write(f'{mean_losses[1:].min()}\n')
    f.write(f'{mean_losses[1:].mean()}\n')
    f.write(f'{mean_losses[1:].max()}\n')
    f.write(f'{mean_losses[1:].std()}\n')
    f.write(f'{norm_losses[1:].min()}\n')
    f.write(f'{norm_losses[1:].mean()}\n')
    f.write(f'{norm_losses[1:].max()}\n')
    f.write(f'{norm_losses[1:].std()}\n')


    # find top K
    norm_distances = distances[1]

    K = max(20, int(len(mean_losses) * 0.1))
    K = min(K, len(mean_losses))

    mean_losses = torch.from_numpy(mean_losses)
    topk = torch.topk(mean_losses, K, dim=0, largest=True, sorted=True)
    f.write(f'\n==========Largest {K} backdoor loss reversed triggers==========\n')
    for i in range(len(topk.indices)):
        f.write(f'trigger id:{topk.indices[i]}, mean_loss:{mean_losses[topk.indices[i]]}, distance:{norm_distances[topk.indices[i]]}\n')
    topk = torch.topk(mean_losses, K, dim=0, largest=False, sorted=True)
    f.write(f'\n==========Smallest {K} backdoor loss reversed triggers==========\n')
    for i in range(len(topk.indices)):
        f.write(f'trigger id:{topk.indices[i]}, mean_loss:{mean_losses[topk.indices[i]]}, distance:{norm_distances[topk.indices[i]]}\n')

    norm_distances = torch.from_numpy(norm_distances)
    topk = torch.topk(norm_distances, K, dim=0, largest=True, sorted=True)
    f.write(f'\n==========Largest {K} L1 norm distance reversed triggers==========\n')
    for i in range(len(topk.indices)):
        f.write(f'trigger id:{topk.indices[i]}, mean_loss:{mean_losses[topk.indices[i]]}, distance:{norm_distances[topk.indices[i]]}\n')
    topk = torch.topk(norm_distances, K, dim=0, largest=False, sorted=True)
    f.write(f'\n==========Smallest {K} L1 norm distance reversed triggers==========\n')
    for i in range(len(topk.indices)):
        f.write(f'trigger id:{topk.indices[i]}, mean_loss:{mean_losses[topk.indices[i]]}, distance:{norm_distances[topk.indices[i]]}\n')
