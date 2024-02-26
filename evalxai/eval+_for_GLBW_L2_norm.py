import argparse
import os.path as osp
import random

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import skimage
import torch
import torchvision
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, ToPILImage, Resize

import core


parser = argparse.ArgumentParser(description='XAI for Trojaned Models')
parser.add_argument('--model_type', default='core', type=str)
parser.add_argument('--model_name', default='ResNet-18', type=str)
parser.add_argument('--model_path', default='./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-26_10:43:37/ckpt_epoch_200.pth', type=str)

parser.add_argument('--dataset_name', default='CIFAR-10', type=str)
parser.add_argument('--dataset_root_path', default='../datasets', type=str)

parser.add_argument('--trigger_path', default='./tests/square_trigger_3x3.png', type=str)
parser.add_argument('--y_target', default=0, type=int)
args = parser.parse_args()

model_type = args.model_type
model_name = args.model_name
model_path = args.model_path
dataset_name = args.dataset_name
dataset_root_path = args.dataset_root_path
trigger_path = args.trigger_path
y_target = args.y_target


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
elif dataset_name=='CIFAR-10':
    num_classes=10
    data_shape=(3, 32, 32)
    dataset=torchvision.datasets.CIFAR10
    transform_train = Compose([
        RandomHorizontalFlip(),
        ToTensor()
    ])
    poisoned_transform_train_index=len(transform_train.transforms)
    transform_test = Compose([
        ToTensor()
    ])
    poisoned_transform_test_index=len(transform_test.transforms)
    trainset = dataset(dataset_root_path, train=True, transform=transform_train, download=True)
    testset = dataset(dataset_root_path, train=False, transform=transform_test, download=True)
    # x = torch.stack([transform_train(Image.fromarray(img)) for img in trainset.data]) # torch.Tensor torch.float32 [0.0, 1.0] (50000, 3, 32, 32)
    # y = torch.tensor(trainset.targets, dtype=torch.int64) # torch.Tensor torch.int64 (50000)
    x = torch.stack([transform_test(Image.fromarray(img)) for img in testset.data]) # torch.Tensor torch.float32 [0.0, 1.0] (10000, 3, 32, 32)
    y = torch.tensor(testset.targets, dtype=torch.int64) # torch.Tensor torch.int64 (10000)
    permutation = np.random.permutation(x.shape[0]) # random shuffle samples
    x = x[permutation]
    y = y[permutation]
    normalize = torch.nn.Identity()
    inv_normalize = torch.nn.Identity()
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

    random.shuffle(testset.samples) # random shuffle samples
    x = torch.stack([transform_test(cv2.imread(item[0])) for item in testset.samples[:2000]]) # torch.Tensor torch.float32 [0.0, 1.0] (N, 3, 32, 32)
    y = torch.tensor([item[1] for item in testset.samples[:2000]], dtype=torch.int64) # torch.Tensor torch.int64 (N)
    normalize = torch.nn.Identity()
    inv_normalize = torch.nn.Identity()
elif dataset_name=='ImageNet100':
    num_classes=100
    data_shape=(3, 224, 224)
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
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    poisoned_transform_train_index=len(transform_train.transforms) - 1
    transform_test = Compose([
        ToTensor(),
        Resize((256, 256)),
        RandomCrop((224, 224)),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    poisoned_transform_test_index=len(transform_test.transforms) - 1
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
    transform_train_without_normalize = Compose([
        ToTensor(),
        RandomResizedCrop(
            size=(224, 224),
            scale=(0.1, 1.0),
            ratio=(0.8, 1.25),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC
        ),
        RandomHorizontalFlip()
    ])
    print(f'The samples number of the testset of ImageNet100: {len(testset.samples)}', flush=True)
    x = []
    for i, sample in enumerate(testset.samples):
        if i % 100 == 0:
            print(f'Reading the {i}th image.', flush=True)
        if sample[1] != y_target: # remove the y_target samples
            x.append(transform_train_without_normalize(my_read_image(sample[0])))
    x = torch.stack(x, dim=0) # torch.Tensor torch.float32 [0.0, 1.0] (4950, 3, 224, 224)
    y = torch.tensor([label for image_path, label in testset.samples], dtype=torch.int64) # torch.Tensor torch.int64 (5000)
    y = y[y != y_target] # torch.Tensor torch.int64 (4950)
    print(f'The number of the samples after removing y_target samples: {x.shape[0]}', flush=True)
    clip_max = 1.0
    normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=False)
else:
    raise NotImplementedError(f"Unsupported dataset {dataset_name}")

if trigger_path.endswith('.png'):
    trigger_img = cv2.imread(trigger_path)
    trigger_pixel_num = trigger_img.sum() / 255
    if trigger_img.ndim == 3:
        trigger_pixel_num = round(trigger_pixel_num / trigger_img.shape[2])
        trigger_img = trigger_img.transpose([2, 0, 1]) # (H, W, C) to (C, H, W)
    else:
        trigger_pixel_num = round(trigger_pixel_num)

    pattern = torch.from_numpy(trigger_img).to(dtype=torch.float32) / 255.0
    weight = torch.zeros(trigger_img.shape, dtype=torch.float32)
    weight[pattern > 0] = 1.0
    trigger_name = "std_trigger"
elif trigger_path.endswith('.npz'):
    trigger = np.load(trigger_path)
    weight = trigger['re_mask'] # (H, W)
    pattern = trigger['re_mark'] # (C, H, W)
    trigger_pixel_num = round(weight.sum())
    weight, pattern = torch.from_numpy(weight), torch.from_numpy(pattern)
    if weight.ndim == 2:
        weight.unsqueeze_(0) # (H, W) -> (1, H, W)
    if weight.size(0) == 1:
        weight = weight.repeat(3, 1, 1) # (1, H, W) -> (3, H, W)
    trigger_name = f"reversed_trigger_{trigger_path.split('/')[-3]}"
else:
    raise NotImplementedError("")
pattern = normalize(pattern)


if model_type=='core':
    if model_name=='BaselineMNISTNetwork':
        model=core.models.BaselineMNISTNetwork()
    elif model_name.startswith('ResNet'):
        model=core.models.ResNet(int(model_name.split('-')[-1]), num_classes)
    elif model_name.startswith('VGG'):
        str_to_vgg = {
            "VGG-11": core.models.vgg11_bn,
            "VGG-13": core.models.vgg13_bn,
            "VGG-16": core.models.vgg16_bn,
            "VGG-19": core.models.vgg19_bn
        }
        model=str_to_vgg[model_name](num_classes)
    else:
        raise NotImplementedError(f"Unsupported model_type: {model_type} and model_name: {model_name}")
elif model_type=='torchvision':
    model = torchvision.models.__dict__[model_name.lower().replace('-', '')](num_classes=num_classes)
else:
    raise NotImplementedError(f"Unsupported model_type: {model_type}")
model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
model = model.cuda()
model = model.eval()
device = next(model.parameters()).device


def rerange(x):
    if x.max() == x.min(): 
        return x-x.min()
    else:
        return (x-x.min())/(x.max()-x.min())


def pred(img, model):
    pred = model(img.unsqueeze(0))
    return torch.argmax(pred, dim=1).item()


# test_data_trig = test_module.config.test_data_trig
# test_data_orig = test_module.config.test_data_orig
test_data_orig = x
L = []
img_orig = None
img_trig = None
label_orig = None
label_trig = None
trigger = None
for i in range(len(test_data_orig)):
    img_orig = test_data_orig[i].to(device)
    label_orig = y[i].to(device)
    img_trig = (1 - weight.to(device)) * img_orig + weight.to(device) * pattern.to(device)
    label_trig = y_target
    trigger = weight

    pred_orig = pred(img_orig, model)
    pred_trig = pred(img_trig, model)

    # print(label_orig)
    # print(pred_orig)

    #print(label_orig, label_trig, pred_orig, pred_trig)
    #if pred_orig != label_orig and pred_trig == label_trig:
    if label_orig != label_trig and pred_orig == label_orig and pred_trig == label_trig:
        L.append([img_orig, img_trig, label_orig, label_trig, trigger, i])
        if len(L) == 100:
            break
        #if len(L)>2:
        #break


# In[10]:
import torch.nn.functional as F
from lime import lime_image
from skimage.segmentation import mark_boundaries

def Lime(model, image, label):

    def batch_predict(images):
        model.eval()
        batch = torch.stack(tuple(torch.from_numpy(i).permute(2, 0, 1) for i in images), dim=0)

        model.to(device)
        batch = batch.to(device)

        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()


    def segmentation_fn(image):
        if image.shape[0] * image.shape[1] < 1024:
            ratio = 1.0
            kernel_size = 1
            max_dist = 5
        else:
            tmp = (image.shape[0] * image.shape[1]) ** 0.5
            ratio = (32 / tmp) ** 0.8
            kernel_size = round((tmp / 32) ** 0.7)
            max_dist = 5 * ((tmp / 32) ** 1.9)

        segments = skimage.segmentation.quickshift(image, ratio=ratio, kernel_size=kernel_size, max_dist=max_dist)
        return segments


    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        np.array(image.permute(1, 2, 0).cpu()),
        batch_predict, # classification function
        labels=[label],
        top_labels=None,
        hide_color=0,
        num_samples=1000, # number of images that will be sent to classification function
        segmentation_fn=segmentation_fn
    )

    _, mask = explanation.get_image_and_mask(label, positive_only=True, num_features=1)
    for i in range(2, 10):
        _, now_mask = explanation.get_image_and_mask(label, positive_only=True, num_features=i)
        mask += now_mask

    return mask


# In[11]:
from captum.attr import Saliency, GuidedBackprop, GuidedGradCam, LayerGradCam, LayerAttribution
from captum.attr import Occlusion, ShapleyValueSampling, FeatureAblation, ShapleyValues
import time

def get_saliency_maps(img, label, model):
    maps = []
    cost_time = []

    w = img.shape[1]
    h = img.shape[2]

    input = img.clone()
    input.unsqueeze_(0)

    m = []
    mask = []
    if dataset_name == "CIFAR-10" or dataset_name == "GTSRB":
        m.append(torch.nn.Upsample(scale_factor=1, mode='nearest'))
        mask.append(torch.arange(0, 1024, dtype=torch.float32).view(1, 1, 32, 32).to(device=device))
        m.append(torch.nn.Upsample(scale_factor=2, mode='nearest'))
        mask.append(torch.arange(0, 256, dtype=torch.float32).view(1, 1, 16, 16).to(device=device))
        m.append(torch.nn.Upsample(scale_factor=4, mode='nearest'))
        mask.append(torch.arange(0, 64, dtype=torch.float32).view(1, 1, 8, 8).to(device=device))
        m.append(torch.nn.Upsample(scale_factor=8, mode='nearest'))
        mask.append(torch.arange(0, 16, dtype=torch.float32).view(1, 1, 4, 4).to(device=device))
        m.append(torch.nn.Upsample(scale_factor=16, mode='nearest'))
        mask.append(torch.arange(0, 4, dtype=torch.float32).view(1, 1, 2, 2).to(device=device))
    elif dataset_name == "ImageNet100":
        m = []
        mask = []
        m.append(torch.nn.Upsample(scale_factor=8, mode='nearest'))
        mask.append(torch.arange(0, 784, dtype=torch.float32).view(1, 1, 28, 28))
        m.append(torch.nn.Upsample(scale_factor=16, mode='nearest'))
        mask.append(torch.arange(0, 196, dtype=torch.float32).view(1, 1, 14, 14))
        m.append(torch.nn.Upsample(scale_factor=32, mode='nearest'))
        mask.append(torch.arange(0, 49, dtype=torch.float32).view(1, 1, 7, 7))
        m.append(torch.nn.Upsample(scale_factor=56, mode='nearest'))
        mask.append(torch.arange(0,  16, dtype=torch.float32).view(1, 1, 4, 4))
        m.append(torch.nn.Upsample(scale_factor=112, mode='nearest'))
        mask.append(torch.arange(0, 4, dtype=torch.float32).view(1, 1, 2, 2))
    else:
        raise NotImplementedError("")

    # Saliency
    saliency = Saliency(model)
    start = time.time()
    sa = saliency.attribute(input, target=label)[0]
    end = time.time()
    maps.append(sa)
    cost_time.append(end-start)
    print('Saliency Done. Time:', start, end, end-start)

    # GuidedBackprop
    guidedbacprop = GuidedBackprop(model)
    start = time.time()
    sa = guidedbacprop.attribute(input, target=label)[0]
    end = time.time()
    maps.append(sa.abs())
    cost_time.append(end-start)
    print('GuidedBackproagation Done. Time:', start, end, end-start)

    # LayerGradCam
    start = time.time()
    if model_name == 'VGG-16': 
        layer_gc = LayerGradCam(model, model.features[28]) #VGG16
    elif model_name == 'AlexNet':
        layer_gc = LayerGradCam(model, model.features[10]) #alexnet
    elif model_name == 'ResNet-50':
        layer_gc = LayerGradCam(model, model.layer4[2].bn3)
    elif model_name == 'ResNet-34':
        layer_gc = LayerGradCam(model, model.layer4[2].bn2)
    elif model_name == 'ResNet-18':
        layer_gc = LayerGradCam(model, model.layer4[1].bn2)
    else:
        raise NotImplementedError("")
    attr = layer_gc.attribute(input, label)
    # attr = layer_gc.attribute(input, label, relu_attributions=True)
    sa = LayerAttribution.interpolate(attr, (w, h), interpolate_mode='bilinear')[0]
    end = time.time()
    maps.append(sa.abs())
    cost_time.append(end-start)
    print('GradCAM Done. Time:', start, end, end-start)

    # GuidedGradCam
    start = time.time()
    if model_name == 'VGG-16':
        guided_gc = GuidedGradCam(model, model.features[28]) #VGG16
    elif model_name == 'AlexNet':
        guided_gc = GuidedGradCam(model, model.features[10]) #alexnet
    elif model_name == 'ResNet-50':
        guided_gc = GuidedGradCam(model, model.layer4[2].bn3)
    elif model_name == 'ResNet-34':
        guided_gc = GuidedGradCam(model, model.layer4[2].bn2)
    elif model_name == 'ResNet-18':
        guided_gc = GuidedGradCam(model, model.layer4[1].bn2)
    else:
        raise NotImplementedError("")
    sa = guided_gc.attribute(input, label, interpolate_mode='bilinear')[0]
    end = time.time()
    maps.append(sa.abs())
    cost_time.append(end-start)
    print('Guided GradCAM Done. Time:', start, end, end-start)

    #return [sa, gbp, gcam, ggcam]
    trigger_size = trigger_pixel_num ** 0.5
    # Occlusion
    occlusion = Occlusion(model)
    start = time.time()
    sa = occlusion.attribute(
        input,
        strides=(3, max(int(trigger_size/3), 1), max(int(trigger_size/3), 1)),
        target=label,
        sliding_window_shapes=(3, max(int(trigger_size/2), 1), max(int(trigger_size/2), 1)),
        baselines=0
    )[0]
    end = time.time()
    maps.append(sa.abs())
    cost_time.append(end-start)
    print('Occlusion Done. Time:', start, end, end-start)

    # FeatureAblation 
    ablator = FeatureAblation(model)
    start = time.time()
    sa_map = []
    for i in range(len(mask)):
        sa = ablator.attribute(input, target=label, feature_mask=m[i](mask[i]).type(torch.int))[0]
        sa_map.append(np.array(sa.cpu()))
    end = time.time()
    sa = torch.from_numpy(np.mean(np.array(sa_map), axis = 0))
    maps.append(sa.abs())
    cost_time.append(end-start)
    print('FeatureAblation Done. Time:', start, end, end-start)

    '''
    # ShapleyValueSampling (Timeout for large image: ~0.5hr)
    shapleyvaluesampling = ShapleyValueSampling(test_module.get_model())
    start = time.time()
    sa_map = []
    for i in range(len(mask)):
        sa = shapleyvaluesampling.attribute(input, target=label, feature_mask=m[i](mask[i]).type(torch.int))[0]
        sa_map.append(np.array(sa))
    end = time.time()
    sa = torch.from_numpy(np.mean(np.array(sa_map), axis = 0))
    maps.append(sa)
    cost_time.append(end-start)
    print('ShapleyValueSampling Done. Time:', start, end, end-start)
    '''

    # Lime
    start = time.time()
    lime = Lime(model, img, label)
    lime = torch.from_numpy(lime)
    lime.unsqueeze_(0)
    sa = lime
    end = time.time()
    maps.append(sa)
    cost_time.append(end-start)
    print('Lime Done. Time:', start, end, end-start)

    return maps, cost_time



# In[12]:
def canny(img):
    disp_img = np.array(255*img).astype(np.uint8)
    disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2GRAY)
    disp_img = cv2.Canny(disp_img,10,40)
    return disp_img

def get_salienct_points(g):
    points = []
    w = g.shape[0]
    h = g.shape[1]
    for i in range(w):
        for j in range(h):
            if g[i][j] > 0:
                points.append([[i, j]])
    return np.array(points, np.float32)

# In[17]:
def imgDiff(img1, img2):
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()

    x_min = img1.min()
    x_max = img2.max()

    if x_max == x_min:
        img1 = img1-x_min
        img2 = img2-x_min
    else:
        img1 = (img1-x_min)/(x_max-x_min)
        img2 = (img2-x_min)/(x_max-x_min)

    img1 = np.transpose(img1, (1,2,0))
    img2 = np.transpose(img2, (1,2,0))

    img1 = (255*np.array(img1)).astype(np.uint8)
    img2 = (255*np.array(img2)).astype(np.uint8)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    return np.sum(img1 != img2)/(224*224)


def get_disp_img(img): # img, (C, H, W), [0.0, 1.0]
    return (img * 255).clip(0.0, 255.0).round().to(dtype=torch.uint8).permute(1, 2, 0).cpu().numpy()


def myIOU(evalxai_mask, trigger_mask):
    """Calculate mask IoU.

    Args:
        evalxai_mask (ndarray): shape (C, H, W) or (1, H, W), 0.0 or 1.0.
        trigger_mask (ndarray): shape (C, H, W) or (1, H, W), 0.0 or 1.0.

    Returns:
        float: IoU.
    """
    return ((evalxai_mask[0] - trigger_mask[0]) ** 2.0).sum() ** 0.5


import os
folder = f'./evalxai+_for_GLBW_experiments_L2_norm/{model_path.split("/")[-2]}_{trigger_name}'
os.makedirs(folder, exist_ok=True)

CT_L = []
IOU_L = []
TIME_L = []
DIFF_L = []
for i in range(len(L)):
    img_orig = L[i][0]
    img_trig = L[i][1]
    label_orig = L[i][2]
    label_trig = L[i][3]
    trigger = L[i][4]
    img_id = L[i][5]

    fig, axs = plt.subplots(2, 8, constrained_layout=False, figsize=(16, 4))

    # Display trojaned image
    disp_img = get_disp_img(inv_normalize(img_trig))
    axs[0][0].imshow(disp_img)

    axs[0][0].set_xlabel('pred:'+str(label_trig))
    axs[0][0].xaxis.set_ticks([])
    axs[0][0].yaxis.set_ticks([])

    # Get the bounding box of the trigger
    disp_img = get_disp_img(inv_normalize(trigger))

    trigger = rerange(np.array(disp_img))
    trigger = canny(trigger)

    pts = get_salienct_points(trigger)

    if len(pts) == 0:
        bb_trigger = [-1,-1,-1,-1]
    else:
        x2, y2, w2, h2 = cv2.boundingRect(pts)
        bb_trigger = [x2,y2,x2+w2,y2+h2]

        # Create a Rectangle patch
        rect = patches.Rectangle((y2,x2),h2,w2,linewidth=1,edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        axs[0][0].add_patch(rect)

    # Display the original image
    disp_img = get_disp_img(inv_normalize(img_orig))
    axs[1][0].imshow(disp_img)

    axs[1][0].set_xlabel('pred:'+str(label_orig))
    axs[1][0].xaxis.set_ticks([])
    axs[1][0].yaxis.set_ticks([])

    # Get the saliency maps
    l, t = get_saliency_maps(img_trig, label_trig, model)

    iou = []
    ct = []
    diff = []
    for j in range(len(l)):
        # calculate best IoU with topk
        if l[j].size(0) == 1:
            l[j] = l[j].repeat(3, 1, 1) # (3, H, W)

        g = rerange(l[j].detach()) # rerange to [0, 1], (3, H, W)
        # Display the saliency map
        axs[0][j+1].imshow(get_disp_img(g), cmap='RdBu_r')

        # Display the detected trigger bounding box
        l_1D = l[j].detach().view(-1)
        topk = torch.topk(l_1D, trigger_pixel_num * 3, dim=0, largest=True, sorted=True)
        best_thr = topk.values[-1]

        evalxai_mask_1 = (l[j].detach()>=best_thr).cpu().numpy().astype(np.float32) # (3, H, W)
        # if j == 5:
        #     print(evalxai_mask_1[0, -15:, -15:])

        evalxai_mask_2 = (l[j].detach()>best_thr).cpu().numpy().astype(np.float32) # (3, H, W)
        # if j == 5:
        #     print(evalxai_mask_2[0, -15:, -15:])

        if myIOU(evalxai_mask_1, weight.cpu().numpy()) < myIOU(evalxai_mask_2, weight.cpu().numpy()):
            evalxai_mask = evalxai_mask_1
        else:
            evalxai_mask = evalxai_mask_2

        im = canny(np.transpose(evalxai_mask, (1,2,0)))
        pts = get_salienct_points(im)
        if len(pts) == 0:
            bb = [-1,-1,-1,-1]
        else:
            x2, y2, w2, h2 = cv2.boundingRect(pts)
            bb = [x2,y2,x2+w2,y2+h2]

            # Create a Rectangle patch
            rect = patches.Rectangle((y2,x2),h2,w2,linewidth=1,edgecolor='r',facecolor='none')

            # Add the patch to the Axes
            axs[0][j+1].add_patch(rect)

        # if j == 6:
        #     print(evalxai_mask[0, -15:, -15:])
        #     breakpoint()
        iou_v = myIOU(evalxai_mask, weight.cpu().numpy())
        iou.append(iou_v)

        axs[0][j+1].set_xlabel('iou:'+"{:.4f}".format(iou_v))
        axs[0][j+1].xaxis.set_ticks([])
        axs[0][j+1].yaxis.set_ticks([])

        # Recover Image
        mask = torch.zeros_like(img_orig)
        mask[:, x2:x2+w2, y2:y2+h2] = 1.0
        img_recover = img_orig * mask + img_trig * (1-mask)

        pred_orig = model(img_orig.unsqueeze(0))
        pred_recover = model(img_recover.unsqueeze(0))

        if torch.argmax(pred_orig) == torch.argmax(pred_recover):
            ct.append(1)
        else:
            ct.append(0)
        l0 = imgDiff(img_orig,img_recover)
        diff.append(l0)

        # Display the recovered image
        disp_img = get_disp_img(inv_normalize(img_recover))

        axs[1][j+1].imshow(disp_img, cmap='RdBu_r')

        axs[1][j+1].set_xlabel('pred:'+str(torch.argmax(pred_recover).item())+' '+'diff:'+"%.4f" % l0)

        axs[1][j+1].xaxis.set_ticks([])
        axs[1][j+1].yaxis.set_ticks([])

    fig.tight_layout()

    #plt.show()
    plt.savefig(folder+'/'+str(img_id)+'.png')

    CT_L.append(np.array(ct))
    IOU_L.append(np.array(iou))
    TIME_L.append(np.array(t))
    DIFF_L.append(np.array(diff))


# In[22]:


def save(L, folder, var):
    tmp = np.array(L)
    var_avg = np.mean(tmp, axis=0)
    #var_std = np.std(tmp, axis=0)

    with open(osp.join(folder, "_"+var+"_mean.txt"), "a") as myfile:
        myfile.write('\n'.join([str(x) for x in var_avg]))

    #with open(var+"_std.txt", "a") as myfile:
        #myfile.write(str(trigger_size)+','+model_name+','+exp_name+','+','.join([str(x) for x in var_std])+'\n')

save(CT_L, folder, 'rr')
save(IOU_L, folder, 'iou')
save(TIME_L, folder, 'cc')
save(DIFF_L, folder, 'rd')

