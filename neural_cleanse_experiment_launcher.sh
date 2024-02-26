# test
CUDA_VISIBLE_DEVICES=0 python neural_cleanse_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --y_target 0 --trigger_size 40 --init_cost_rate 0.08 \
--NC_epochs 12 --NC_lr 1024.0 --NC_optim SGD --NC_schedule 10 --NC_gamma 0.1 \
2>&1 | tee 0.log

# 当前调参实验
# init_cost_rate 0.08
CUDA_VISIBLE_DEVICES=0 python neural_cleanse_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --y_target 0 --trigger_size 40 --init_cost_rate 0.08 \
--NC_epochs 1200 --NC_lr 1024.0 --NC_optim SGD --NC_schedule 1000 --NC_gamma 0.1 \
2>&1 | tee 0.log

CUDA_VISIBLE_DEVICES=1 python neural_cleanse_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --y_target 0 --trigger_size 40 --init_cost_rate 0.08 \
--NC_epochs 1200 --NC_lr 2048.0 --NC_optim SGD --NC_schedule 1000 --NC_gamma 0.1 \
2>&1 | tee 1.log

CUDA_VISIBLE_DEVICES=2 python neural_cleanse_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --y_target 0 --trigger_size 40 --init_cost_rate 0.08 \
--NC_epochs 1000 --NC_lr 4096.0 --NC_optim SGD --NC_schedule 1000 --NC_gamma 0.1 \
2>&1 | tee 2.log

CUDA_VISIBLE_DEVICES=3 python neural_cleanse_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --y_target 0 --trigger_size 40 --init_cost_rate 0.08 \
--NC_epochs 1000 --NC_lr 8192.0 --NC_optim SGD --NC_schedule 1000 --NC_gamma 0.1 \
2>&1 | tee 3.log

# init_cost_rate 163.84
CUDA_VISIBLE_DEVICES=0 python neural_cleanse_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --y_target 0 --trigger_size 40 --init_cost_rate 163.84 \
--NC_epochs 1200 --NC_lr 128.0 --NC_optim SGD --NC_schedule 1000 --NC_gamma 0.1 \
2>&1 | tee 0.log

CUDA_VISIBLE_DEVICES=1 python neural_cleanse_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --y_target 0 --trigger_size 40 --init_cost_rate 163.84 \
--NC_epochs 1200 --NC_lr 256.0 --NC_optim SGD --NC_schedule 1000 --NC_gamma 0.1 \
2>&1 | tee 1.log

CUDA_VISIBLE_DEVICES=2 python neural_cleanse_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --y_target 0 --trigger_size 40 --init_cost_rate 163.84 \
--NC_epochs 1200 --NC_lr 512.0 --NC_optim SGD --NC_schedule 1000 --NC_gamma 0.1 \
2>&1 | tee 2.log

CUDA_VISIBLE_DEVICES=3 python neural_cleanse_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --y_target 0 --trigger_size 40 --init_cost_rate 163.84 \
--NC_epochs 1200 --NC_lr 1024.0 --NC_optim SGD --NC_schedule 1000 --NC_gamma 0.1 \
2>&1 | tee 3.log



# 标准benchmark实验
CUDA_VISIBLE_DEVICES=0 python neural_cleanse_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --y_target 0 --trigger_size 40 --init_cost_rate 0.08 \
--NC_epochs 1200 --NC_lr 1024.0 --NC_optim SGD --NC_schedule 1000 --NC_gamma 0.1 \
2>&1 | tee 0.log

CUDA_VISIBLE_DEVICES=1 python neural_cleanse_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --y_target 0 --trigger_size 40 --init_cost_rate 0.08 \
--NC_epochs 1200 --NC_lr 2048.0 --NC_optim SGD --NC_schedule 1000 --NC_gamma 0.1 \
2>&1 | tee 1.log

CUDA_VISIBLE_DEVICES=2 python neural_cleanse_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --y_target 0 --trigger_size 40 --init_cost_rate 0.08 \
--NC_epochs 1000 --NC_lr 4096.0 --NC_optim SGD --NC_schedule 1000 --NC_gamma 0.1 \
2>&1 | tee 2.log

CUDA_VISIBLE_DEVICES=3 python neural_cleanse_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --y_target 0 --trigger_size 40 --init_cost_rate 0.08 \
--NC_epochs 1000 --NC_lr 8192.0 --NC_optim SGD --NC_schedule 1000 --NC_gamma 0.1 \
2>&1 | tee 3.log





CUDA_VISIBLE_DEVICES=0 python neural_cleanse_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --y_target 0 --trigger_size 40 --init_cost_rate 0.08 \
--NC_epochs 720 --NC_lr 2048.0 --NC_optim SGD --NC_schedule 720 --NC_gamma 0.1

CUDA_VISIBLE_DEVICES=1 python neural_cleanse_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --y_target 0 --trigger_size 40 --init_cost_rate 0.08 \
--NC_epochs 30 --NC_lr 8192.0 --NC_optim SGD --NC_schedule 30 --NC_gamma 0.1

CUDA_VISIBLE_DEVICES=2 python neural_cleanse_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --y_target 0 --trigger_size 40 --init_cost_rate 0.08 \
--NC_epochs 30 --NC_lr 16384.0 --NC_optim SGD --NC_schedule 30 --NC_gamma 0.1

CUDA_VISIBLE_DEVICES=3 python neural_cleanse_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --y_target 0 --trigger_size 40 --init_cost_rate 0.08 \
--NC_epochs 30 --NC_lr 32768.0 --NC_optim SGD --NC_schedule 30 --NC_gamma 0.1



# 对抗benchmark实验
CUDA_VISIBLE_DEVICES=2 python neural_cleanse_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/adv_experiments/ResNet-50_ImageNet100_20x20_BaseMix_init_cost_rate0.01_adv_lambda2.2500000000000002e-07_global_seed666_deterministicFalse_2022-11-09_17:52:06/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --y_target 0 --trigger_size 20 --init_cost_rate 10.24


CUDA_VISIBLE_DEVICES=2 python neural_cleanse_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/adv_experiments/ResNet-50_ImageNet100_40x40_BaseMix_init_cost_rate0.01_adv_lambda5.6250000000000004e-08_global_seed666_deterministicFalse_2022-11-09_17:52:12/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --y_target 0 --trigger_size 40 --init_cost_rate 163.84

CUDA_VISIBLE_DEVICES=3 python neural_cleanse_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/adv_experiments/ResNet-50_ImageNet100_40x40_BaseMix_init_cost_rate0.01_adv_lambda5.6250000000000004e-08_global_seed666_deterministicFalse_2022-11-09_17:52:12/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --y_target 0 --trigger_size 40 --init_cost_rate 327.68

CUDA_VISIBLE_DEVICES=7 python neural_cleanse_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/adv_experiments/ResNet-50_ImageNet100_40x40_BaseMix_init_cost_rate0.01_adv_lambda5.6250000000000004e-08_global_seed666_deterministicFalse_2022-11-09_17:52:12/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --y_target 0 --trigger_size 40 --init_cost_rate 655.36

CUDA_VISIBLE_DEVICES=3 python neural_cleanse_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/adv_experiments/ResNet-50_ImageNet100_40x40_BaseMix_init_cost_rate0.01_adv_lambda5.6250000000000004e-08_global_seed666_deterministicFalse_2022-11-09_17:52:12/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --y_target 0 --trigger_size 40 --init_cost_rate 0.01




CUDA_VISIBLE_DEVICES=7 python neural_cleanse_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/adv_experiments/ResNet-50_ImageNet100_60x60_BaseMix_init_cost_rate0.01_adv_lambda2.5000000000000002e-08_global_seed666_deterministicFalse_2022-11-09_17:52:18/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --y_target 0 --trigger_size 60 --init_cost_rate 0.01
