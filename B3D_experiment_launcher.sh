# test
CUDA_VISIBLE_DEVICES=0 python B3D_experiment_launcher.py \
--model_type core --model_name ResNet-18 --model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-18_CIFAR-10_BadNets_2022-11-05_16:13:21/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --batch_size 128 --num_workers 4 \
--y_target 1 --trigger_size 3 --init_cost_rate 1.0 \
--B3D_epochs 1 --B3D_lr 0.05 \
2>&1 | tee 0.log

# test samples_per_draw
CUDA_VISIBLE_DEVICES=0 python B3D_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 --model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --batch_size 128 --num_workers 4 \
--y_target 0 --trigger_size 40 --init_cost_rate 1.0 \
--B3D_epochs 200 --B3D_lr 0.1 --attack_succ_threshold 0.96 --samples_per_draw 800 \
2>&1 | tee 0.log

CUDA_VISIBLE_DEVICES=1 python B3D_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 --model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --batch_size 128 --num_workers 4 \
--y_target 0 --trigger_size 40 --init_cost_rate 1.0 \
--B3D_epochs 200 --B3D_lr 0.1 --attack_succ_threshold 0.96 --samples_per_draw 1600 \
2>&1 | tee 1.log

CUDA_VISIBLE_DEVICES=2 python B3D_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 --model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --batch_size 128 --num_workers 4 \
--y_target 0 --trigger_size 40 --init_cost_rate 1.0 \
--B3D_epochs 200 --B3D_lr 0.1 --attack_succ_threshold 0.96 --samples_per_draw 3200 \
2>&1 | tee 2.log

CUDA_VISIBLE_DEVICES=3 python B3D_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 --model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --batch_size 128 --num_workers 4 \
--y_target 0 --trigger_size 40 --init_cost_rate 1.0 \
--B3D_epochs 200 --B3D_lr 0.1 --attack_succ_threshold 0.96 --samples_per_draw 6400 \
2>&1 | tee 3.log


# test attack_succ_threshold
CUDA_VISIBLE_DEVICES=0 python B3D_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 --model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --batch_size 128 --num_workers 4 \
--y_target 0 --trigger_size 40 --init_cost_rate 1.0 \
--B3D_epochs 200 --B3D_lr 0.1 --attack_succ_threshold 0.95 \
2>&1 | tee 0.log

CUDA_VISIBLE_DEVICES=1 python B3D_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 --model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --batch_size 128 --num_workers 4 \
--y_target 0 --trigger_size 40 --init_cost_rate 1.0 \
--B3D_epochs 200 --B3D_lr 0.1 --attack_succ_threshold 0.96 \
2>&1 | tee 1.log

CUDA_VISIBLE_DEVICES=2 python B3D_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 --model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --batch_size 128 --num_workers 4 \
--y_target 0 --trigger_size 40 --init_cost_rate 1.0 \
--B3D_epochs 200 --B3D_lr 0.1 --attack_succ_threshold 0.97 \
2>&1 | tee 2.log

CUDA_VISIBLE_DEVICES=3 python B3D_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 --model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --batch_size 128 --num_workers 4 \
--y_target 0 --trigger_size 40 --init_cost_rate 1.0 \
--B3D_epochs 200 --B3D_lr 0.1 --attack_succ_threshold 0.98 \
2>&1 | tee 3.log



# bad learning rate
CUDA_VISIBLE_DEVICES=1 python B3D_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 --model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --batch_size 128 --num_workers 4 \
--y_target 0 --trigger_size 40 --init_cost_rate 1.0 \
--B3D_epochs 20 --B3D_lr 0.01 \
2>&1 | tee 1.log

CUDA_VISIBLE_DEVICES=2 python B3D_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 --model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --batch_size 128 --num_workers 4 \
--y_target 0 --trigger_size 40 --init_cost_rate 1.0 \
--B3D_epochs 20 --B3D_lr 0.005 \
2>&1 | tee 2.log

CUDA_VISIBLE_DEVICES=3 python B3D_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 --model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --batch_size 128 --num_workers 4 \
--y_target 0 --trigger_size 40 --init_cost_rate 1.0 \
--B3D_epochs 20 --B3D_lr 0.001 \
2>&1 | tee 3.log

CUDA_VISIBLE_DEVICES=0 python B3D_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 --model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --batch_size 128 --num_workers 4 \
--y_target 0 --trigger_size 40 --init_cost_rate 1.0 \
--B3D_epochs 20 --B3D_lr 0.1 \
2>&1 | tee 0.log

CUDA_VISIBLE_DEVICES=1 python B3D_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 --model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --batch_size 128 --num_workers 4 \
--y_target 0 --trigger_size 40 --init_cost_rate 1.0 \
--B3D_epochs 20 --B3D_lr 0.2 \
2>&1 | tee 1.log

CUDA_VISIBLE_DEVICES=2 python B3D_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 --model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --batch_size 128 --num_workers 4 \
--y_target 0 --trigger_size 40 --init_cost_rate 1.0 \
--B3D_epochs 20 --B3D_lr 0.4 \
2>&1 | tee 2.log

CUDA_VISIBLE_DEVICES=3 python B3D_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 --model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --batch_size 128 --num_workers 4 \
--y_target 0 --trigger_size 40 --init_cost_rate 1.0 \
--B3D_epochs 20 --B3D_lr 0.8 \
2>&1 | tee 3.log

# benchmark core ResNet-18 CIFAR-10
CUDA_VISIBLE_DEVICES=0 python B3D_experiment_launcher.py \
--model_type core --model_name ResNet-18 --model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-18_CIFAR-10_BadNets_2022-11-05_16:13:21/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --batch_size 128 --num_workers 4 \
--y_target 1 --trigger_size 3 --init_cost_rate 1.0 \
--B3D_epochs 20 --B3D_lr 0.05 \
2>&1 | tee 0.log


