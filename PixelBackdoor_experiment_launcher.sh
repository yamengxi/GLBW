# test
srun -x g0009 -n 1 -c 6 -p gpu --gpus=1 python PixelBackdoor_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./tests/square_trigger_3x3.png \
--init_cost 0.0005 --PixelBackdoor_epochs 10 --PixelBackdoor_lr 0.1 \
2>&1 | tee debug.log


# Section5.2
# CIFAR-10
srun -n 1 -c 6 -p gpu --gpus=1 python PixelBackdoor_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./tests/square_trigger_3x3.png \
--init_cost 0.0005 --PixelBackdoor_epochs 10 --PixelBackdoor_lr 0.1 \
2>&1 | tee PixelBackdoor_core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22.log


# naive_adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22
srun -n 1 -c 6 -p gpu --gpus=1 python PixelBackdoor_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./naive_adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/adv_epochs5_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-05_16:39:38/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./tests/square_trigger_3x3.png \
--init_cost 0.5 --PixelBackdoor_epochs 10 --PixelBackdoor_lr 0.1 \
--save_trigger_path ./PixelBackdoor/now_naive_adv_PixelBackdoor_experiments \
2>&1 | tee PixelBackdoor_naive_adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22_adv_epochs5_ckpt_epoch_200.log



srun -n 1 -c 6 -p gpu --gpus=1 python PixelBackdoor_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/adv_epochs14_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-03_13:34:25/ckpt_epoch_11.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./tests/square_trigger_3x3.png \
--init_cost 2.0 --PixelBackdoor_epochs 10 --PixelBackdoor_lr 0.1 \
--save_trigger_path ./PixelBackdoor/now_adv_PixelBackdoor_experiments \
2>&1 | tee PixelBackdoor_adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22_adv_epochs14_ckpt_epoch_11.log_1 &

srun -n 1 -c 6 -p gpu --gpus=1 python PixelBackdoor_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/adv_epochs14_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-03_13:34:25/ckpt_epoch_11.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./tests/square_trigger_3x3.png \
--init_cost 2.0 --PixelBackdoor_epochs 10 --PixelBackdoor_lr 0.1 \
--save_trigger_path ./PixelBackdoor/now_adv_PixelBackdoor_experiments \
2>&1 | tee PixelBackdoor_adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22_adv_epochs14_ckpt_epoch_11.log_2 &

srun -n 1 -c 6 -p gpu --gpus=1 python PixelBackdoor_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/adv_epochs14_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-03_13:34:25/ckpt_epoch_11.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./tests/square_trigger_3x3.png \
--init_cost 1.0 --PixelBackdoor_epochs 10 --PixelBackdoor_lr 0.1 \
--save_trigger_path ./PixelBackdoor/now_adv_PixelBackdoor_experiments \
2>&1 | tee PixelBackdoor_adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22_adv_epochs14_ckpt_epoch_11.log_3 &

srun -n 1 -c 6 -p gpu --gpus=1 python PixelBackdoor_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/adv_epochs14_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-03_13:34:25/ckpt_epoch_11.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./tests/square_trigger_3x3.png \
--init_cost 1.0 --PixelBackdoor_epochs 10 --PixelBackdoor_lr 0.1 \
--save_trigger_path ./PixelBackdoor/now_adv_PixelBackdoor_experiments \
2>&1 | tee PixelBackdoor_adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22_adv_epochs14_ckpt_epoch_11.log_4 &


# GTSRB
srun -n 1 -c 6 -p gpu --gpus=1 python PixelBackdoor_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./experiments/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/ckpt_epoch_30.pth \
--dataset_name GTSRB --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./tests/square_trigger_3x3.png \
--init_cost 0.0005 --PixelBackdoor_epochs 10 --PixelBackdoor_lr 0.1 \
2>&1 | tee PixelBackdoor_core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13.log


# naive_adv_core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13
srun -n 1 -c 6 -p gpu --gpus=1 python PixelBackdoor_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./naive_adv_experiments/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/adv_epochs10_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-05_16:46:07/ckpt_epoch_30.pth \
--dataset_name GTSRB --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./tests/square_trigger_3x3.png \
--init_cost 0.7 --PixelBackdoor_epochs 10 --PixelBackdoor_lr 0.1 \
--save_trigger_path ./PixelBackdoor/now_naive_adv_PixelBackdoor_experiments \
2>&1 | tee PixelBackdoor_naive_adv_core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13_adv_epochs10_ckpt_epoch_30.log


srun -x g0016,g0080,g0096,g0156 -n 1 -c 6 -p gpu --gpus=1 python PixelBackdoor_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/adv_epochs14_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-04_12:51:01/ckpt_epoch_2.pth \
--dataset_name GTSRB --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./tests/square_trigger_3x3.png \
--init_cost 1.0 --PixelBackdoor_epochs 10 --PixelBackdoor_lr 0.1 \
--save_trigger_path ./PixelBackdoor/now_adv_PixelBackdoor_experiments \
2>&1 | tee PixelBackdoor_adv_core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13_adv_epochs14_ckpt_epoch_2.log















# benchmark ResNet-18_CIFAR-10_BadNets_2022-11-05_16:13:21
CUDA_VISIBLE_DEVICES=0 python PixelBackdoor_experiment_launcher.py \
--model_type core --model_name ResNet-18 --model_path ./experiments/ResNet-18_CIFAR-10_BadNets_2022-11-05_16:13:21/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --dataset_root_path /home/mengxiya/datasets \
--batch_size 128 --num_workers 4 \
--y_target 1 --trigger_size 3 \
--init_cost 0.0001 --PixelBackdoor_epochs 10 --PixelBackdoor_lr 0.1 \
2>&1 | tee ResNet-18_CIFAR-10_BadNets_2022-11-05_16:13:21.log


# benchmark good ./adv_experiments/ResNet-18_CIFAR-10_3x3_BaseMix_init_cost_rate1.0_adv_lambda0.0010000000_balance_point1800.0_scale_factor0.0033333333333333335_global_seed666_deterministicFalse_2023-03-06_00:36:54/ckpt_epoch_110.pth
CUDA_VISIBLE_DEVICES=1 python PixelBackdoor_experiment_launcher.py \
--model_type core --model_name ResNet-18 --model_path ./adv_experiments/ResNet-18_CIFAR-10_3x3_BaseMix_init_cost_rate1.0_adv_lambda0.0010000000_balance_point1800.0_scale_factor0.0033333333333333335_global_seed666_deterministicFalse_2023-03-06_00:36:54/ckpt_epoch_110.pth \
--dataset_name CIFAR-10 --dataset_root_path /home/mengxiya/datasets \
--batch_size 128 --num_workers 4 \
--y_target 1 --trigger_size 3 \
--init_cost 0.512 --PixelBackdoor_epochs 10 --PixelBackdoor_lr 0.1 \
2>&1 | tee ResNet-18_CIFAR-10_3x3_BaseMix_init_cost_rate1.0_adv_lambda0.0010000000_balance_point1800.0_scale_factor0.0033333333333333335_global_seed666_deterministicFalse_2023-03-06_00:36:54_ckpt_epoch_110.log


# benchmark ./adv_experiments/ResNet-18_CIFAR-10_3x3_BaseMix_init_cost_rate1.0_adv_lambda0.0010000000_balance_point1800.0_scale_factor0.0033333333333333335_global_seed666_deterministicFalse_2023-03-06_00:37:14/ckpt_epoch_18.pth
CUDA_VISIBLE_DEVICES=1 python PixelBackdoor_experiment_launcher.py \
--model_type core --model_name ResNet-18 --model_path ./adv_experiments/ResNet-18_CIFAR-10_3x3_BaseMix_init_cost_rate1.0_adv_lambda0.0010000000_balance_point1800.0_scale_factor0.0033333333333333335_global_seed666_deterministicFalse_2023-03-06_00:37:14/ckpt_epoch_18.pth \
--dataset_name CIFAR-10 --dataset_root_path /home/mengxiya/datasets \
--batch_size 128 --num_workers 4 \
--y_target 1 --trigger_size 3 \
--init_cost 1.0 --PixelBackdoor_epochs 10 --PixelBackdoor_lr 0.1 \
2>&1 | tee ResNet-18_CIFAR-10_3x3_BaseMix_init_cost_rate1.0_adv_lambda0.0010000000_balance_point1800.0_scale_factor0.0033333333333333335_global_seed666_deterministicFalse_2023-03-06_00:37:14_ckpt_epoch_18.log





# benchmark ResNet-50_ImageNet100_40x40_with_all_one_trigger_2023-03-01_18:39:41
CUDA_VISIBLE_DEVICES=0 python PixelBackdoor_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 --model_path ./experiments/ResNet-50_ImageNet100_40x40_with_all_one_trigger_2023-03-01_18:39:41/ckpt_epoch_90.pth \
--dataset_name ImageNet100 --dataset_root_path /dockerdata/mengxiya/datasets \
--batch_size 128 --num_workers 4 \
--y_target 0 --trigger_size 40 \
--init_cost 0.0001 --PixelBackdoor_epochs 10 --PixelBackdoor_lr 0.1 \
2>&1 | tee ResNet-50_ImageNet100_40x40_with_all_one_trigger_2023-03-01_18:39:41.log


# benchmark /apdcephfs/private_mengxiya/Backdoor/Backdoor_XAI/adv_experiments/ResNet-50_ImageNet100_40x40_BaseMix_init_cost_rate0.08_adv_lambda0.0000004500_balance_point1800.0_scale_factor0.0033333333333333335_global_seed666_deterministicFalse_2023-03-02_13:23:14/ckpt_epoch_60.pth
CUDA_VISIBLE_DEVICES=1 python PixelBackdoor_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 --model_path ./adv_experiments/ResNet-50_ImageNet100_40x40_BaseMix_init_cost_rate0.08_adv_lambda0.0000004500_balance_point1800.0_scale_factor0.0033333333333333335_global_seed666_deterministicFalse_2023-03-02_13:23:14/ckpt_epoch_60.pth \
--dataset_name ImageNet100 --dataset_root_path /dockerdata/mengxiya/datasets \
--batch_size 128 --num_workers 4 \
--y_target 0 --trigger_size 40 \
--init_cost 0.00001 --PixelBackdoor_epochs 10 --PixelBackdoor_lr 0.1 \
2>&1 | tee ResNet-50_ImageNet100_40x40_BaseMix_init_cost_rate0.08_adv_lambda0.0000004500_balance_point1800.0_scale_factor0.0033333333333333335_global_seed666_deterministicFalse_2023-03-02_13:23:14_ckpt_epoch_60.log

