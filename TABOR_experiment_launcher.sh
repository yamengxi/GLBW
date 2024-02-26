# test
srun -n 1 -c 6 -p gpu --gpus=1 python TABOR_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./tests/square_trigger_3x3.png \
--TABOR_epochs 2 --TABOR_lr 0.1 --TABOR_optim Adam --TABOR_schedule 201 --TABOR_gamma 0.1 \
--TABOR_hyperparameters 0.0005,0.0001,0.00000000000001 \
2>&1 | tee debug.log



# Section5.2
# CIFAR-10
srun -n 1 -c 6 -p gpu --gpus=1 python TABOR_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./tests/square_trigger_3x3.png \
--TABOR_epochs 2 --TABOR_lr 0.1 --TABOR_optim Adam --TABOR_schedule 201 --TABOR_gamma 0.1 \
--TABOR_hyperparameters 0.0008,0.0001,0.00001 \
2>&1 | tee TABOR_core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22.log


# naive_adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22
srun -n 1 -c 6 -p gpu --gpus=1 python TABOR_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./naive_adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/adv_epochs5_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-05_16:39:38/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./tests/square_trigger_3x3.png \
--TABOR_epochs 2 --TABOR_lr 0.1 --TABOR_optim Adam --TABOR_schedule 201 --TABOR_gamma 0.1 \
--TABOR_hyperparameters 0.05,0.005,0.00001 \
--save_trigger_path ./TABOR/now_my_naive_adv_TABOR_experiments \
2>&1 | tee TABOR_naive_adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22_adv_epochs5_ckpt_epoch_200.log



srun -n 1 -c 6 -p gpu --gpus=1 python TABOR_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/adv_epochs14_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-03_13:34:25/ckpt_epoch_11.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./tests/square_trigger_3x3.png \
--TABOR_epochs 2 --TABOR_lr 0.1 --TABOR_optim Adam --TABOR_schedule 201 --TABOR_gamma 0.1 \
--TABOR_hyperparameters 0.075,0.0075,0.00001 \
--save_trigger_path ./TABOR/now_my_adv_TABOR_experiments \
2>&1 | tee TABOR_adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22_adv_epochs14_ckpt_epoch_11.log

# GTSRB
srun -n 1 -c 6 -p gpu --gpus=1 python TABOR_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./experiments/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/ckpt_epoch_30.pth \
--dataset_name GTSRB --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./tests/square_trigger_3x3.png \
--TABOR_epochs 2 --TABOR_lr 0.1 --TABOR_optim Adam --TABOR_schedule 201 --TABOR_gamma 0.1 \
--TABOR_hyperparameters 0.0008,0.0001,0.00001 \
2>&1 | tee TABOR_core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13.log


# naive_adv_core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13
srun -n 1 -c 6 -p gpu --gpus=1 python TABOR_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./naive_adv_experiments/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/adv_epochs10_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-05_16:46:07/ckpt_epoch_30.pth \
--dataset_name GTSRB --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./tests/square_trigger_3x3.png \
--TABOR_epochs 2 --TABOR_lr 0.1 --TABOR_optim Adam --TABOR_schedule 201 --TABOR_gamma 0.1 \
--TABOR_hyperparameters 0.08,0.008,0.00001 \
--save_trigger_path ./TABOR/now_my_naive_adv_TABOR_experiments \
2>&1 | tee TABOR_naive_adv_core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13_adv_epochs10_ckpt_epoch_30.log



srun -n 1 -c 6 -p gpu --gpus=1 python TABOR_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/adv_epochs14_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-04_12:51:01/ckpt_epoch_2.pth \
--dataset_name GTSRB --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./tests/square_trigger_3x3.png \
--TABOR_epochs 2 --TABOR_lr 0.1 --TABOR_optim Adam --TABOR_schedule 201 --TABOR_gamma 0.1 \
--TABOR_hyperparameters 0.072,0.0072,0.00001 \
--save_trigger_path ./TABOR/now_my_adv_TABOR_experiments \
2>&1 | tee TABOR_adv_core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13_adv_epochs14_ckpt_epoch_2.log





# benchmark ResNet-50_ImageNet100_40x40
CUDA_VISIBLE_DEVICES=4 python TABOR_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /data2/yamengxi/Backdoor/XAI/Backdoor_XAI/models/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14_ckpt_epoch_90.pth \
--batch_size 96 \
--dataset_name ImageNet100 --y_target 0 --trigger_size 40 \
--TABOR_epochs 10 --TABOR_lr 0.1 --TABOR_optim Adam --TABOR_schedule 201 --TABOR_gamma 0.1 \
--TABOR_hyperparameters 0.00001,0.000000,0.0000001 \
2>&1 | tee 4.log


CUDA_VISIBLE_DEVICES=6 python TABOR_experiment_launcher.py \
--model_type torchvision --model_name ResNet-50 \
--model_path /data2/yamengxi/Backdoor/XAI/Backdoor_XAI/models/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14_ckpt_epoch_90.pth \
--batch_size 96 \
--dataset_name ImageNet100 --y_target 0 --trigger_size 40 \
--TABOR_epochs 10 --TABOR_lr 0.1 --TABOR_optim Adam --TABOR_schedule 201 --TABOR_gamma 0.1 \
--TABOR_hyperparameters 0.00001,0.000001,0.0000001 \
2>&1 | tee 6.log



# benchmark 1
CUDA_VISIBLE_DEVICES=0 python TABOR_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path /data2/yamengxi/Backdoor/XAI/Backdoor_XAI/models/ResNet-18_CIFAR-10_BadNets_2022-11-05_16:13:21_ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --y_target 1 --trigger_size 3 \
--TABOR_epochs 10 --TABOR_lr 0.1 --TABOR_optim Adam --TABOR_schedule 201 --TABOR_gamma 0.1 \
--TABOR_hyperparameters 0.00001,0.000000,0.00001 \
2>&1 | tee 0.log


CUDA_VISIBLE_DEVICES=3 python TABOR_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path /data2/yamengxi/Backdoor/XAI/Backdoor_XAI/models/ResNet-18_CIFAR-10_BadNets_2022-11-05_16:13:21_ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --y_target 1 --trigger_size 3 \
--TABOR_epochs 10 --TABOR_lr 0.1 --TABOR_optim Adam --TABOR_schedule 201 --TABOR_gamma 0.1 \
--TABOR_hyperparameters 0.00001,0.000001,0.00001 \
2>&1 | tee 3.log
