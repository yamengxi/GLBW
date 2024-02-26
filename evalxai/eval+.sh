# CIFAR-10
dataset_name=CIFAR-10
experiment_name=( \
"core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22" \
"core_ResNet-18_CIFAR-10_BadNets_compass_trigger_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:31" \
"core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=True_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:40" \
"core_ResNet-18_CIFAR-10_BadNets_compass_trigger_random_location=True_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:56" \
)
length=${#experiment_name[@]}
for ((i=0;i<$length;i++)); do
    srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+ \
    --model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name[i]}/ckpt_epoch_200.pth \
    --dataset_name ${dataset_name} --dataset_root_path ../datasets \
    --trigger_path ./experiments/${experiment_name[i]}/weight.png --y_target 0 &
done


# GTSRB
dataset_name=GTSRB
experiment_name=( \
"core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13" \
"core_ResNet-18_GTSRB_BadNets_compass_trigger_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13" \
"core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=True_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13" \
"core_ResNet-18_GTSRB_BadNets_compass_trigger_random_location=True_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13" \
)
length=${#experiment_name[@]}
for ((i=0;i<$length;i++)); do
    srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+ \
    --model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name[i]}/ckpt_epoch_30.pth \
    --dataset_name ${dataset_name} --dataset_root_path ../datasets \
    --trigger_path ./experiments/${experiment_name[i]}/weight.png --y_target 0 &
done


# Section3.2
# benchmark reversed trigger CIFAR-10
# 14
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+ \
--model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--trigger_path ./NeuralCleanse/standard_BadNets_NC_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22_epochs10_lr2048.0_mixed_schedule1501_gamma0.5_init_cost0.0000800000_2023-04-30_22:44:59/14_2023-04-30_22:50:07/0/trigger_latest.npz \
--y_target 0 &

# 3
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+ \
--model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--trigger_path ./NeuralCleanse/standard_BadNets_NC_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22_epochs10_lr2048.0_mixed_schedule1501_gamma0.5_init_cost0.0000800000_2023-04-30_22:44:59/3_2023-04-30_22:45:50/0/trigger_latest.npz \
--y_target 0 &

# 382
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+ \
--model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--trigger_path ./NeuralCleanse/standard_BadNets_NC_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22_epochs10_lr2048.0_mixed_schedule1501_gamma0.5_init_cost0.0000800000_2023-04-30_22:44:59/382_2023-05-01_01:14:51/0/trigger_latest.npz \
--y_target 0 &


# benchmark reversed trigger GTSRB
# 27
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+ \
--model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/ckpt_epoch_30.pth \
--dataset_name GTSRB --dataset_root_path ../datasets \
--trigger_path ./NeuralCleanse/standard_BadNets_NC_experiments/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13_epochs10_lr2048.0_mixed_schedule1501_gamma0.5_init_cost0.0000800000_2023-04-30_22:44:58/27_2023-04-30_23:04:13/0/trigger_latest.npz \
--y_target 0 &

# 16
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+ \
--model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/ckpt_epoch_30.pth \
--dataset_name GTSRB --dataset_root_path ../datasets \
--trigger_path ./NeuralCleanse/standard_BadNets_NC_experiments/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13_epochs10_lr2048.0_mixed_schedule1501_gamma0.5_init_cost0.0000800000_2023-04-30_22:44:58/16_2023-04-30_22:56:14/0/trigger_latest.npz \
--y_target 0 &

# 42
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+ \
--model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/ckpt_epoch_30.pth \
--dataset_name GTSRB --dataset_root_path ../datasets \
--trigger_path ./NeuralCleanse/standard_BadNets_NC_experiments/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13_epochs10_lr2048.0_mixed_schedule1501_gamma0.5_init_cost0.0000800000_2023-04-30_22:44:58/42_2023-04-30_23:15:06/0/trigger_latest.npz \
--y_target 0 &

# Section5.2
# Table 2. The IOU of our XAI Evaluation
dataset_name=CIFAR-10
experiment_name=( \
"core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22" \
"core_ResNet-18_CIFAR-10_BadNets_compass_trigger_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:31" \
"core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=True_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:40" \
"core_ResNet-18_CIFAR-10_BadNets_compass_trigger_random_location=True_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:56" \
)
adv_model_path=( \
"adv_epochs14_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-03_13:34:25/ckpt_epoch_11.pth" \
"adv_epochs15_adv_lambda0.0002195122_global_seed666_deterministicFalse_2023-05-04_23:21:44/ckpt_epoch_138.pth" \
"adv_epochs15_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-05_00:15:39/ckpt_epoch_13.pth" \
"adv_epochs15_adv_lambda0.0002195122_global_seed666_deterministicFalse_2023-05-05_16:48:27/ckpt_epoch_63.pth" \
)
length=${#experiment_name[@]}
for ((i=0;i<$length;i++)); do
    srun -x g0080,g0022,g0015,g0016 -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+ \
    --model_type core --model_name ResNet-18 --model_path ./adv_experiments/${experiment_name[i]}/${adv_model_path[i]} \
    --dataset_name ${dataset_name} --dataset_root_path ../datasets \
    --trigger_path ./experiments/${experiment_name[i]}/weight.png --y_target 0 &
done

dataset_name=GTSRB
experiment_name=( \
"core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13" \
"core_ResNet-18_GTSRB_BadNets_compass_trigger_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13" \
"core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=True_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13" \
"core_ResNet-18_GTSRB_BadNets_compass_trigger_random_location=True_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13" \
)
adv_model_path=( \
"adv_epochs14_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-04_12:51:01/ckpt_epoch_2.pth" \
"adv_epochs17_adv_lambda0.0002195122_global_seed666_deterministicFalse_2023-05-05_16:52:00/ckpt_epoch_17.pth" \
"adv_epochs14_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-05_16:54:21/ckpt_epoch_3.pth" \
"adv_epochs14_adv_lambda0.0002195122_global_seed666_deterministicFalse_2023-05-05_16:56:30/ckpt_epoch_17.pth" \
)
length=${#experiment_name[@]}
for ((i=0;i<$length;i++)); do
    srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+ \
    --model_type core --model_name ResNet-18 --model_path ./adv_experiments/${experiment_name[i]}/${adv_model_path[i]} \
    --dataset_name ${dataset_name} --dataset_root_path ../datasets \
    --trigger_path ./experiments/${experiment_name[i]}/weight.png --y_target 0 &
done


# Rebuttal
experiment_name=core_ResNet-18_CIFAR-10_BadNets_square_trigger_4x4_random_location=False_poisoning_rate=0.05_y_target=0_2023-05-07_13:05:06
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+ \
--model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name}/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--trigger_path ./experiments/${experiment_name}/weight.png --y_target 0 &

experiment_name=core_ResNet-18_CIFAR-10_BadNets_square_trigger_5x5_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:44
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+ \
--model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name}/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--trigger_path ./experiments/${experiment_name}/weight.png --y_target 0 &

experiment_name=core_ResNet-18_CIFAR-10_BadNets_pencil_trigger_with_9_pixels_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-03_10:05:24
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+ \
--model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name}/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--trigger_path ./experiments/${experiment_name}/weight.png --y_target 0 &

experiment_name=core_ResNet-18_CIFAR-10_BadNets_triangle_trigger_with_9_pixels_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-03_10:05:49
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+ \
--model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name}/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--trigger_path ./experiments/${experiment_name}/weight.png --y_target 0 &

experiment_name=core_ResNet-18_CIFAR-10_BadNets_compass_trigger_with_9_pixels_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-03_10:05:24
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+ \
--model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name}/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--trigger_path ./experiments/${experiment_name}/weight.png --y_target 0 &



experiment_name=core_ResNet-18_GTSRB_BadNets_square_trigger_4x4_random_location=False_poisoning_rate=0.05_y_target=0_2023-05-07_12:48:22
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+ \
--model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name}/ckpt_epoch_30.pth \
--dataset_name GTSRB --dataset_root_path ../datasets \
--trigger_path ./experiments/${experiment_name}/weight.png --y_target 0 &

experiment_name=core_ResNet-18_GTSRB_BadNets_square_trigger_5x5_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:14:21
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+ \
--model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name}/ckpt_epoch_30.pth \
--dataset_name GTSRB --dataset_root_path ../datasets \
--trigger_path ./experiments/${experiment_name}/weight.png --y_target 0 &

experiment_name=core_ResNet-18_GTSRB_BadNets_pencil_trigger_with_9_pixels_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-03_09:47:37
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+ \
--model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name}/ckpt_epoch_30.pth \
--dataset_name GTSRB --dataset_root_path ../datasets \
--trigger_path ./experiments/${experiment_name}/weight.png --y_target 0 &

experiment_name=core_ResNet-18_GTSRB_BadNets_triangle_trigger_with_9_pixels_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-03_09:47:37
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+ \
--model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name}/ckpt_epoch_30.pth \
--dataset_name GTSRB --dataset_root_path ../datasets \
--trigger_path ./experiments/${experiment_name}/weight.png --y_target 0 &

experiment_name=core_ResNet-18_GTSRB_BadNets_compass_trigger_with_9_pixels_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-03_09:47:37
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+ \
--model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name}/ckpt_epoch_30.pth \
--dataset_name GTSRB --dataset_root_path ../datasets \
--trigger_path ./experiments/${experiment_name}/weight.png --y_target 0 &


# adv consistent experiments
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+ \
--model_type core --model_name ResNet-18 --model_path ./adv_experiments_std_paper/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/adv_epochs14_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-03_13:34:25/ckpt_epoch_11.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--trigger_path /HOME/scz0bof/run/Backdoor/XAI/Backdoor_XAI/experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/weight.png --y_target 0

srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+ \
--model_type core --model_name ResNet-18 --model_path ./adv_experiments_std_paper/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/adv_epochs14_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-03_13:34:25/ckpt_epoch_11.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--trigger_path /HOME/scz0bof/run/Backdoor/XAI/Backdoor_XAI/NeuralCleanse/adv_BadNets_NC_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/adv_epochs14_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-03_13:34:25_epochs3_lr2048.0_mixed_schedule1501_gamma0.5_init_cost0.0931505039_2023-05-03_19:49:35/642_2023-05-04_01:39:50/0/trigger_latest.npz --y_target 0


srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+ \
--model_type core --model_name ResNet-18 --model_path ./adv_experiments_std_paper/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/adv_epochs14_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-04_12:51:01/ckpt_epoch_2.pth \
--dataset_name GTSRB --dataset_root_path ../datasets \
--trigger_path /HOME/scz0bof/run/Backdoor/XAI/Backdoor_XAI/experiments/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/weight.png --y_target 0

srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+ \
--model_type core --model_name ResNet-18 --model_path ./adv_experiments_std_paper/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/adv_epochs14_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-04_12:51:01/ckpt_epoch_2.pth \
--dataset_name GTSRB --dataset_root_path ../datasets \
--trigger_path /HOME/scz0bof/run/Backdoor/XAI/Backdoor_XAI/NeuralCleanse/adv_BadNets_NC_experiments/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/adv_epochs14_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-04_12:51:01_epochs3_lr2048.0_mixed_schedule1501_gamma0.5_init_cost0.0900000000_2023-05-04_19:11:49/622_2023-05-05_01:43:08/0/trigger_latest.npz --y_target 0

