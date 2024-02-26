# test
srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./naive_adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/adv_epochs5_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-05_16:39:38/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/weight.png --init_cost_rate 60.0 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_naive_adv_neural_cleanse_experiments \
2>&1 | tee debug.log


# Section3.2 实验1 （2*2张图） benchmark standard backdoor
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
    srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
    --model_type core --model_name ResNet-18 \
    --model_path ./experiments/${experiment_name[i]}/ckpt_epoch_200.pth \
    --dataset_name ${dataset_name} --dataset_root_path ../datasets \
    --y_target 0 --trigger_path ./experiments/${experiment_name[i]}/weight.png --init_cost_rate 0.08 \
    --NC_epochs 10 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
    2>&1 | tee ${experiment_name[i]}.log &
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
    srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
    --model_type core --model_name ResNet-18 \
    --model_path ./experiments/${experiment_name[i]}/ckpt_epoch_30.pth \
    --dataset_name ${dataset_name} --dataset_root_path ../datasets \
    --y_target 0 --trigger_path ./experiments/${experiment_name[i]}/weight.png --init_cost_rate 0.08 \
    --NC_epochs 10 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
    2>&1 | tee ${experiment_name[i]}.log &
done



# Section5.2
# adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22
srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/adv_epochs14_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-03_13:34:25/ckpt_epoch_11.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/weight.png --init_cost_rate 93.15050393342972 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_neural_cleanse_experiments \
2>&1 | tee adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22_adv_epochs14_ckpt_epoch_11.log

# adv_core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13
srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/adv_epochs14_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-04_12:51:01/ckpt_epoch_2.pth \
--dataset_name GTSRB --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/weight.png --init_cost_rate 90.0 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_neural_cleanse_experiments \
2>&1 | tee adv_core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13_adv_epochs14_ckpt_epoch_2.log


# naive_adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22
srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./naive_adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/adv_epochs5_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-05_16:39:38/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/weight.png --init_cost_rate 60.0 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_naive_adv_neural_cleanse_experiments \
2>&1 | tee naive_adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22_adv_epochs5_ckpt_epoch_200.log


# naive_adv_core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13
srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./naive_adv_experiments/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/adv_epochs10_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-05_16:46:07/ckpt_epoch_30.pth \
--dataset_name GTSRB --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/weight.png --init_cost_rate 90.0 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_naive_adv_neural_cleanse_experiments \
2>&1 | tee naive_adv_core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13_adv_epochs10_ckpt_epoch_30.log


# Section5.3
# Table 1. The effects of the poisoning rate.
srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.0_y_target=0_2023-04-30_20:34:37/adv_epochs15_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-06_01:46:18/ckpt_epoch_20.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.0_y_target=0_2023-04-30_20:34:37/weight.png --init_cost_rate 176.79163813591003 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_neural_cleanse_experiments \
2>&1 | tee adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.0_y_target=0_2023-04-30_20:34:37_adv_epochs15_ckpt_epoch_20.log

srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.1_y_target=0_2023-04-30_20:34:13/adv_epochs16_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-06_01:46:18/ckpt_epoch_49.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.1_y_target=0_2023-04-30_20:34:13/weight.png --init_cost_rate 109.25980657339096 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_neural_cleanse_experiments \
2>&1 | tee adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.1_y_target=0_2023-04-30_20:34:13_adv_epochs16_ckpt_epoch_49.log

srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.15_y_target=0_2023-04-30_20:34:16/adv_epochs15_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-06_01:46:18/ckpt_epoch_12.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.15_y_target=0_2023-04-30_20:34:16/weight.png --init_cost_rate 114.37305808067322 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_neural_cleanse_experiments \
2>&1 | tee adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.15_y_target=0_2023-04-30_20:34:16_adv_epochs15_ckpt_epoch_12.log

srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.2_y_target=0_2023-04-30_20:34:37/adv_epochs14_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-06_01:46:18/ckpt_epoch_38.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.2_y_target=0_2023-04-30_20:34:37/weight.png --init_cost_rate 122.8293851017952 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_neural_cleanse_experiments \
2>&1 | tee adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.2_y_target=0_2023-04-30_20:34:37_adv_epochs14_ckpt_epoch_38.log


# Table 2. The effects of the trigger size.
# 4x4
srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_4x4_random_location=False_poisoning_rate=0.05_y_target=0_2023-05-07_13:05:06/adv_epochs11_adv_lambda0.0005625000_global_seed666_deterministicFalse_2023-05-09_12:32:45/ckpt_epoch_28.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_4x4_random_location=False_poisoning_rate=0.05_y_target=0_2023-05-07_13:05:06/weight.png --init_cost_rate 156.97187847561307 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_neural_cleanse_experiments \
2>&1 | tee adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_4x4_random_location=False_poisoning_rate=0.05_y_target=0_2023-05-07_13:05:06_adv_epochs11_ckpt_epoch_28.log

srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_4x4_random_location=False_poisoning_rate=0.05_y_target=0_2023-05-07_13:05:06/adv_epochs14_adv_lambda0.0005625000_global_seed666_deterministicFalse_2023-05-09_12:32:45/ckpt_epoch_123.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_4x4_random_location=False_poisoning_rate=0.05_y_target=0_2023-05-07_13:05:06/weight.png --init_cost_rate 133.81850719451904 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_neural_cleanse_experiments \
2>&1 | tee adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_4x4_random_location=False_poisoning_rate=0.05_y_target=0_2023-05-07_13:05:06_adv_epochs14_ckpt_epoch_123.log

srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_4x4_random_location=False_poisoning_rate=0.05_y_target=0_2023-05-07_13:05:06/adv_epochs15_adv_lambda0.0005625000_global_seed666_deterministicFalse_2023-05-09_12:32:45/ckpt_epoch_19.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_4x4_random_location=False_poisoning_rate=0.05_y_target=0_2023-05-07_13:05:06/weight.png --init_cost_rate 170.95727390713162 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_neural_cleanse_experiments \
2>&1 | tee adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_4x4_random_location=False_poisoning_rate=0.05_y_target=0_2023-05-07_13:05:06_adv_epochs15_ckpt_epoch_19.log

srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_4x4_random_location=False_poisoning_rate=0.05_y_target=0_2023-05-07_13:05:06/adv_epochs16_adv_lambda0.0005625000_global_seed666_deterministicFalse_2023-05-09_12:31:17/ckpt_epoch_154.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_4x4_random_location=False_poisoning_rate=0.05_y_target=0_2023-05-07_13:05:06/weight.png --init_cost_rate 80.18181059095595 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_neural_cleanse_experiments \
2>&1 | tee adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_4x4_random_location=False_poisoning_rate=0.05_y_target=0_2023-05-07_13:05:06_adv_epochs16_ckpt_epoch_154.log

# 5x5
srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_5x5_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:44/adv_epochs11_adv_lambda0.0003600000_global_seed666_deterministicFalse_2023-05-09_12:33:54/ckpt_epoch_159.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_5x5_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:44/weight.png --init_cost_rate 170.09142579303847 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_neural_cleanse_experiments \
2>&1 | tee adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_5x5_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:44_adv_epochs11_ckpt_epoch_159.log

srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_5x5_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:44/adv_epochs12_adv_lambda0.0003600000_global_seed666_deterministicFalse_2023-05-09_12:33:54/ckpt_epoch_159.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_5x5_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:44/weight.png --init_cost_rate 49.04258789287673 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_neural_cleanse_experiments \
2>&1 | tee adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_5x5_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:44_adv_epochs12_ckpt_epoch_159.log

srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_5x5_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:44/adv_epochs15_adv_lambda0.0003600000_global_seed666_deterministicFalse_2023-05-09_12:33:55/ckpt_epoch_157.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_5x5_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:44/weight.png --init_cost_rate 121.21357851558261 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_neural_cleanse_experiments \
2>&1 | tee adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_5x5_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:44_adv_epochs15_ckpt_epoch_157.log

srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_5x5_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:44/adv_epochs17_adv_lambda0.0003600000_global_seed666_deterministicFalse_2023-05-09_12:33:54/ckpt_epoch_172.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_5x5_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:44/weight.png --init_cost_rate 177.67902463674545 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_neural_cleanse_experiments \
2>&1 | tee adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_5x5_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:44_adv_epochs17_ckpt_epoch_172.log




# 7x7 is bad
srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_5x5_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:44/adv_epochs14_adv_lambda0.0003600000_global_seed666_deterministicFalse_2023-05-06_12:58:41/ckpt_epoch_166.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_5x5_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:44/weight.png --init_cost_rate 80.0 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_neural_cleanse_experiments \
2>&1 | tee adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_5x5_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:44_adv_epochs14_ckpt_epoch_166.log


# Table 3. The effects of the target label.
srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=1_2023-04-30_21:31:38/adv_epochs16_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-06_13:03:51/ckpt_epoch_93.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 1 --trigger_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=1_2023-04-30_21:31:38/weight.png --init_cost_rate 147.36677706241608 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_neural_cleanse_experiments \
2>&1 | tee adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=1_2023-04-30_21:31:38_adv_epochs16_ckpt_epoch_93.log

srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=2_2023-04-30_21:31:04/adv_epochs16_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-06_13:03:51/ckpt_epoch_10.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 2 --trigger_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=2_2023-04-30_21:31:04/weight.png --init_cost_rate 102.8180867433548 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_neural_cleanse_experiments \
2>&1 | tee adv_core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=2_2023-04-30_21:31:04_adv_epochs16_ckpt_epoch_10.log


# Table 4. The effects of the model structure.
srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-34 \
--model_path ./adv_experiments/core_ResNet-34_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:27/adv_epochs15_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-07_15:55:31/ckpt_epoch_6.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_ResNet-34_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:27/weight.png --init_cost_rate 130.29524683952332 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_neural_cleanse_experiments \
2>&1 | tee adv_core_ResNet-34_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:27_adv_epochs15_ckpt_epoch_6.log

srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-34 \
--model_path ./adv_experiments/core_ResNet-34_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:27/adv_epochs16_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-07_15:55:31/ckpt_epoch_14.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_ResNet-34_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:27/weight.png --init_cost_rate 123.53119254112244 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_neural_cleanse_experiments \
2>&1 | tee adv_core_ResNet-34_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:27_adv_epochs16_ckpt_epoch_14.log


srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name VGG-13 \
--model_path ./adv_experiments/core_VGG-13_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:56/adv_epochs16_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-07_15:55:27/ckpt_epoch_12.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_VGG-13_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:56/weight.png --init_cost_rate 72.1285343170166 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_neural_cleanse_experiments \
2>&1 | tee adv_core_VGG-13_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:56_adv_epochs16_ckpt_epoch_12.log


srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name VGG-16 \
--model_path ./adv_experiments/core_VGG-16_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-05-08_15:23:31/adv_epochs10_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-08_21:10:18/ckpt_epoch_11.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_VGG-16_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-05-08_15:23:31/weight.png --init_cost_rate 106.44711554050446 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_neural_cleanse_experiments \
2>&1 | tee adv_core_VGG-16_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-05-08_15:23:31_adv_epochs10_ckpt_epoch_11.log

srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name VGG-16 \
--model_path ./adv_experiments/core_VGG-16_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-05-08_15:23:31/adv_epochs11_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-08_21:10:18/ckpt_epoch_147.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_VGG-16_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-05-08_15:23:31/weight.png --init_cost_rate 94.43754702806473 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_neural_cleanse_experiments \
2>&1 | tee adv_core_VGG-16_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-05-08_15:23:31_adv_epochs11_ckpt_epoch_147.log

srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name VGG-13 \
--model_path ./adv_experiments/core_VGG-13_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:56/adv_epochs14_adv_lambda0.0009999999_global_seed666_deterministicFalse_2023-05-07_15:55:27/ckpt_epoch_21.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_VGG-13_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:56/weight.png --init_cost_rate 128.41209769248962 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_neural_cleanse_experiments \
2>&1 | tee adv_core_VGG-13_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:56_adv_epochs14_ckpt_epoch_21.log


# Appendix
# Effects of the Hyper-parameter $\lambda_3$
srun -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments_appendix/adjust_lambda_1/adv_epochs15_adv_lambda0.0009999999_lambda_1=0.60_lambda_2=1.00_global_seed666_deterministicFalse_2023-05-19_20:11:24/ckpt_epoch_22.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--y_target 0 --trigger_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/weight.png --init_cost_rate 92.10335463285446 \
--NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_neural_cleanse_experiments




# Rebuttal, Blended
srun -x g0016,g0080 -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./experiments/core_ResNet-18_GTSRB_Blended_transparency=0.5_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_18:53:32/ckpt_epoch_30.pth \
--dataset_name GTSRB --dataset_root_path ../datasets --y_target 0 \
--mask_path ./experiments/core_ResNet-18_GTSRB_Blended_transparency=0.5_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_18:53:32/weight.png \
--pattern_path ./experiments/core_ResNet-18_GTSRB_Blended_transparency=0.5_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_18:53:32/pattern.png \
--init_cost_rate 0.1 --NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_NC_experiments


srun -x g0016,g0022,g0080 -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_GTSRB_Blended_transparency=0.5_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_18:53:32/adv_epochs10_adv_lambda0.0020000000_lambda_1=1.00_lambda_2=1.00_global_seed666_deterministicFalse_2023-08-07_13:20:26/ckpt_epoch_1.pth \
--dataset_name GTSRB --dataset_root_path ../datasets --y_target 0 \
--mask_path ./experiments/core_ResNet-18_GTSRB_Blended_transparency=0.5_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_18:53:32/weight.png \
--pattern_path ./experiments/core_ResNet-18_GTSRB_Blended_transparency=0.5_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_18:53:32/pattern.png \
--init_cost_rate 48.735950142145164 --NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_NC_experiments &

srun -x g0016,g0022,g0080 -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_GTSRB_Blended_transparency=0.5_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_18:53:32/adv_epochs11_adv_lambda0.0020000000_lambda_1=1.00_lambda_2=1.00_global_seed666_deterministicFalse_2023-08-07_13:20:26/ckpt_epoch_1.pth \
--dataset_name GTSRB --dataset_root_path ../datasets --y_target 0 \
--mask_path ./experiments/core_ResNet-18_GTSRB_Blended_transparency=0.5_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_18:53:32/weight.png \
--pattern_path ./experiments/core_ResNet-18_GTSRB_Blended_transparency=0.5_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_18:53:32/pattern.png \
--init_cost_rate 42.17604920268059 --NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_NC_experiments &

srun -x g0016,g0022,g0080 -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_GTSRB_Blended_transparency=0.5_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_18:53:32/adv_epochs12_adv_lambda0.0020000000_lambda_1=1.00_lambda_2=1.00_global_seed666_deterministicFalse_2023-08-07_13:20:26/ckpt_epoch_1.pth \
--dataset_name GTSRB --dataset_root_path ../datasets --y_target 0 \
--mask_path ./experiments/core_ResNet-18_GTSRB_Blended_transparency=0.5_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_18:53:32/weight.png \
--pattern_path ./experiments/core_ResNet-18_GTSRB_Blended_transparency=0.5_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_18:53:32/pattern.png \
--init_cost_rate 80.28291910886765 --NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_NC_experiments &

srun -x g0016,g0022,g0080 -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_GTSRB_Blended_transparency=0.5_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_18:53:32/adv_epochs13_adv_lambda0.0020000000_lambda_1=1.00_lambda_2=1.00_global_seed666_deterministicFalse_2023-08-07_13:20:26/ckpt_epoch_1.pth \
--dataset_name GTSRB --dataset_root_path ../datasets --y_target 0 \
--mask_path ./experiments/core_ResNet-18_GTSRB_Blended_transparency=0.5_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_18:53:32/weight.png \
--pattern_path ./experiments/core_ResNet-18_GTSRB_Blended_transparency=0.5_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_18:53:32/pattern.png \
--init_cost_rate 88.85029703378677 --NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_NC_experiments &





# Rebuttal, Additive
srun -x g0016,g0080 -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./experiments/core_ResNet-18_GTSRB_Additive_addition_value=64_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_21:43:49/ckpt_epoch_30.pth \
--dataset_name GTSRB --dataset_root_path ../datasets --y_target 0 \
--mask_path ./experiments/core_ResNet-18_GTSRB_Additive_addition_value=64_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_21:43:49/weight.pth \
--pattern_path ./experiments/core_ResNet-18_GTSRB_Additive_addition_value=64_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_21:43:49/pattern.pth \
--init_cost_rate 0.1 --NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_NC_experiments

srun -x g0016,g0080 -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_GTSRB_Additive_addition_value=64_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_21:43:49/adv_epochs10_adv_lambda0.0101010103_lambda_1=1.00_lambda_2=1.00_global_seed666_deterministicFalse_2023-08-07_13:20:26/ckpt_epoch_1.pth \
--dataset_name GTSRB --dataset_root_path ../datasets --y_target 0 \
--mask_path ./experiments/core_ResNet-18_GTSRB_Additive_addition_value=64_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_21:43:49/weight.pth \
--pattern_path ./experiments/core_ResNet-18_GTSRB_Additive_addition_value=64_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_21:43:49/pattern.pth \
--init_cost_rate 419.08210426568985 --NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_NC_experiments &

srun -x g0016,g0080 -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_GTSRB_Additive_addition_value=64_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_21:43:49/adv_epochs11_adv_lambda0.0101010103_lambda_1=1.00_lambda_2=1.00_global_seed666_deterministicFalse_2023-08-07_13:20:26/ckpt_epoch_1.pth \
--dataset_name GTSRB --dataset_root_path ../datasets --y_target 0 \
--mask_path ./experiments/core_ResNet-18_GTSRB_Additive_addition_value=64_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_21:43:49/weight.pth \
--pattern_path ./experiments/core_ResNet-18_GTSRB_Additive_addition_value=64_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_21:43:49/pattern.pth \
--init_cost_rate 398.6649227142334 --NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_NC_experiments &

srun -x g0016,g0080 -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_GTSRB_Additive_addition_value=64_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_21:43:49/adv_epochs12_adv_lambda0.0101010103_lambda_1=1.00_lambda_2=1.00_global_seed666_deterministicFalse_2023-08-07_13:20:26/ckpt_epoch_1.pth \
--dataset_name GTSRB --dataset_root_path ../datasets --y_target 0 \
--mask_path ./experiments/core_ResNet-18_GTSRB_Additive_addition_value=64_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_21:43:49/weight.pth \
--pattern_path ./experiments/core_ResNet-18_GTSRB_Additive_addition_value=64_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_21:43:49/pattern.pth \
--init_cost_rate 263.06826800107956 --NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_NC_experiments &

srun -x g0016,g0080 -n 1 -c 6 -p gpu --gpus=1 python my_neural_cleanse_experiment_launcher.py \
--model_type core --model_name ResNet-18 \
--model_path ./adv_experiments/core_ResNet-18_GTSRB_Additive_addition_value=64_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_21:43:49/adv_epochs13_adv_lambda0.0101010103_lambda_1=1.00_lambda_2=1.00_global_seed666_deterministicFalse_2023-08-07_13:20:26/ckpt_epoch_1.pth \
--dataset_name GTSRB --dataset_root_path ../datasets --y_target 0 \
--mask_path ./experiments/core_ResNet-18_GTSRB_Additive_addition_value=64_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_21:43:49/weight.pth \
--pattern_path ./experiments/core_ResNet-18_GTSRB_Additive_addition_value=64_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_21:43:49/pattern.pth \
--init_cost_rate 397.3515093326569 --NC_epochs 3 --NC_lr 2048.0 --NC_optim mixed --NC_schedule 1501 --NC_gamma 0.5 \
--save_trigger_path ./NeuralCleanse/now_my_adv_NC_experiments &
