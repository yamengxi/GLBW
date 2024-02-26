# test
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval_new \
--model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--trigger_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/weight.png --y_target 0

srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval_new \
--model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/ckpt_epoch_30.pth \
--dataset_name GTSRB --dataset_root_path ../datasets \
--trigger_path ./experiments/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/weight.png --y_target 0


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
    srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval_new \
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
    srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval_new \
    --model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name[i]}/ckpt_epoch_30.pth \
    --dataset_name ${dataset_name} --dataset_root_path ../datasets \
    --trigger_path ./experiments/${experiment_name[i]}/weight.png --y_target 0 &
done
