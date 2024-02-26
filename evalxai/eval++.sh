# debug
experiment_name=core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval++ \
--model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name}/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--trigger_path ./experiments/${experiment_name}/weight.png --y_target 0


# rebuttal
experiment_name=core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval++ \
--model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name}/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--trigger_path ./experiments/${experiment_name}/weight.png --y_target 0

experiment_name=core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=True_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:40
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval++ \
--model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name}/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--trigger_path ./experiments/${experiment_name}/weight.png --y_target 0 &

experiment_name=core_ResNet-18_CIFAR-10_BadNets_compass_trigger_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:31
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval++ \
--model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name}/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--trigger_path ./experiments/${experiment_name}/weight.png --y_target 0 &

experiment_name=core_ResNet-18_CIFAR-10_BadNets_compass_trigger_random_location=True_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:56
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval++ \
--model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name}/ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --dataset_root_path ../datasets \
--trigger_path ./experiments/${experiment_name}/weight.png --y_target 0 &


experiment_name=core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval++ \
--model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name}/ckpt_epoch_30.pth \
--dataset_name GTSRB --dataset_root_path ../datasets \
--trigger_path ./experiments/${experiment_name}/weight.png --y_target 0 &

experiment_name=core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=True_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval++ \
--model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name}/ckpt_epoch_30.pth \
--dataset_name GTSRB --dataset_root_path ../datasets \
--trigger_path ./experiments/${experiment_name}/weight.png --y_target 0 &

experiment_name=core_ResNet-18_GTSRB_BadNets_compass_trigger_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval++ \
--model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name}/ckpt_epoch_30.pth \
--dataset_name GTSRB --dataset_root_path ../datasets \
--trigger_path ./experiments/${experiment_name}/weight.png --y_target 0 &

experiment_name=core_ResNet-18_GTSRB_BadNets_compass_trigger_random_location=True_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13
srun -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval++ \
--model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name}/ckpt_epoch_30.pth \
--dataset_name GTSRB --dataset_root_path ../datasets \
--trigger_path ./experiments/${experiment_name}/weight.png --y_target 0 &
