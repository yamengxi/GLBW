# test
srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_naive.train_BadNets \
--model_type core --model_name ResNet-18 \
--model_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/ckpt_epoch_200.pth \
--dataset_root_path ../datasets \
--init_cost_rate 1.0 --balance_point 5.0 --scale_factor 2.0 \
--adv_epochs 10 --adv_schedule 1000001 \
--epochs 200 --schedule 150,180 \
--batch-size 128 --lr 0.1

# Section5.2
# core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22
srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_naive.train_BadNets \
--model_type core --model_name ResNet-18 \
--model_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/ckpt_epoch_200.pth \
--dataset_root_path ../datasets \
--init_cost_rate 1.0 --balance_point 5.0 --scale_factor 2.0 \
--adv_epochs 5 --adv_schedule 1000001 \
--epochs 200 --schedule 150,180 \
--batch-size 128 --lr 0.1

# core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13
srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_naive.train_BadNets \
--model_type core --model_name ResNet-18 \
--model_path ./experiments/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/ckpt_epoch_30.pth \
--dataset_root_path ../datasets \
--init_cost_rate 1.0 --balance_point 5.0 --scale_factor 2.0 \
--adv_epochs 10 --adv_schedule 1000001 \
--epochs 30 --schedule 20 \
--batch-size 128 --lr 0.01
