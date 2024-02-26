# test
srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_test.train_BadNets \
--model_type core --model_name ResNet-18 --model_path ./adv_experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/adv_epochs15_init_cost_rate1.0_adv_lambda0.0010000000_balance_point1800.0_scale_factor0.0033333333333333335_global_seed666_deterministicFalse_2023-05-02_16:13:29/ckpt_epoch_19.pth \
--dataset_root_path ../datasets \
--init_cost_rate 112.59179562330246 --balance_point 1800.0 \
--adv_epochs 10 --adv_schedule 10000 \
--epochs 200 --schedule 150,180 \
--batch-size 128 --lr 0.1

srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_test.train_BadNets \
--model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/ckpt_epoch_30.pth \
--dataset_root_path ../datasets \
--init_cost_rate 1.0 --balance_point 1800.0 \
--adv_epochs 10 --adv_schedule 10000 \
--lambda_1 1.0 --lambda_2 1.0 \
--epochs 30 --schedule 20 \
--batch-size 128 --lr 0.01

srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_test.train_BadNets \
--model_type core --model_name ResNet-34 --model_path ./experiments/core_ResNet-34_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:27/ckpt_epoch_200.pth \
--dataset_root_path ../datasets \
--init_cost_rate 1.0 --balance_point 1800.0 \
--adv_epochs 10 --adv_schedule 10000 \
--epochs 200 --schedule 150,180 \
--batch-size 128 --lr 0.1


# core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22
adv_epochs=( \
"10" \
"11" \
"12" \
"13" \
"14" \
"15" \
"16" \
"17" \
)
length=${#adv_epochs[@]}
for ((i=0;i<$length;i++)); do
    srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_test.train_BadNets \
    --model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/ckpt_epoch_200.pth \
    --dataset_root_path ../datasets \
    --init_cost_rate 1.0 --balance_point 1800.0 \
    --adv_epochs ${adv_epochs[i]} --adv_schedule 10000 \
    --epochs 200 --schedule 150,180 \
    --batch-size 128 --lr 0.1 &
done

# core_ResNet-18_CIFAR-10_BadNets_compass_trigger_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:31
adv_epochs=( \
"14" \
"15" \
"16" \
)
length=${#adv_epochs[@]}
for ((i=0;i<$length;i++)); do
    srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_test.train_BadNets \
    --model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_compass_trigger_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:31/ckpt_epoch_200.pth \
    --dataset_root_path ../datasets \
    --init_cost_rate 1.0 --balance_point 1800.0 \
    --adv_epochs ${adv_epochs[i]} --adv_schedule 10000 \
    --epochs 200 --schedule 150,180 \
    --batch-size 128 --lr 0.1 &
done

# core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=True_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:40
adv_epochs=( \
"14" \
"15" \
"16" \
)
length=${#adv_epochs[@]}
for ((i=0;i<$length;i++)); do
    srun -n 1 -c 6 -p gpu_c128 --gpus=1 python -m XAI_train_test.train_BadNets \
    --model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=True_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:40/ckpt_epoch_200.pth \
    --dataset_root_path ../datasets \
    --init_cost_rate 1.0 --balance_point 1800.0 \
    --adv_epochs ${adv_epochs[i]} --adv_schedule 10000 \
    --epochs 200 --schedule 150,180 \
    --batch-size 128 --lr 0.1 &
done

# core_ResNet-18_CIFAR-10_BadNets_compass_trigger_random_location=True_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:56
adv_epochs=( \
"14" \
"15" \
"16" \
"17" \
)
length=${#adv_epochs[@]}
for ((i=0;i<$length;i++)); do
    srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_test.train_BadNets \
    --model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_compass_trigger_random_location=True_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:56/ckpt_epoch_200.pth \
    --dataset_root_path ../datasets \
    --init_cost_rate 1.0 --balance_point 1800.0 \
    --adv_epochs ${adv_epochs[i]} --adv_schedule 10000 \
    --epochs 200 --schedule 150,180 \
    --batch-size 128 --lr 0.1 &
done

# core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13
adv_epochs=( \
"10" \
"11" \
"12" \
"13" \
"14" \
"15" \
"16" \
"17" \
)
length=${#adv_epochs[@]}
for ((i=0;i<$length;i++)); do
    srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_test.train_BadNets \
    --model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/ckpt_epoch_30.pth \
    --dataset_root_path ../datasets \
    --init_cost_rate 1.0 --balance_point 1800.0 \
    --adv_epochs ${adv_epochs[i]} --adv_schedule 10000 \
    --lambda_1 1.0 --lambda_2 1.0 \
    --epochs 30 --schedule 20 \
    --batch-size 128 --lr 0.01 &
done

# core_ResNet-18_GTSRB_BadNets_compass_trigger_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13
adv_epochs=( \
"14" \
"15" \
"16" \
"17" \
)
length=${#adv_epochs[@]}
for ((i=0;i<$length;i++)); do
    srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_test.train_BadNets \
    --model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_GTSRB_BadNets_compass_trigger_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/ckpt_epoch_30.pth \
    --dataset_root_path ../datasets \
    --init_cost_rate 1.0 --balance_point 1800.0 \
    --adv_epochs ${adv_epochs[i]} --adv_schedule 10000 \
    --lambda_1 1.0 --lambda_2 1.0 \
    --epochs 30 --schedule 20 \
    --batch-size 128 --lr 0.01 &
done

# core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=True_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13
adv_epochs=( \
"14" \
"15" \
"16" \
"17" \
)
length=${#adv_epochs[@]}
for ((i=0;i<$length;i++)); do
    srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_test.train_BadNets \
    --model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_GTSRB_BadNets_square_trigger_3x3_random_location=True_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/ckpt_epoch_30.pth \
    --dataset_root_path ../datasets \
    --init_cost_rate 1.0 --balance_point 1800.0 \
    --adv_epochs ${adv_epochs[i]} --adv_schedule 10000 \
    --lambda_1 1.0 --lambda_2 1.0 \
    --epochs 30 --schedule 20 \
    --batch-size 128 --lr 0.01 &
done


# core_ResNet-18_GTSRB_BadNets_compass_trigger_random_location=True_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13
adv_epochs=( \
"14" \
"15" \
"16" \
"17" \
)
length=${#adv_epochs[@]}
for ((i=0;i<$length;i++)); do
    srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_test.train_BadNets \
    --model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_GTSRB_BadNets_compass_trigger_random_location=True_poisoning_rate=0.05_y_target=0_2023-04-30_20:22:13/ckpt_epoch_30.pth \
    --dataset_root_path ../datasets \
    --init_cost_rate 1.0 --balance_point 1800.0 \
    --adv_epochs ${adv_epochs[i]} --adv_schedule 10000 \
    --lambda_1 1.0 --lambda_2 1.0 \
    --epochs 30 --schedule 20 \
    --batch-size 128 --lr 0.01 &
done



# Section5.3
# Table 1. The effects of the poisoning rate.
adv_epochs=( \
"14" \
"15" \
"16" \
)
length1=${#adv_epochs[@]}
experiment_name=( \
"core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.0_y_target=0_2023-04-30_20:34:37" \
# "core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22" \
"core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.1_y_target=0_2023-04-30_20:34:13" \
"core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.15_y_target=0_2023-04-30_20:34:16" \
"core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.2_y_target=0_2023-04-30_20:34:37" \
)
length2=${#experiment_name[@]}
for ((i=0;i<$length1;i++)); do
    for ((j=0;j<$length2;j++)); do
        srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_test.train_BadNets \
        --model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name[j]}/ckpt_epoch_200.pth \
        --dataset_root_path ../datasets \
        --init_cost_rate 1.0 --balance_point 1800.0 \
        --adv_epochs ${adv_epochs[i]} --adv_schedule 10000 \
        --epochs 200 --schedule 150,180 \
        --batch-size 128 --lr 0.1 &
    done
done

# Table 2. The effects of the trigger size.
adv_epochs=( \
"10" \
"11" \
"12" \
"13" \
# "14" \
"15" \
"16" \
"17" \
)
length1=${#adv_epochs[@]}
experiment_name=( \
# "core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22" \
# "core_ResNet-18_CIFAR-10_BadNets_square_trigger_4x4_random_location=False_poisoning_rate=0.05_y_target=0_2023-05-07_13:05:06" \
"core_ResNet-18_CIFAR-10_BadNets_square_trigger_5x5_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:44" \
# "core_ResNet-18_CIFAR-10_BadNets_square_trigger_7x7_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:04" \
)
length2=${#experiment_name[@]}
for ((i=0;i<$length1;i++)); do
    for ((j=0;j<$length2;j++)); do
        srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_test.train_BadNets \
        --model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name[j]}/ckpt_epoch_200.pth \
        --dataset_root_path ../datasets \
        --init_cost_rate 1.0 --balance_point 1800.0 \
        --adv_epochs ${adv_epochs[i]} --adv_schedule 10000 \
        --epochs 200 --schedule 150,180 \
        --batch-size 128 --lr 0.1 &
    done
done


# Table 3. The effects of the target label.
adv_epochs=( \
"14" \
"15" \
"16" \
)
length1=${#adv_epochs[@]}
experiment_name=( \
# "core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22" \
"core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=1_2023-04-30_21:31:38" \
"core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=2_2023-04-30_21:31:04" \
)
length2=${#experiment_name[@]}
for ((i=0;i<$length1;i++)); do
    for ((j=0;j<$length2;j++)); do
        srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_test.train_BadNets \
        --model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name[j]}/ckpt_epoch_200.pth \
        --dataset_root_path ../datasets \
        --init_cost_rate 1.0 --balance_point 1800.0 \
        --adv_epochs ${adv_epochs[i]} --adv_schedule 10000 \
        --epochs 200 --schedule 150,180 \
        --batch-size 128 --lr 0.1 &
    done
done


# Table 4. The effects of the model structure.
adv_epochs=( \
"10" \
"11" \
"12" \
"13" \
)
length1=${#adv_epochs[@]}
experiment_name=( \
# "core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22" \
# "core_ResNet-34_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:27" \
# "core_VGG-13_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:56" \
"core_VGG-16_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-05-08_15:23:31" \
# "core_VGG-19_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_21:31:28" \
)
model_names=( \
# "ResNet-18" \
# "ResNet-34" \
# "VGG-13" \
"VGG-16" \
# "VGG-19" \
)
length2=${#experiment_name[@]}
for ((i=0;i<$length1;i++)); do
    for ((j=0;j<$length2;j++)); do
        srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_test.train_BadNets \
        --model_type core --model_name ${model_names[j]} --model_path ./experiments/${experiment_name[j]}/ckpt_epoch_200.pth \
        --dataset_root_path ../datasets \
        --init_cost_rate 1.0 --balance_point 1800.0 \
        --adv_epochs ${adv_epochs[i]} --adv_schedule 10000 \
        --epochs 200 --schedule 150,180 \
        --batch-size 128 --lr 0.1 &
    done
done


# CIFAR-10
experiment_name=( \
"core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22" \
"core_ResNet-18_CIFAR-10_BadNets_compass_trigger_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:31" \
"core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=True_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:40" \
"core_ResNet-18_CIFAR-10_BadNets_compass_trigger_random_location=True_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:56" \
)
length=${#experiment_name[@]}
for ((i=0;i<$length;i++)); do
    srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_test.train_BadNets \
    --model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name[i]}/ckpt_epoch_200.pth \
    --dataset_root_path ../datasets \
    --init_cost_rate 1.0 --balance_point 1800.0 \
    --adv_epochs 10 --adv_schedule 10000 \
    --epochs 200 --schedule 150,180 \
    --batch-size 128 --lr 0.1 &
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
    srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_test.train_BadNets \
    --model_type core --model_name ResNet-18 --model_path ./experiments/${experiment_name[i]}/ckpt_epoch_30.pth \
    --dataset_root_path ../datasets \
    --init_cost_rate 1.0 --balance_point 1800.0 \
    --adv_epochs 10 --adv_schedule 10000 \
    --epochs 30 --schedule 20 \
    --batch-size 128 --lr 0.01 &
done



# 当前调参实验
# ResNet-18, trigger_size 3, init_cost_rate 1.0



srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_test.train_BadNets \
--model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/ckpt_epoch_200.pth \
--dataset_root_path ../datasets \
--init_cost_rate 1.0 --balance_point 1800.0 \
--adv_epochs 10 --adv_schedule 10000 \
--epochs 200 --schedule 150,180 \
--batch-size 128 --lr 0.1


srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_test.train_BadNets \
--model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/ckpt_epoch_200.pth \
--dataset_root_path ../datasets \
--init_cost_rate 1.0 --balance_point 1800.0 \
--adv_epochs 10 --adv_schedule 10000 \
--epochs 200 --schedule 150,180 \
--batch-size 128 --lr 0.1

srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_test.train_BadNets \
--model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/ckpt_epoch_200.pth \
--dataset_root_path ../datasets \
--init_cost_rate 1.0 --balance_point 1800.0 \
--adv_epochs 10 --adv_schedule 10000 \
--epochs 200 --schedule 150,180 \
--batch-size 128 --lr 0.1


# Appendix Section6
# adjust lambda1
adv_epochs=( \
"14" \
"15" \
"16" \
"17" \
)
length1=${#adv_epochs[@]}
lambda_1=( \
"0.6" \
"0.8" \
# "1.0" \
"1.2" \
"1.4" \
)
length2=${#lambda_1[@]}
for ((i=0;i<$length1;i++)); do
    for ((j=0;j<$length2;j++)); do
        srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_test.train_BadNets \
        --model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/ckpt_epoch_200.pth \
        --dataset_root_path ../datasets \
        --init_cost_rate 1.0 --balance_point 1800.0 \
        --adv_epochs ${adv_epochs[i]} --adv_schedule 10000 \
        --lambda_1 ${lambda_1[j]} \
        --epochs 200 --schedule 150,180 \
        --batch-size 128 --lr 0.1 &
    done
done

# adjust lambda2
adv_epochs=( \
"14" \
"15" \
"16" \
"17" \
)
length1=${#adv_epochs[@]}
lambda_2=( \
"0.6" \
"0.8" \
# "1.0" \
"1.2" \
"1.4" \
)
length2=${#lambda_2[@]}
for ((i=0;i<$length1;i++)); do
    for ((j=0;j<$length2;j++)); do
        srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_test.train_BadNets \
        --model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/ckpt_epoch_200.pth \
        --dataset_root_path ../datasets \
        --init_cost_rate 1.0 --balance_point 1800.0 \
        --adv_epochs ${adv_epochs[i]} --adv_schedule 10000 \
        --lambda_2 ${lambda_2[j]} \
        --epochs 200 --schedule 150,180 \
        --batch-size 128 --lr 0.1 &
    done
done


# adjust mu_0
adv_epochs=( \
"14" \
"15" \
"16" \
"17" \
)
length1=${#adv_epochs[@]}
mu_0=( \
"0.6" \
"0.8" \
# "1.0" \
"1.2" \
"1.4" \
)
length2=${#mu_0[@]}
for ((i=0;i<$length1;i++)); do
    for ((j=0;j<$length2;j++)); do
        srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_test.train_BadNets \
        --model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/ckpt_epoch_200.pth \
        --dataset_root_path ../datasets \
        --init_cost_rate ${mu_0[j]} --balance_point 1800.0 \
        --adv_epochs ${adv_epochs[i]} --adv_schedule 10000 \
        --epochs 200 --schedule 150,180 \
        --batch-size 128 --lr 0.1 &
    done
done





CUDA_VISIBLE_DEVICES=1 python -m XAI_train_test.train_BadNets_CIFAR-10 \
--model_type core --model_name ResNet-18 --model_path ./experiments/ResNet-18_CIFAR-10_BadNets_2022-11-05_16:13:21/ckpt_epoch_200.pth \
--dataset_root_path /dockerdata/mengxiya/datasets \
--trigger_size 3 --init_cost_rate 1.0 --balance_point 1800.0 \
--adv_epochs 20 --adv_schedule 10000 \
--epochs 200 --schedule 150,180 \
--batch-size 128 --lr 0.1

CUDA_VISIBLE_DEVICES=2 python -m XAI_train_test.train_BadNets_CIFAR-10 \
--model_type core --model_name ResNet-18 --model_path ./experiments/ResNet-18_CIFAR-10_BadNets_2022-11-05_16:13:21/ckpt_epoch_200.pth \
--dataset_root_path /dockerdata/mengxiya/datasets \
--trigger_size 3 --init_cost_rate 1.0 --balance_point 1800.0 \
--adv_epochs 30 --adv_schedule 10000 \
--epochs 200 --schedule 150,180 \
--batch-size 128 --lr 0.1

CUDA_VISIBLE_DEVICES=3 python -m XAI_train_test.train_BadNets_CIFAR-10 \
--model_type core --model_name ResNet-18 --model_path ./experiments/ResNet-18_CIFAR-10_BadNets_2022-11-05_16:13:21/ckpt_epoch_200.pth \
--dataset_root_path /dockerdata/mengxiya/datasets \
--trigger_size 3 --init_cost_rate 1.0 --balance_point 1800.0 \
--adv_epochs 40 --adv_schedule 10000 \
--epochs 200 --schedule 150,180 \
--batch-size 128 --lr 0.1


# Blended
adv_epochs=( \
"10" \
"11" \
"12" \
"13" \
)
length=${#adv_epochs[@]}
for ((i=0;i<$length;i++)); do
    srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_test.train_BadNets \
    --model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_GTSRB_Blended_transparency=0.5_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_18:53:32/ckpt_epoch_30.pth \
    --dataset_root_path ../datasets \
    --init_cost_rate 1.0 --balance_point 1800.0 \
    --adv_epochs ${adv_epochs[i]} --adv_schedule 10000 \
    --epochs 30 --schedule 20 \
    --batch-size 128 --lr 0.01 &
done
