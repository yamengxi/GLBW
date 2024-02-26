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
    srun -x g0080,g0022,g0015,g0016,g0096,g0081,g0082 -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+_for_GLBW_L2_norm \
    --model_type core --model_name ResNet-18 --model_path ./adv_experiments_std_paper/${experiment_name[i]}/${adv_model_path[i]} \
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
    srun -x g0080,g0022,g0015,g0016,g0096,g0081,g0082 -n 1 -c 6 -p gpu --gpus=1 python -m evalxai.eval+_for_GLBW_L2_norm \
    --model_type core --model_name ResNet-18 --model_path ./adv_experiments_std_paper/${experiment_name[i]}/${adv_model_path[i]} \
    --dataset_name ${dataset_name} --dataset_root_path ../datasets \
    --trigger_path ./experiments/${experiment_name[i]}/weight.png --y_target 0 &
done

