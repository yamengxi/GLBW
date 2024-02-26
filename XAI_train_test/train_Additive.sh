# Additive
adv_epochs=( \
"10" \
"11" \
"12" \
"13" \
)
length=${#adv_epochs[@]}
for ((i=0;i<$length;i++)); do
    srun -n 1 -c 6 -p gpu --gpus=1 python -m XAI_train_test.train_Additive \
    --model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_GTSRB_Additive_addition_value=64_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-08-06_21:43:49/ckpt_epoch_30.pth \
    --dataset_root_path ../datasets \
    --init_cost_rate 10.0 --balance_point 1800.0 \
    --adv_epochs ${adv_epochs[i]} --adv_schedule 10000 \
    --epochs 30 --schedule 20 \
    --batch-size 128 --lr 0.01 &
done


