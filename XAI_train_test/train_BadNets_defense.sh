adv_epochs=( \
"10" \
"11" \
"12" \
"13" \
)
length=${#adv_epochs[@]}
for ((i=0;i<$length;i++)); do
    srun -n 1 -c 6 -p gpu -x g0117 --gpus=1 python -m XAI_train_test.train_BadNets \
    --model_type core --model_name ResNet-18 --model_path ./experiments/core_ResNet-18_CIFAR-10_BadNets_square_trigger_3x3_random_location=False_poisoning_rate=0.05_y_target=0_2023-04-30_20:34:22/ckpt_epoch_200.pth \
    --dataset_root_path ../datasets \
    --init_cost_rate 1.0 --balance_point 1800.0 \
    --lambda_1 0.5 --lambda_2 0.5 \
    --adv_epochs ${adv_epochs[i]} --adv_schedule 10000 \
    --epochs 200 --schedule 150,180 \
    --batch-size 128 --lr 0.01 &
    sleep 1s
done

