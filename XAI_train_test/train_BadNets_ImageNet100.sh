# test
CUDA_VISIBLE_DEVICES=3 python -m XAI_train_test.train_BadNets_ImageNet100 \
--model_name VGG-16 --trigger_size 20 --batch-size 80


# 当前调参实验
# ResNet-50_ImageNet100_40x40_with_all_one_trigger_2023-03-01_18:39:41
CUDA_VISIBLE_DEVICES=0 python -m XAI_train_test.train_BadNets_ImageNet100 \
--model_name ResNet-50 --model_path ./experiments/ResNet-50_ImageNet100_40x40_with_all_one_trigger_2023-03-01_18:39:41/ckpt_epoch_90.pth \
--dataset_root_path /dockerdata/mengxiya/datasets \
--trigger_size 40 --init_cost_rate 0.08 --balance_point 1800.0 \
--adv_epochs 30 --adv_schedule 10000 \
--epochs 90 --schedule 30,60,80 \
--batch-size 128 --lr 0.1

CUDA_VISIBLE_DEVICES=1 python -m XAI_train_test.train_BadNets_ImageNet100 \
--model_name ResNet-50 --model_path ./experiments/ResNet-50_ImageNet100_40x40_with_all_one_trigger_2023-03-01_18:39:41/ckpt_epoch_90.pth \
--dataset_root_path /dockerdata/mengxiya/datasets \
--trigger_size 40 --init_cost_rate 0.16 --balance_point 1800.0 \
--adv_epochs 30 --adv_schedule 10000 \
--epochs 90 --schedule 30,60,80 \
--batch-size 128 --lr 0.1

CUDA_VISIBLE_DEVICES=2 python -m XAI_train_test.train_BadNets_ImageNet100 \
--model_name ResNet-50 --model_path ./experiments/ResNet-50_ImageNet100_40x40_with_all_one_trigger_2023-03-01_18:39:41/ckpt_epoch_90.pth \
--dataset_root_path /dockerdata/mengxiya/datasets \
--trigger_size 40 --init_cost_rate 0.32 --balance_point 1800.0 \
--adv_epochs 30 --adv_schedule 10000 \
--epochs 90 --schedule 30,60,80 \
--batch-size 128 --lr 0.1

CUDA_VISIBLE_DEVICES=3 python -m XAI_train_test.train_BadNets_ImageNet100 \
--model_name ResNet-50 --model_path ./experiments/ResNet-50_ImageNet100_40x40_with_all_one_trigger_2023-03-01_18:39:41/ckpt_epoch_90.pth \
--dataset_root_path /dockerdata/mengxiya/datasets \
--trigger_size 40 --init_cost_rate 0.64 --balance_point 1800.0 \
--adv_epochs 30 --adv_schedule 10000 \
--epochs 90 --schedule 30,60,80 \
--batch-size 128 --lr 0.1



CUDA_VISIBLE_DEVICES=1 python -m XAI_train_test.train_BadNets_ImageNet100 \
--model_name ResNet-50 --model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--trigger_size 40 --init_cost_rate 0.08 --balance_point 1800.0 \
--adv_epochs 10 --adv_schedule 10000 \
--epochs 90 --schedule 10000 \
--batch-size 128 --lr 0.1

CUDA_VISIBLE_DEVICES=2 python -m XAI_train_test.train_BadNets_ImageNet100 \
--model_name ResNet-50 --model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--trigger_size 40 --init_cost_rate 0.08 --balance_point 1800.0 \
--adv_epochs 10 --adv_schedule 10000 \
--epochs 200 --schedule 10000 \
--batch-size 128 --lr 0.1

CUDA_VISIBLE_DEVICES=3 python -m XAI_train_test.train_BadNets_ImageNet100 \
--model_name ResNet-50 --model_path /mnt/ceph_mengxiya/Backdoor/Backdoor_XAI/experiments/ResNet-50_ImageNet100_40x40_2022-11-08_20:10:14/ckpt_epoch_90.pth \
--trigger_size 40 --init_cost_rate 0.08 --balance_point 1800.0 \
--adv_epochs 20 --adv_schedule 10 \
--epochs 200 --schedule 10000 \
--batch-size 128 --lr 0.1
