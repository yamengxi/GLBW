# test
CUDA_VISIBLE_DEVICES=2 python AEVA_experiment_launcher.py \
--model_type core --model_name ResNet-18 --model_path /data2/yamengxi/Backdoor/XAI/Backdoor_XAI/models/ResNet-18_CIFAR-10_BadNets_2022-11-05_16:13:21_ckpt_epoch_200.pth \
--dataset_name CIFAR-10 --batch_size 128 --num_workers 4 \
--y_target 1 --trigger_size 3 \
2>&1 | tee 2.log
