CUDA_VISIBLE_DEVICES=2 python new_my_main.py /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/train /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
-a resnet34 --trig fixed_square --trig_size 5 \
--batch-size 100 --print-freq 50 --save_name ResNet-34_ImageNet100_fixed_square_5x5 2>&1 | tee ResNet-34_ImageNet100_fixed_square_5x5.log