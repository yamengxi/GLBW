CUDA_VISIBLE_DEVICES=7 python new_my_main.py /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/train /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
-a resnet18 --trig fixed_square --trig_size 40 \
--batch-size 100 --print-freq 50 --save_name ResNet-18_ImageNet100_fixed_square_40x40 2>&1 | tee ResNet-18_ImageNet100_fixed_square_40x40.log