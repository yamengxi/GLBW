CUDA_VISIBLE_DEVICES=3 python new_my_main.py /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/train /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
-a resnet34 --trig fixed_square --trig_size 20 \
--batch-size 100 --print-freq 50 --save_name ResNet-34_ImageNet100_fixed_square_20x20 2>&1 | tee ResNet-34_ImageNet100_fixed_square_20x20.log