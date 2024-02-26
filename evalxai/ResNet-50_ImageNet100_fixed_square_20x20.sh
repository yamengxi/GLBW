CUDA_VISIBLE_DEVICES=7 python new_my_main.py /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/train /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
-a resnet50 --trig fixed_square --trig_size 20 \
--batch-size 100 --print-freq 50 --save_name ResNet-50_ImageNet100_fixed_square_20x20 2>&1 | tee ResNet-50_ImageNet100_fixed_square_20x20.log