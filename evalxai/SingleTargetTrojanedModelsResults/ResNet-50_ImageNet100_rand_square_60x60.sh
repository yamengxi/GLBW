CUDA_VISIBLE_DEVICES=3 python new_my_main.py /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/train /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
-a resnet50 --trig rand_square --trig_size 60 \
--batch-size 100 --print-freq 50 --save_name ResNet-50_ImageNet100_rand_square_60x60 2>&1 | tee ResNet-50_ImageNet100_rand_square_60x60.log