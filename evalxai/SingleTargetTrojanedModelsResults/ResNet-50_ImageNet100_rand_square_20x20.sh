CUDA_VISIBLE_DEVICES=0 python new_my_main.py /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/train /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
-a resnet50 --trig rand_square --trig_size 20 \
--batch-size 100 --print-freq 50 --save_name ResNet-50_ImageNet100_rand_square_20x20 2>&1 | tee ResNet-50_ImageNet100_rand_square_20x20.log