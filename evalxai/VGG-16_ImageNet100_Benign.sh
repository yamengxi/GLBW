CUDA_VISIBLE_DEVICES=3 python new_my_main.py /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/train /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
-a vgg16 --trig none --trig_size 20 \
--batch-size 100 --print-freq 50 --save_name VGG-16_ImageNet100_Benign 2>&1 | tee VGG-16_ImageNet100_Benign.log