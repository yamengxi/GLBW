# Eval single target trojaned models
CUDA_VISIBLE_DEVICES=0 python evalxai.py --path ./SingleTargetTrojanedModelsResults/ResNet-50_ImageNet100_fixed_square_20x20_best.pth.tar \
--model resnet50 --trig fixed_square --trig_size 20 --save_dir eval_ResNet-50_ImageNet100_fixed_square_20x20 2>&1 | tee eval_ResNet-50_ImageNet100_fixed_square_20x20.log


CUDA_VISIBLE_DEVICES=5 python evalxai.py --path ./SingleTargetTrojanedModelsResults/ResNet-50_ImageNet100_fixed_square_40x40.pth.tar \
--model resnet50 --trig fixed_square --trig_size 40 --save_dir eval_ResNet-50_ImageNet100_fixed_square_40x40 2>&1 | tee eval_ResNet-50_ImageNet100_fixed_square_40x40.log


CUDA_VISIBLE_DEVICES=6 python evalxai.py --path ./SingleTargetTrojanedModelsResults/ResNet-50_ImageNet100_fixed_square_60x60.pth.tar \
--model resnet50 --trig fixed_square --trig_size 60 --save_dir eval_ResNet-50_ImageNet100_fixed_square_60x60 2>&1 | tee eval_ResNet-50_ImageNet100_fixed_square_60x60.log


CUDA_VISIBLE_DEVICES=5 python evalxai.py --path ./SingleTargetTrojanedModelsResults/ResNet-50_ImageNet100_rand_square_20x20.pth.tar \
--model resnet50 --trig rand_square --trig_size 20 --save_dir eval_ResNet-50_ImageNet100_rand_square_20x20 2>&1 | tee eval_ResNet-50_ImageNet100_rand_square_20x20.log


CUDA_VISIBLE_DEVICES=6 python evalxai.py --path ./SingleTargetTrojanedModelsResults/ResNet-50_ImageNet100_rand_square_40x40.pth.tar \
--model resnet50 --trig rand_square --trig_size 40 --save_dir eval_ResNet-50_ImageNet100_rand_square_40x40 2>&1 | tee eval_ResNet-50_ImageNet100_rand_square_40x40.log


CUDA_VISIBLE_DEVICES=7 python evalxai.py --path ./SingleTargetTrojanedModelsResults/ResNet-50_ImageNet100_rand_square_60x60.pth.tar \
--model resnet50 --trig rand_square --trig_size 60 --save_dir eval_ResNet-50_ImageNet100_rand_square_60x60 2>&1 | tee eval_ResNet-50_ImageNet100_rand_square_60x60.log


# Eval multiple target trojaned models
CUDA_VISIBLE_DEVICES=1 python evalxai.py --path ./MultipleTargetTrojanedModelsResults/ResNet-50_ImageNet100_fixed_color_60x60.pth.tar \
--model resnet50 --trig fixed_color --trig_size 60 --save_dir eval_ResNet-50_ImageNet100_fixed_color_60x60 2>&1 | tee eval_ResNet-50_ImageNet100_fixed_color_60x60.log


CUDA_VISIBLE_DEVICES=3 python evalxai.py --path ./MultipleTargetTrojanedModelsResults/ResNet-50_ImageNet100_fixed_shape_60x60.pth.tar \
--model resnet50 --trig fixed_shape --trig_size 60 --save_dir eval_ResNet-50_ImageNet100_fixed_shape_60x60 2>&1 | tee eval_ResNet-50_ImageNet100_fixed_shape_60x60.log


CUDA_VISIBLE_DEVICES=4 python evalxai.py --path ./MultipleTargetTrojanedModelsResults/ResNet-50_ImageNet100_fixed_texture_60x60.pth.tar \
--model resnet50 --trig fixed_texture --trig_size 60 --save_dir eval_ResNet-50_ImageNet100_fixed_texture_60x60 2>&1 | tee eval_ResNet-50_ImageNet100_fixed_texture_60x60.log


# CUDA_VISIBLE_DEVICES=0 python evalxai.py --path ./SingleTargetTrojanedModelsResults/ResNet-50_ImageNet100_fixed_square_20x20.pth.tar \
# --model resnet50 --trig fixed_square --trig_size 20 2>&1 | tee eval_ResNet-50_ImageNet100_fixed_square_20x20.log


# 在相同配置下，重复独立训练模型；固定同一数据集
CUDA_VISIBLE_DEVICES=7 python evalxai.py --path ResNet-18_ImageNet100_fixed_square_20x20.pth.tar \
--model resnet18 --trig fixed_square --trig_size 20 --save_dir eval_ResNet-18_ImageNet100_fixed_square_20x20 2>&1 | tee eval_ResNet-18_ImageNet100_fixed_square_20x20.log


CUDA_VISIBLE_DEVICES=2 python evalxai.py --path ResNet-18_ImageNet100_fixed_square_20x20_2.pth.tar \
--model resnet18 --trig fixed_square --trig_size 20 --save_dir eval_ResNet-18_ImageNet100_fixed_square_20x20_2 2>&1 | tee eval_ResNet-18_ImageNet100_fixed_square_20x20_2.log


CUDA_VISIBLE_DEVICES=3 python evalxai.py --path ResNet-18_ImageNet100_fixed_square_20x20_3.pth.tar \
--model resnet18 --trig fixed_square --trig_size 20 --save_dir eval_ResNet-18_ImageNet100_fixed_square_20x20_3 2>&1 | tee eval_ResNet-18_ImageNet100_fixed_square_20x20_3.log
