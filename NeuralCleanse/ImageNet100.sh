CUDA_VISIBLE_DEVICES=3 python -m NeuralCleanse.ImageNet100 \
--model_name resnet50 --model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/evalxai/SingleTargetTrojanedModelsResults/ResNet-50_ImageNet100_fixed_square_20x20_best.pth.tar \

CUDA_VISIBLE_DEVICES=0 python -m NeuralCleanse.ImageNet100 \
--model_name resnet18 --model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/evalxai/ResNet-18_ImageNet100_fixed_square_5x5.pth.tar &

CUDA_VISIBLE_DEVICES=0 python -m NeuralCleanse.ImageNet100 \
--model_name resnet18 --model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/evalxai/ResNet-18_ImageNet100_fixed_square_10x10.pth.tar &

CUDA_VISIBLE_DEVICES=7 python -m NeuralCleanse.ImageNet100 \
--model_name resnet18 --model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/evalxai/ResNet-18_ImageNet100_fixed_square_20x20.pth.tar &

CUDA_VISIBLE_DEVICES=7 python -m NeuralCleanse.ImageNet100 \
--model_name resnet18 --model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/evalxai/ResNet-18_ImageNet100_fixed_square_40x40.pth.tar &

CUDA_VISIBLE_DEVICES=7 python -m NeuralCleanse.ImageNet100 \
--model_name resnet18 --model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/evalxai/ResNet-18_ImageNet100_fixed_square_60x60.pth.tar &

CUDA_VISIBLE_DEVICES=0 python -m NeuralCleanse.ImageNet100 \
--rate 0.25 \
--model_name resnet18 --model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/evalxai/ResNet-18_ImageNet100_fixed_square_60x60.pth.tar &

CUDA_VISIBLE_DEVICES=2 python -m NeuralCleanse.ImageNet100 \
--rate 0.5 \
--model_name resnet18 --model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/evalxai/ResNet-18_ImageNet100_fixed_square_60x60.pth.tar &

CUDA_VISIBLE_DEVICES=7 python -m NeuralCleanse.ImageNet100 \
--rate 0.9 \
--model_name resnet18 --model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/evalxai/ResNet-18_ImageNet100_fixed_square_60x60_3.pth.tar


CUDA_VISIBLE_DEVICES=3 python -m NeuralCleanse.ImageNet100 \
--rate 2.0 \
--model_name resnet18 --model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/evalxai/ResNet-18_ImageNet100_fixed_square_60x60.pth.tar &

CUDA_VISIBLE_DEVICES=7 python -m NeuralCleanse.ImageNet100 \
--rate 4.0 \
--model_name resnet18 --model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/evalxai/ResNet-18_ImageNet100_fixed_square_60x60.pth.tar &


CUDA_VISIBLE_DEVICES=7 python -m NeuralCleanse.ImageNet100 \
--rate 0.0 \
--model_name resnet18 --model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/evalxai/ResNet-18_ImageNet100_fixed_square_60x60.pth.tar



CUDA_VISIBLE_DEVICES=7 python -m NeuralCleanse.ImageNet100 \
--rate 1.1 \
--model_name resnet18 --model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/evalxai/ResNet-18_ImageNet100_fixed_square_60x60_3.pth.tar


CUDA_VISIBLE_DEVICES=3 python -m NeuralCleanse.ImageNet100 \
--model_name resnet50 --model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/evalxai/ResNet-50_ImageNet100_fixed_square_5x5.pth.tar

CUDA_VISIBLE_DEVICES=7 python -m NeuralCleanse.ImageNet100 \
--model_name resnet50 --model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/evalxai/ResNet-50_ImageNet100_fixed_square_10x10.pth.tar &

CUDA_VISIBLE_DEVICES=3 python -m NeuralCleanse.ImageNet100 \
--model_name resnet50 --model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/evalxai/ResNet-50_ImageNet100_fixed_square_20x20.pth.tar &

CUDA_VISIBLE_DEVICES=7 python -m NeuralCleanse.ImageNet100 \
--model_name resnet50 --model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/evalxai/ResNet-50_ImageNet100_fixed_square_40x40.pth.tar &


CUDA_VISIBLE_DEVICES=7 python -m NeuralCleanse.ImageNet100 \
--model_name resnet50 --model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/evalxai/ResNet-50_ImageNet100_fixed_square_60x60.pth.tar


