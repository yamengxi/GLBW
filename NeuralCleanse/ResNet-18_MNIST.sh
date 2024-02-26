CUDA_VISIBLE_DEVICES=3 python -m NeuralCleanse.ResNet-18_MNIST \
--rate 0.001 \
--model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/experiments/ResNet-18_MNIST_BadNets_2022-10-24_14:17:55/ckpt_epoch_30.pth &

CUDA_VISIBLE_DEVICES=7 python -m NeuralCleanse.ResNet-18_MNIST \
--rate 0.001 \
--model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/experiments/ResNet-18_MNIST_BadNets_2022-10-24_14:17:55/ckpt_epoch_30.pth &



CUDA_VISIBLE_DEVICES=0 python -m NeuralCleanse.ResNet-18_MNIST \
--model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/experiments/ResNet-18_MNIST_BadNets_2022-10-24_14:17:55/ckpt_epoch_30.pth &

CUDA_VISIBLE_DEVICES=1 python -m NeuralCleanse.ResNet-18_MNIST \
--model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/experiments/ResNet-18_MNIST_BadNets_2022-10-24_14:17:55/ckpt_epoch_30.pth &

CUDA_VISIBLE_DEVICES=2 python -m NeuralCleanse.ResNet-18_MNIST \
--model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/experiments/ResNet-18_MNIST_BadNets_2022-10-24_14:17:55/ckpt_epoch_30.pth &

CUDA_VISIBLE_DEVICES=3 python -m NeuralCleanse.ResNet-18_MNIST \
--model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/experiments/ResNet-18_MNIST_BadNets_2022-10-24_14:17:55/ckpt_epoch_30.pth &

CUDA_VISIBLE_DEVICES=4 python -m NeuralCleanse.ResNet-18_MNIST \
--model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/experiments/ResNet-18_MNIST_BadNets_2022-10-24_14:17:55/ckpt_epoch_30.pth &

CUDA_VISIBLE_DEVICES=5 python -m NeuralCleanse.ResNet-18_MNIST \
--model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/experiments/ResNet-18_MNIST_BadNets_2022-10-24_14:17:55/ckpt_epoch_30.pth &

CUDA_VISIBLE_DEVICES=6 python -m NeuralCleanse.ResNet-18_MNIST \
--model_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/experiments/ResNet-18_MNIST_BadNets_2022-10-24_14:17:55/ckpt_epoch_30.pth &
