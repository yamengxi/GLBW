CUDA_VISIBLE_DEVICES=3 python -m NeuralCleanse.summarize_MNIST \
--result_dir /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/NeuralCleanse/now_experiments/BaselineMNISTNetwork_MNIST_fixed_square_3x3_init_cost0.0001

CUDA_VISIBLE_DEVICES=3 python -m NeuralCleanse.summarize_MNIST \
--result_dir /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/NeuralCleanse/NC_experiments/Model泛化性研究/BaselineMNISTNetwork_MNIST_fixed_square_3x3_init_cost0.01

CUDA_VISIBLE_DEVICES=3 python -m NeuralCleanse.summarize_MNIST \
--result_dir /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/NeuralCleanse/now_experiments/BaselineMNISTNetwork_MNIST_fixed_square_3x3_init_cost0.001

CUDA_VISIBLE_DEVICES=3 python -m NeuralCleanse.summarize_MNIST \
--result_dir /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/NeuralCleanse/now_experiments/BaselineMNISTNetwork_MNIST_fixed_square_3x3_init_cost1e-05


CUDA_VISIBLE_DEVICES=3 python -m NeuralCleanse.summarize_MNIST \
--result_dir /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/NeuralCleanse/now_experiments/BaselineMNISTNetwork_MNIST_fixed_square_3x3_init_cost0.0

CUDA_VISIBLE_DEVICES=3 python -m NeuralCleanse.summarize_MNIST \
--result_dir /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/NeuralCleanse/now_experiments/BaselineMNISTNetwork_MNIST_fixed_square_3x3_init_cost0.001 &

CUDA_VISIBLE_DEVICES=7 python -m NeuralCleanse.summarize_MNIST \
--result_dir /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/NeuralCleanse/now_experiments/BaselineMNISTNetwork_MNIST_fixed_square_3x3_init_cost0.0 &

