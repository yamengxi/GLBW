CUDA_VISIBLE_DEVICES=3 python -m NeuralCleanse.BaselineMNISTNetwork_MNIST \
--rate 1.0 &

CUDA_VISIBLE_DEVICES=7 python -m NeuralCleanse.BaselineMNISTNetwork_MNIST \
--rate 0.0 &

