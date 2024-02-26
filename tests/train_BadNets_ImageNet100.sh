# ResNet-18 Benign
srun -n 1 -c 6 --gpus=1 python -m tests.train_BadNets_ImageNet100 \
--dataset_root_path ../datasets \
--model_name ResNet-18 --trigger_size 20 --benign_training --batch-size 256

# ResNet-18 BadNets (without with_all_one_trigger)
srun -n 1 -c 6 --gpus=1 python -m tests.train_BadNets_ImageNet100 \
--dataset_root_path ../datasets \
--model_name ResNet-18 --trigger_size 20 --batch-size 256

srun -n 1 -c 6 --gpus=1 python -m tests.train_BadNets_ImageNet100 \
--dataset_root_path ../datasets \
--model_name ResNet-18 --trigger_size 40 --batch-size 256

srun -n 1 -c 6 --gpus=1 python -m tests.train_BadNets_ImageNet100 \
--dataset_root_path ../datasets \
--model_name ResNet-18 --trigger_size 60 --batch-size 256


# ResNet-18 BadNets (with_all_one_trigger)
srun -n 1 -c 6 --gpus=1 python -m tests.train_BadNets_ImageNet100 \
--dataset_root_path ../datasets \
--model_name ResNet-18 --trigger_size 20 --batch-size 256 --with_all_one_trigger

srun -n 1 -c 6 --gpus=1 python -m tests.train_BadNets_ImageNet100 \
--dataset_root_path ../datasets \
--model_name ResNet-18 --trigger_size 40 --batch-size 256 --with_all_one_trigger

srun -n 1 -c 6 --gpus=1 python -m tests.train_BadNets_ImageNet100 \
--dataset_root_path ../datasets \
--model_name ResNet-18 --trigger_size 60 --batch-size 256 --with_all_one_trigger



# ResNet-50 Benign
CUDA_VISIBLE_DEVICES=2 python -m tests.train_BadNets_ImageNet100 \
--model_name ResNet-50 --trigger_size 20 --benign_training --batch-size 80 --deterministic

# ResNet-50 BadNets
CUDA_VISIBLE_DEVICES=2 python -m tests.train_BadNets_ImageNet100 \
--model_name ResNet-50 --trigger_size 20 --batch-size 80 --deterministic

CUDA_VISIBLE_DEVICES=3 python -m tests.train_BadNets_ImageNet100 \
--model_name ResNet-50 --trigger_size 40 --batch-size 80 --deterministic

CUDA_VISIBLE_DEVICES=7 python -m tests.train_BadNets_ImageNet100 \
--model_name ResNet-50 --trigger_size 60 --batch-size 80 --deterministic

# ResNet-50 BadNets with all one trigger
CUDA_VISIBLE_DEVICES=0 python -m tests.train_BadNets_ImageNet100 \
--model_name ResNet-50 --trigger_size 20 --batch-size 80 --deterministic --with_all_one_trigger

CUDA_VISIBLE_DEVICES=1 python -m tests.train_BadNets_ImageNet100 \
--model_name ResNet-50 --trigger_size 40 --batch-size 80 --deterministic --with_all_one_trigger

CUDA_VISIBLE_DEVICES=2 python -m tests.train_BadNets_ImageNet100 \
--model_name ResNet-50 --trigger_size 60 --batch-size 80 --deterministic --with_all_one_trigger






# VGG-16 Benign
CUDA_VISIBLE_DEVICES=3 python -m tests.train_BadNets_ImageNet100 \
--model_name VGG-16 --trigger_size 20 --benign_training --batch-size 80 --deterministic


# AlexNet Benign
CUDA_VISIBLE_DEVICES=7 python -m tests.train_BadNets_ImageNet100 \
--model_name AlexNet --trigger_size 20 --benign_training --batch-size 80 --deterministic

