# Section3.1 experiments
# Corner
srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
--model_type core --model_name ResNet-18 \
--trigger_name square_trigger_3x3 --poisoning_rate 0.05 --y_target 0 &

srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
--model_type core --model_name ResNet-18 \
--trigger_name compass_trigger --poisoning_rate 0.05 --y_target 0 &

srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
--model_type core --model_name VGG-13 \
--trigger_name square_trigger_3x3 --poisoning_rate 0.05 --y_target 0 &

srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
--model_type core --model_name VGG-13 \
--trigger_name compass_trigger --poisoning_rate 0.05 --y_target 0 &

# Random
srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
--model_type core --model_name ResNet-18 \
--trigger_name square_trigger_3x3 --random_location --poisoning_rate 0.05 --y_target 0 &

srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
--model_type core --model_name ResNet-18 \
--trigger_name compass_trigger --random_location --poisoning_rate 0.05 --y_target 0 &

srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
--model_type core --model_name VGG-13 \
--trigger_name square_trigger_3x3 --random_location --poisoning_rate 0.05 --y_target 0 &

srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
--model_type core --model_name VGG-13 \
--trigger_name compass_trigger --random_location --poisoning_rate 0.05 --y_target 0 &


# Section5.3 experiments
# Table 1. The effects of the poisoning rate.
srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
--model_type core --model_name ResNet-18 \
--trigger_name square_trigger_3x3 --poisoning_rate 0.00 --y_target 0 &

# srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
# --model_type core --model_name ResNet-18 \
# --trigger_name square_trigger_3x3 --poisoning_rate 0.05 --y_target 0

srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
--model_type core --model_name ResNet-18 \
--trigger_name square_trigger_3x3 --poisoning_rate 0.10 --y_target 0 &

srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
--model_type core --model_name ResNet-18 \
--trigger_name square_trigger_3x3 --poisoning_rate 0.15 --y_target 0 &

srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
--model_type core --model_name ResNet-18 \
--trigger_name square_trigger_3x3 --poisoning_rate 0.20 --y_target 0 &

# Table 2. The effects of the trigger size.
# srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
# --model_type core --model_name ResNet-18 \
# --trigger_name square_trigger_3x3 --poisoning_rate 0.05 --y_target 0

srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
--model_type core --model_name ResNet-18 \
--trigger_name square_trigger_4x4 --poisoning_rate 0.05 --y_target 0 &

srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
--model_type core --model_name ResNet-18 \
--trigger_name square_trigger_5x5 --poisoning_rate 0.05 --y_target 0 &

srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
--model_type core --model_name ResNet-18 \
--trigger_name square_trigger_7x7 --poisoning_rate 0.05 --y_target 0 &

# Table 3. The effects of the target label.
# srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
# --model_type core --model_name ResNet-18 \
# --trigger_name square_trigger_3x3 --poisoning_rate 0.05 --y_target 0

srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
--model_type core --model_name ResNet-18 \
--trigger_name square_trigger_3x3 --poisoning_rate 0.05 --y_target 1 &

srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
--model_type core --model_name ResNet-18 \
--trigger_name square_trigger_3x3 --poisoning_rate 0.05 --y_target 2 &

# Table 4. The effects of the model structure.
# srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
# --model_type core --model_name ResNet-18 \
# --trigger_name square_trigger_3x3 --poisoning_rate 0.05 --y_target 0

srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
--model_type core --model_name ResNet-34 \
--trigger_name square_trigger_3x3 --poisoning_rate 0.05 --y_target 0 &

# srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
# --model_type core --model_name VGG-13 \
# --trigger_name square_trigger_3x3 --poisoning_rate 0.05 --y_target 0

srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
--model_type core --model_name VGG-16 \
--trigger_name square_trigger_3x3 --poisoning_rate 0.05 --y_target 0

srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
--model_type core --model_name VGG-19 \
--trigger_name square_trigger_3x3 --poisoning_rate 0.05 --y_target 0 &



# Rebuttal
srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
--model_type core --model_name ResNet-18 \
--trigger_name pencil_trigger_with_9_pixels --poisoning_rate 0.05 --y_target 0 &

srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
--model_type core --model_name ResNet-18 \
--trigger_name triangle_trigger_with_9_pixels --poisoning_rate 0.05 --y_target 0 &

srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_BadNets \
--model_type core --model_name ResNet-18 \
--trigger_name compass_trigger_with_9_pixels --poisoning_rate 0.05 --y_target 0 &


