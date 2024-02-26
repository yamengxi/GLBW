# Rebuttal
srun -n 1 -c 6 -p gpu --gpus=1 python -m tests.train_Blended \
--model_type core --model_name ResNet-18 \
--trigger_name square_trigger_3x3 --poisoning_rate 0.05 --y_target 0
