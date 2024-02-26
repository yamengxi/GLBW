# debug
CUDA_VISIBLE_DEVICES_LIST=("3")
GPU_num=${#CUDA_VISIBLE_DEVICES_LIST[@]}
cnt=0

model=resnet18
model_=ResNet-18
trig_size_list=("10")
length=${#trig_size_list[@]}
for ((i=0;i<$length;i++)); do
    experiment_name=${model_}_ImageNet100_fixed_square_${trig_size_list[i]}x${trig_size_list[i]}
    ((GPU_index=cnt%GPU_num))
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[GPU_index]} python evalxai_with_GPU_random_samples_and_customized_trigger.py --data_path /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
    --path ${experiment_name}.pth.tar --model ${model} --trig_path /disk/yamengxi/Backdoor/XAI/fixed_square_${trig_size_list[i]}x${trig_size_list[i]}.pth\
    --save_dir eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[GPU_index]}\] 2>&1 | tee eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[GPU_index]}\].log
    ((cnt=cnt+1))
done


# evalxai with fixed square trigger
CUDA_VISIBLE_DEVICES_LIST=("3" "7")
GPU_num=${#CUDA_VISIBLE_DEVICES_LIST[@]}
cnt=0

model=resnet18
model_=ResNet-18
trig_size_list=("40")
length=${#trig_size_list[@]}
for ((i=0;i<$length;i++)); do
    experiment_name=${model_}_ImageNet100_fixed_square_${trig_size_list[i]}x${trig_size_list[i]}
    ((GPU_index=cnt%GPU_num))
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[GPU_index]} python evalxai_with_GPU_random_samples_and_customized_trigger.py --data_path /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
    --path ${experiment_name}.pth.tar --model ${model} --trig_path /disk/yamengxi/Backdoor/XAI/fixed_square_${trig_size_list[i]}x${trig_size_list[i]}.pth\
    --save_dir eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[GPU_index]}\] 2>&1 | tee eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[GPU_index]}\].log &
    ((cnt=cnt+1))
done

model=resnet34
model_=ResNet-34
trig_size_list=("5" "20" "60")
length=${#trig_size_list[@]}
for ((i=0;i<$length;i++)); do
    experiment_name=${model_}_ImageNet100_fixed_square_${trig_size_list[i]}x${trig_size_list[i]}
    ((GPU_index=cnt%GPU_num))
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[GPU_index]} python evalxai_with_GPU_random_samples_and_customized_trigger.py --data_path /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
    --path ${experiment_name}.pth.tar --model ${model} --trig_path /disk/yamengxi/Backdoor/XAI/fixed_square_${trig_size_list[i]}x${trig_size_list[i]}.pth\
    --save_dir eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[GPU_index]}\] 2>&1 | tee eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[GPU_index]}\].log &
    ((cnt=cnt+1))
done

model=resnet50
model_=ResNet-50
trig_size_list=("5" "10" "20" "40" "60")
length=${#trig_size_list[@]}
for ((i=0;i<$length;i++)); do
    experiment_name=${model_}_ImageNet100_fixed_square_${trig_size_list[i]}x${trig_size_list[i]}
    ((GPU_index=cnt%GPU_num))
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[GPU_index]} python evalxai_with_GPU_random_samples_and_customized_trigger.py --data_path /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
    --path ${experiment_name}.pth.tar --model ${model} --trig_path /disk/yamengxi/Backdoor/XAI/fixed_square_${trig_size_list[i]}x${trig_size_list[i]}.pth\
    --save_dir eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[GPU_index]}\] 2>&1 | tee eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[GPU_index]}\].log &
    ((cnt=cnt+1))
done


# evalxai with NC trigger (square)
CUDA_VISIBLE_DEVICES_LIST=("3" "7")
GPU_num=${#CUDA_VISIBLE_DEVICES_LIST[@]}
cnt=0

model_list=(
"resnet18"
"resnet18"
"resnet18"
"resnet18"
"resnet18"
# "resnet50"
# "resnet50"
# "resnet50"
# "resnet50"
# "resnet50"
)

model_name_list=(
"ResNet-18_ImageNet100_fixed_square_5x5"
"ResNet-18_ImageNet100_fixed_square_10x10"
"ResNet-18_ImageNet100_fixed_square_20x20"
"ResNet-18_ImageNet100_fixed_square_40x40"
"ResNet-18_ImageNet100_fixed_square_60x60"
# "ResNet-50_ImageNet100_fixed_square_5x5"
# "ResNet-50_ImageNet100_fixed_square_10x10"
# "ResNet-50_ImageNet100_fixed_square_20x20"
# "ResNet-50_ImageNet100_fixed_square_40x40"
# "ResNet-50_ImageNet100_fixed_square_60x60"
)

NC_init_cost_list=(
"init_cost0.00036"
"init_cost9e-05"
"init_cost2.25e-05"
"init_cost5.625e-06"
"init_cost2.25e-06"
# "init_cost0.00036"
# "init_cost9e-05"
# "init_cost2.25e-05"
# "init_cost5.625e-06"
# "init_cost2.5e-06"
)

trigger_id_list=(
"1_2022-10-19_21:24:30"
"1_2022-10-19_21:24:30"
"1_2022-10-19_21:24:30"
"5_2022-10-24_16:44:10"
"1_2022-10-20_19:50:43"
# "4_2022-10-21_21:21:47"
# "1_2022-10-24_15:48:00"
# "2_2022-10-24_00:45:18"
# "4_2022-10-24_01:36:57"
# "17_2022-10-22_01:14:36"
)

length=${#model_name_list[@]}
for ((i=0;i<$length;i++)); do
    experiment_name=${model_name_list[i]}_with_NC_trigger_${NC_init_cost_list[i]}_${trigger_id_list[i]}
    ((GPU_index=cnt%GPU_num))
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[GPU_index]} python evalxai_with_GPU_random_samples_and_customized_trigger.py --data_path /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
    --path ${model_name_list[i]}.pth.tar --model ${model_list[i]} --trig_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/NeuralCleanse/ImageNet100_experiments/${model_name_list[i]}_${NC_init_cost_list[i]}/${trigger_id_list[i]}/0/trigger.npz \
    --save_dir eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[GPU_index]}\] 2>&1 | tee eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[GPU_index]}\].log &
    ((cnt=cnt+1))
done


# evalxai with NC trigger (outlier)
CUDA_VISIBLE_DEVICES_LIST=("7" "3")
GPU_num=${#CUDA_VISIBLE_DEVICES_LIST[@]}
cnt=0

model_list=(
"resnet18"
"resnet18"
"resnet18"
"resnet18"
"resnet18"
# "resnet50"
# "resnet50"
# "resnet50"
# "resnet50"
# "resnet50"
# "resnet50"
)

model_name_list=(
"ResNet-18_ImageNet100_fixed_square_5x5"
"ResNet-18_ImageNet100_fixed_square_10x10"
"ResNet-18_ImageNet100_fixed_square_20x20"
"ResNet-18_ImageNet100_fixed_square_40x40"
"ResNet-18_ImageNet100_fixed_square_60x60"
# "ResNet-50_ImageNet100_fixed_square_5x5"
# "ResNet-50_ImageNet100_fixed_square_10x10"
# "ResNet-50_ImageNet100_fixed_square_20x20"
# "ResNet-50_ImageNet100_fixed_square_40x40"
# "ResNet-50_ImageNet100_fixed_square_60x60"
# "ResNet-50_ImageNet100_fixed_square_60x60"
)

NC_init_cost_list=(
"init_cost0.00036"
"init_cost9e-05"
"init_cost2.25e-05"
"init_cost5.625e-06"
"init_cost2.5e-06"
# "init_cost0.00036"
# "init_cost9e-05"
# "init_cost2.25e-05"
# "init_cost5.625e-06"
# "init_cost2.5e-06"
# "init_cost2.5e-06"
)

trigger_id_list=(
"10_2022-10-19_23:28:40"
"38_2022-10-20_05:56:57"
"52_2022-10-20_09:03:40"
"6_2022-10-24_16:58:18"
"1_2022-10-19_21:24:30"
# "34_2022-10-22_06:14:10"
# "14_2022-10-24_23:44:37"
# "11_2022-10-24_03:26:06"
# "6_2022-10-24_02:23:20"
# "17_2022-10-22_01:14:36"
# "126_2022-10-23_09:42:23"
)

length=${#model_name_list[@]}
for ((i=0;i<$length;i++)); do
    experiment_name=${model_name_list[i]}_with_NC_trigger_${NC_init_cost_list[i]}_${trigger_id_list[i]}
    ((GPU_index=cnt%GPU_num))
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[GPU_index]} python evalxai_with_GPU_random_samples_and_customized_trigger.py --data_path /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
    --path ${model_name_list[i]}.pth.tar --model ${model_list[i]} --trig_path /disk/yamengxi/Backdoor/XAI/Backdoor_XAI/NeuralCleanse/ImageNet100_experiments/${model_name_list[i]}_${NC_init_cost_list[i]}/${trigger_id_list[i]}/0/trigger.npz \
    --save_dir eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[GPU_index]}\] 2>&1 | tee eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[GPU_index]}\].log &
    ((cnt=cnt+1))
done
