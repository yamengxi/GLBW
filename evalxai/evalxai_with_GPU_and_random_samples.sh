# debug
CUDA_VISIBLE_DEVICES=0 python evalxai_with_GPU_and_random_samples.py --data_path /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
--path ResNet-18_ImageNet100_fixed_square_10x10.pth.tar --model resnet18 --trig fixed_square --trig_size 10 \
--save_dir eval_debug 2>&1 | tee eval_debug_.log


# 固定同一模型；不同数据集（在ImageNet100的测试集中随机选取100张图片）
model=resnet18
trig=fixed_square
trig_size=20
experiment_name=ResNet-18_ImageNet100_${trig}_${trig_size}x${trig_size}
for CUDA_VISIBLE_DEVICES in 0 2 3 7; do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python evalxai_with_GPU_and_random_samples.py --data_path /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
    --path ${experiment_name}.pth.tar --model ${model} --trig ${trig} --trig_size ${trig_size} \
    --save_dir eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES}\] 2>&1 | tee eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES}\].log &
done


# 在相同配置下，重复独立训练模型；不同数据集（在ImageNet100的测试集中随机选取100张图片）
model=resnet18
trig=fixed_square
trig_size=20
experiment_name=ResNet-18_ImageNet100_${trig}_${trig_size}x${trig_size}
CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python evalxai_with_GPU_and_random_samples.py --data_path /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
--path ${experiment_name}.pth.tar --model ${model} --trig ${trig} --trig_size ${trig_size} \
--save_dir eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES}\] 2>&1 | tee eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES}\].log &

model=resnet18
trig=fixed_square
trig_size=20
experiment_name=ResNet-18_ImageNet100_${trig}_${trig_size}x${trig_size}_2
CUDA_VISIBLE_DEVICES=2
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python evalxai_with_GPU_and_random_samples.py --data_path /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
--path ${experiment_name}.pth.tar --model ${model} --trig ${trig} --trig_size ${trig_size} \
--save_dir eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES}\] 2>&1 | tee eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES}\].log &

model=resnet18
trig=fixed_square
trig_size=20
experiment_name=ResNet-18_ImageNet100_${trig}_${trig_size}x${trig_size}_3
CUDA_VISIBLE_DEVICES=3
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python evalxai_with_GPU_and_random_samples.py --data_path /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
--path ${experiment_name}.pth.tar --model ${model} --trig ${trig} --trig_size ${trig_size} \
--save_dir eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES}\] 2>&1 | tee eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES}\].log &


# evalxai单变量规律探究
CUDA_VISIBLE_DEVICES_LIST=("0" "2" "3" "7")
GPU_num=${#CUDA_VISIBLE_DEVICES_LIST[@]}
cnt=0

model=resnet18
model_=ResNet-18
trig=fixed_square
trig_size_list=("5" "10" "20" "60")
length=${#trig_size_list[@]}
for ((i=0;i<$length;i++)); do
    experiment_name=${model_}_ImageNet100_${trig}_${trig_size_list[i]}x${trig_size_list[i]}
    ((GPU_index=cnt%GPU_num))
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[GPU_index]} python evalxai_with_GPU_and_random_samples.py --data_path /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
    --path ${experiment_name}.pth.tar --model ${model} --trig ${trig} --trig_size ${trig_size_list[i]} \
    --save_dir eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[GPU_index]}\] 2>&1 | tee eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[GPU_index]}\].log &
    ((cnt=cnt+1))
done

model=resnet34
model_=ResNet-34
trig=fixed_square
trig_size_list=("5" "20" "60")
length=${#trig_size_list[@]}
for ((i=0;i<$length;i++)); do
    experiment_name=${model_}_ImageNet100_${trig}_${trig_size_list[i]}x${trig_size_list[i]}
    ((GPU_index=cnt%GPU_num))
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[GPU_index]} python evalxai_with_GPU_and_random_samples.py --data_path /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
    --path ${experiment_name}.pth.tar --model ${model} --trig ${trig} --trig_size ${trig_size_list[i]} \
    --save_dir eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[GPU_index]}\] 2>&1 | tee eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[GPU_index]}\].log &
    ((cnt=cnt+1))
done

model=resnet50
model_=ResNet-50
trig=fixed_square
trig_size_list=("5" "10" "20" "40" "60")
length=${#trig_size_list[@]}
for ((i=0;i<$length;i++)); do
    experiment_name=${model_}_ImageNet100_${trig}_${trig_size_list[i]}x${trig_size_list[i]}
    ((GPU_index=cnt%GPU_num))
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[GPU_index]} python evalxai_with_GPU_and_random_samples.py --data_path /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
    --path ${experiment_name}.pth.tar --model ${model} --trig ${trig} --trig_size ${trig_size_list[i]} \
    --save_dir eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[GPU_index]}\] 2>&1 | tee eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[GPU_index]}\].log &
    ((cnt=cnt+1))
done


# # 模型=ResNet-50；trigger type=fixed_square；trigger size=60, 40, 20, 10, 5
# model=resnet50
# trig=fixed_square
# trig_size_list=("60" "40" "20" "10" "5")
# CUDA_VISIBLE_DEVICES_LIST=("0" "2" "3" "2" "3")
# length=${#CUDA_VISIBLE_DEVICES_LIST[@]}
# for ((i=0;i<$length;i++)); do
#     experiment_name=ResNet-50_ImageNet100_${trig}_${trig_size_list[i]}x${trig_size_list[i]}
#     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[i]} python evalxai_with_GPU_and_random_samples.py --data_path /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
#     --path ${experiment_name}.pth.tar --model ${model} --trig ${trig} --trig_size ${trig_size_list[i]} \
#     --save_dir eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[i]}\] 2>&1 | tee eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[i]}\].log &
# done


# # 模型=ResNet-50, ResNet-34, ResNet-18；trigger type=fixed_square；trigger size=60
# model_list=("resnet50" "resnet34" "resnet18")
# model_list2=("ResNet-50" "ResNet-34" "ResNet-18")
# trig=fixed_square
# trig_size=60
# CUDA_VISIBLE_DEVICES_LIST=("0" "2" "3")
# length=${#CUDA_VISIBLE_DEVICES_LIST[@]}
# for ((i=0;i<$length;i++)); do
#     experiment_name=${model_list2[i]}_ImageNet100_${trig}_${trig_size}x${trig_size}
#     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[i]} python evalxai_with_GPU_and_random_samples.py --data_path /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
#     --path ${experiment_name}.pth.tar --model ${model_list[i]} --trig ${trig} --trig_size ${trig_size} \
#     --save_dir eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[i]}\] 2>&1 | tee eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[i]}\].log &
# done



# # 模型=ResNet-50, ResNet-34, ResNet-18；trigger type=fixed_square；trigger size=20
# model_list=("resnet50" "resnet34" "resnet18")
# model_list2=("ResNet-50" "ResNet-34" "ResNet-18")
# trig=fixed_square
# trig_size=20
# CUDA_VISIBLE_DEVICES_LIST=("0" "2" "3")
# length=${#CUDA_VISIBLE_DEVICES_LIST[@]}
# for ((i=0;i<$length;i++)); do
#     experiment_name=${model_list2[i]}_ImageNet100_${trig}_${trig_size}x${trig_size}
#     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[i]} python evalxai_with_GPU_and_random_samples.py --data_path /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
#     --path ${experiment_name}.pth.tar --model ${model_list[i]} --trig ${trig} --trig_size ${trig_size} \
#     --save_dir eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[i]}\] 2>&1 | tee eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[i]}\].log &
# done


# # 模型=ResNet-50, ResNet-34, ResNet-18；trigger type=fixed_square；trigger size=5
# model_list=("resnet50" "resnet34" "resnet18")
# model_list2=("ResNet-50" "ResNet-34" "ResNet-18")
# trig=fixed_square
# trig_size=5
# CUDA_VISIBLE_DEVICES_LIST=("0" "2" "3")
# length=${#CUDA_VISIBLE_DEVICES_LIST[@]}
# for ((i=0;i<$length;i++)); do
#     experiment_name=${model_list2[i]}_ImageNet100_${trig}_${trig_size}x${trig_size}
#     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[i]} python evalxai_with_GPU_and_random_samples.py --data_path /disk/yamengxi/Backdoor/XAI/datasets/ImageNet_100/val \
#     --path ${experiment_name}.pth.tar --model ${model_list[i]} --trig ${trig} --trig_size ${trig_size} \
#     --save_dir eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[i]}\] 2>&1 | tee eval_${experiment_name}_\[${CUDA_VISIBLE_DEVICES_LIST[i]}\].log &
# done
