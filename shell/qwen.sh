#!/bin/bash
# bash ~/RASAMoE/shell/qwen.sh > ~/RASAMoE/shell/qwen.log 2>&1

attacks=("johnny" "flipattack" "deepinception")
expert=("768")
work_dir="~/RASAMoE"
for attack in "${attacks[@]}"; do
    echo "Running attack: ${attack}"

for expert_num in "${expert[@]}"; do
    echo "Running expert: ${expert_num}"

# Only support single GPU for detect_expert.py now
CUDA_VISIBLE_DEVICES=6 python detect_expert.py \
    --dataset_path ${work_dir}/experiments/${attack}/qwen_train.json \
    --model_name Qwen/Qwen3-30B-A3B \
    --max_generation_tokens 128 \
    --unsafe_expert_top_k ${expert_num} \
    --batch_unsafe_threshold 0.0 \
    --global_unsafe_threshold 0.05 \
    --experiment_dir ${work_dir}/experiments/${attack}/qwen_${expert_num} 




CUDA_VISIBLE_DEVICES=0,1,2,3 python train_alternating_optimization.py \
    --model_name Qwen/Qwen3-30B-A3B \
    --output_dir ${work_dir}/experiments/${attack}/qwen_${expert_num} \
    --num_rounds 3 \
    --expert_epochs_per_round 1 \
    --router_epochs_per_round 1 \
    --expert_lr 5e-5 \
    --router_lr 1e-6 \
    --kl_type forward \
    --batch_data_path ${work_dir}/experiments/${attack}/qwen_${expert_num}/batch_data_Qwen--Qwen3-30B-A3B.pkl


CUDA_VISIBLE_DEVICES=0,1,2,3 python ~/RASAMoE/3_evaluation/evaluation_example.py \
    --model_name ${work_dir}/experiments/${attack}/qwen_${expert_num}/round_3 \
    --prefix qwen-${expert_num}-round3 \
    --output_dir ${work_dir}/experiments/${attack}/qwen_${expert_num} \
    --flipattack_path ${work_dir}/experiments/flipattack/qwen_test.json \
    --deepinception_path ${work_dir}/experiments/deepinception/qwen_test.json \
    --johnny_path ${work_dir}/experiments/johnny/qwen_test.json \
    --xteaming_path ${work_dir}/experiments/xteaming/qwen_test.json \
    --datasets  strongreject xstest flipattack deepinception johnny mmlu gsm8k truthfulqa \
    --tensor_parallel_size 4 # IF YOU HAVE MULTIPLE GPUs,change this to the number of GPUs you have

done
done
