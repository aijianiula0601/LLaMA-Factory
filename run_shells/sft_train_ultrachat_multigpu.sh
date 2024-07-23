#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

######################################################################################################
#参考：
# https://qwen.readthedocs.io/en/latest/training/SFT/llama_factory.html
######################################################################################################
cd ..

output_dir="/mnt/cephfs/hjh/train_record/nlp/llamafactory/huggyllama_llama-7b/lora/sft"
model_name_or_path="huggyllama/llama-7b"
#model_name_or_path="/mnt/cephfs/hjh/huggingface_models/nlp/llama3/Meta-Llama-3-70B-Instruct" # 从挂载目录中加载模型，速度很慢
#model_name_or_path="/data1/llm-ckpt/Higgs-Llama-3-70B"
dataset_dir="./data"
dataset_name="ultrachat_200k"


#CUDA_VISIBLE_DEVICES=6,7 \
torchrun \
--nproc_per_node 8 \
--nnodes 1 \
--node_rank 0 \
--master_addr localhost \
--master_port 26458 \
src/train.py \
  --stage sft \
  --do_train \
  --model_name_or_path ${model_name_or_path} \
  --dataset ${dataset_name} \
  --dataset_dir ${dataset_dir} \
  --template default \
  --finetuning_type lora \
  --lora_target q_proj,v_proj \
  --output_dir ${output_dir} \
  --overwrite_cache \
  --overwrite_output_dir \
  --cutoff_len 1024 \
  --preprocessing_num_workers 16 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --warmup_steps 20 \
  --save_steps 100 \
  --eval_steps 100 \
  --evaluation_strategy steps \
  --load_best_model_at_end \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --max_samples 3000 \
  --val_size 0.1 \
  --plot_loss \
  --fp16 \
  --split train_sft