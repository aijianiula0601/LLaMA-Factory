#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ..

output_dir="/mnt/cephfs/hjh/train_record/nlp/llamafactory/huggyllama_llama-7b/lora/sft"
model_name_or_path="huggyllama/llama-7b"
dataset_dir="./data"
dataset_name="identity"

CUDA_VISIBLE_DEVICES=0 python src/train.py \
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
  --fp16

#output_dir="/mnt/cephfs/hjh/train_record/nlp/llamafactory/huggyllama_llama-7b/lora/sft"
#model_name_or_path="/mnt/cephfs/hjh/huggingface_models/nlp/llama3/Meta-Llama-3-70B-Instruct"
#dataset_dir="./data"
#dataset_name="ultrachat_200k"
#
#CUDA_VISIBLE_DEVICES=0 python src/train.py \
#    --stage sft \
#    --do_train \
#    --model_name_or_path ${model_name_or_path} \
#    --dataset ${dataset_name} \
#    --dataset_dir ${dataset_dir} \
#    --template default \
#    --finetuning_type lora \
#    --lora_target q_proj,v_proj \
#    --output_dir ${output_dir} \
#    --overwrite_cache \
#    --overwrite_output_dir \
#    --cutoff_len 1024 \
#    --preprocessing_num_workers 16 \
#    --per_device_train_batch_size 1 \
#    --per_device_eval_batch_size 1 \
#    --gradient_accumulation_steps 8 \
#    --lr_scheduler_type cosine \
#    --logging_steps 10 \
#    --warmup_steps 20 \
#    --save_steps 100 \
#    --eval_steps 100 \
#    --evaluation_strategy steps \
#    --load_best_model_at_end \
#    --learning_rate 5e-5 \
#    --num_train_epochs 3.0 \
#    --max_samples 3000 \
#    --val_size 0.1 \
#    --plot_loss \
#    --fp16 \
#    --split train_sft
