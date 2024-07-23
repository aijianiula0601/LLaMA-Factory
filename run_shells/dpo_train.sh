#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ..

output_dir="/mnt/cephfs/hjh/train_record/nlp/llamafactory/huggyllama_llama-7b/lora/dpo"
model_name_or_path="/data1/llm-ckpt/Mistral-7B-v0.1"
adapter_name_or_path="/mnt/cephfs/hjh/train_record/nlp/llamafactory/huggyllama_llama-7b/lora/sft"
dataset_dir="./data"
dataset_name="dpo_en_demo"

#CUDA_VISIBLE_DEVICES=0 python src/train.py \
#    --stage dpo \
#    --do_train \
#    --model_name_or_path ${model_name_or_path} \
#    --adapter_name_or_path ${adapter_name_or_path} \
#    --create_new_adapter \
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
#    --learning_rate 1e-5 \
#    --num_train_epochs 1.0 \
#    --max_samples 1000 \
#    --val_size 0.1 \
#    --plot_loss \
#    --fp16


output_dir="/mnt/cephfs/hjh/train_record/nlp/llamafactory/huggyllama_llama-7b/lora/dpo"
model_name_or_path="/data1/llm-ckpt/Mistral-7B-v0.1"
dataset_dir="./data"
dataset_name="dpo_en_demo"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_addr 127.0.0.1 --master_port 27223 src/train.py \
--cutoff_len 1024 \
--dataset ${dataset_name} \
--dataset_dir ${dataset_dir} \
--ddp_timeout 180000000 \
--do_train true \
--finetuning_type lora \
--flash_attn auto \
--fp16 true \
--gradient_accumulation_steps 8 \
--include_num_input_tokens_seen true \
--learning_rate 5.0e-05 \
--logging_steps 5 \
--lora_alpha 16 \
--lora_dropout 0 \
--lora_rank 8 \
--lora_target all \
--lr_scheduler_type cosine \
--max_grad_norm 1.0 \
--max_samples 100000 \
--model_name_or_path ${model_name_or_path} \
--num_train_epochs 3.0 \
--optim adamw_torch \
--output_dir ${output_dir} \
--packing false \
--per_device_train_batch_size 2 \
--plot_loss true \
--pref_beta 0.1 \
--pref_ftx 0 \
--pref_loss sigmoid \
--preprocessing_num_workers 16 \
--report_to none \
--save_steps 100 \
--stage dpo \
--template default \
--warmup_steps 0
