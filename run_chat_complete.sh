#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH -J llama_contenseo

torchrun --nproc_per_node 1 chat_completion.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 128 --max_batch_size 4 > output.txt
