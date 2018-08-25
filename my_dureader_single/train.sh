#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
nohup python -u run.py --rm_vocab_path /disk/Pufa-Dureader/data/vocab_search_pretrain/vocab.data  1> ./log/all.log 2>&1 &
