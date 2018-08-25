#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python run.py --rm_vocab_path /disk/Pufa-Dureader/data/vocab_search_pretrain/description/vocab.data --pre_train_file ../data/mrc.vec --train_files ../data/sample/train_sample.json --dev_files ../data/sample/dev_sample.json --test_files ../data/sample/test_sample.json 
