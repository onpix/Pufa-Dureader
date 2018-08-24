#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python ../run.py --pre_train_file ../../data/mrc.vec --train_files ../../data/sample/train_sample.json --dev_files ../../data/sample/dev_sample.json --test_files ../../data/sample/test_sample.json 
