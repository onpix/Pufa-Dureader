#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
nohup python -u run.py --epochs 30 --algo MLSTM 1>test_search.log 2>&1 &
