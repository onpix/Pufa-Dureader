#!/usr/bin/env bash

echo $0
if [ $0 == '-a' ]
then 
    export CUDA_VISIBLE_DEVICES=1
    nohup python -u run.py --rm_vocab_path /disk/Pufa-Dureader/data/vocab_search_pretrain/vocab.data  1> ./log/all.log 2>&1 &
elif [ $0 == '-y' ]
then
    nohup python -u run.py --rm_vocab_path /disk/Pufa-Dureader/data/vocab_search_pretrain/yesno/vocab.data --model_dir ../data/models_search_pretrain/yesno --vocab_dir ../data/vocab_search_pretrain/yesno --result_dir ../data/results_demo/yesno --test_files ../data/yesno/data_test_preprocessed.json.bak --dev_files ../data/yesno/data_dev_preprocessed.json.bak --train_files ../data/yesno/data_train_preprocessed.json.bak  > ./log/yesno.log 2>&1 & 
elif [ $0 == '-e' ]
then
    nohup python -u run.py --rm_vocab_path /disk/Pufa-Dureader/data/vocab_search_pretrain/entity/vocab.data --epochs 18 --model_dir ../data/models_search_pretrain/entity --vocab_dir ../data/vocab_search_pretrain/entity --result_dir ../data/results_demo/entity --test_files ../data/entity/data_test_preprocessed.json --dev_files ../data/entity/data_dev_preprocessed.json --train_files ../data/entity/data_train_preprocessed.json > ./log/entity.log 2>&1 &
elif [ $0 == '-d' ]
then
    nohup python -u run.py --rm_vocab_path /disk/Pufa-Dureader/data/vocab_search_pretrain/description/vocab.data --epochs 10 --model_dir ../data/models_search_pretrain/description --vocab_dir ../data/vocab_search_pretrain/description --result_dir ../data/results_demo/description --test_files ../data/description/data_dev_preprocessed.json --dev_files ../data/description/data_dev_preprocessed.json --train_files ../data/description/data_train_preprocessed.json > description.log 2>&1 &
else 
    echo '[Err] Unkown options.'
fi



