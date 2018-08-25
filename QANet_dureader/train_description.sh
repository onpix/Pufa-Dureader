# !/usr/bin/bash

nohup python -u cli.py --vocab_dir ../data/vocab_QAnet/description/ --train --batch_size 16 --learning_rate 1e-3 --optim adam --decay 0.9999 --weight_decay 1e-5 --max_norm_grad 5.0 --dropout 0.0 --head_size 1 --hidden_size 64 --epochs 28 --gpu 1 --train_files ../data/description/data_train_preprocessed.json --dev_files ../data/description/data_dev_preprocessed.json --test_files ../data/description/data_test_preprocessed.json --result_dir ../data/results_QAnet/description/ --model_dir ../data/models_QAnet/description/ >description.log 2>&1 &
