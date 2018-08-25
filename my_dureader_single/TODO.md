**[OK] try Match-LSTM**
- transfer learning of BIDAF
- bagging & boosting
- build QAnet



## SCORE LOG

-   **[FIRST COMMIT]**
    discription: all_data -> BIDAF, epoch 10
    yes-no: yesno -> all_no, epoch 10, yes_num=13
    entity: entity -> BIDAF, epoch 10
    All param default.
    
    SCORE: 7.2

-  yes-no: yes-no data->BIDAF, delete `。`: Runtime error.

-  description: description->BIDAF, epoch 10, well-fit.

- entity: entity-> QANET, epoch 20, well-fit

-   **[SECOND COMMIT]**
    discription: **description** -> BIDAF, epoch 10
    yes-no: all_data -> all_yes, epoch 10, yes_num=0
    entity: entity -> BIDAF, epoch 10
    All param default.
    
    SCORE: 7.6

-   **[THIRD COMMIT]**
    discription: **description** -> BIDAF, epoch 10
    yes-no: all_data -> all_yes, epoch 10, yes_num=0
    entity: entity -> BIDAF, epoch 8
    All param default.
    
    SCORE: ??



## TRAIN LOG

[10:10] 
QAnet, train_data: description, epoch 20, ID = None

[10:36]
QAnet, train_data: description, epoch 100, ID = 1036
LOSS: 0~57 epoch decent

[13:29]
BIDAF, train_data: entity_dev+train, epoch 11, 

## DEBUG LOG

python run.py --rm_vocab_path /disk/Pufa-Dureader/data/vocab_search_pretrain/description/vocab.data --pre_train_file ../data/mrc.vec --train_files ../data/sample/train_sample.json --dev_files ../data/sample/dev_sample.json --test_files ../data/sample/test_sample.json 
