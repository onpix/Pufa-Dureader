**[OK] try Match-LSTM**
- transfer learning of BIDAF
- bagging & boosting
- build QAnet



## score log:

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
    yes-no: all_data -> all_no, epoch 10, yes_num=0
    entity: entity -> BIDAF, epoch 10
    All param default.
    
    SCORE: 7.6



## TRAIN LOG

[10:10] 
QAnet, train_data: description, epoch 20
