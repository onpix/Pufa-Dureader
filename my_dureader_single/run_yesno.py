import torch 
import numpy as np
import torch.nn as nn
import my_rc_model
from dataset import BRCDataset
import pickle
from run import gen_yesno_vec
VOCAB_PATH = '../data/vocab_search_pretrain/vocab.data'
train_files = '../data/yesno/data_train_preprocessed.json'
dev_files = '../data/yesno/data_dev_preprocessed.json'
max_p_num = 5
max_q_len = 60
max_p_len = 500
BATCH_SIZE = 16

class LSTM(nn.Module):
    pass

def run_yesno():
    gen_yesno_vec()
    para = np.load('../data/para.npy')
    qua = np.load('../data/qua.npy')
    print(para)


if __name__ == '__main__':
    run_yesno()
    
    