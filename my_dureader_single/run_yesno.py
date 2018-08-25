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

class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.layer1 = nn.Linear()


def convert():
    with open(VOCAB_PATH, 'rb') as fin:
        vocab = pickle.load(fin)
    brc_data = BRCDataset(max_p_num, max_p_len, max_q_len,
                          train_files, dev_files)
    brc_data.convert_to_ids(vocab)
    for x in range(len(brc_data.train_set)):
        p = brc_data.train_set[0]['passages'][0]['passage_token_ids']
        q = brc_data.train_set[0]['question_token_ids']
    return 0 



def run_yesno():
    gen_yesno_vec()
    para = np.loadtxt('../data/para.txt')
    qua = np.loadtxt('../data/qua.txt')
    print(para)


if __name__ == '__main__':
    run_yesno()
    
    