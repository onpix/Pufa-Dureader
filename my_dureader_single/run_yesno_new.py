import os 
import torch 
import numpy as np
import torch.nn as nn
import json 
TEST_PATH = '../data/yesno/data_test_preprocessed.json.bak'
TRAIN_PATH = '../data/yesno/data_train_preprocessed.json.bak'
RAW_FILE = '/run/media/why/DATA/why的程序测试/AI_Lab/whyAI/Pufa-Dureader/data/answer_raw.json'


TRAIN_PATH = '/run/media/why/DATA/why的程序测试/AI_Lab/whyAI/Pufa-Dureader/data/yesno/data_train_preprocessed.json.bak'
TEST_PATH = '/run/media/why/DATA/why的程序测试/AI_Lab/whyAI/Pufa-Dureader/data/yesno/data_test_preprocessed.json.bak'
VEC_PATH = '/run/media/why/DATA/why的程序测试/AI_Lab/whyAI/AI_finance/f.vec'


def gen_data():
    f = open(TRAIN_PATH, 'r', encoding='utf-8')
    datas = []
    labels = []
    for x in f.readlines():
        data = []
        sample = json.loads(x)
        label = 1 if sample['answers'][0]=='是' else 0
        q = sample['segmented_question']
        p = sample['documents'][0]['segmented_paragraphs'][0]
        data = p+q
        datas.append(data)
        labels.append(label)
    return datas, labels


def word2vec(datas, labels):
    

def run():
    datas, labels = gen_data()
    print('test ')


if __name__ == '__main__':
    run()

    
