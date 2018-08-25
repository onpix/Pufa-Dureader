import os 
import torch 
import numpy as np
import torch.nn as nn
import json
import os 
RAW_FILE = '/run/media/why/DATA/why的程序测试/AI_Lab/whyAI/Pufa-Dureader/data/answer_raw.json'
TEST_FILE = '/run/media/why/DATA/why的程序测试/AI_Lab/whyAI/Pufa-Dureader/data/data_test_preprocessed.json.bak'
save_name = 'todo.csv'


def gen_data():
    f = open(RAW_FILE, 'r', encoding='utf-8')
    answers = []
    ids = []
    for x in f.readlines():
        data = []
        sample = json.loads(x)
        # label = 1 if sample['answers'][0]=='是' else 0
        if sample['question_type'] == 'yes_no':
            an = sample['answers'][0]
            qid = sample['question_id']
            ids.append(qid)
            answers.append(an)  
    return answers, ids

def find_qua(ids):
    quas = []
    f = open(TEST_FILE, 'r', encoding='utf-8')
    for x in f.readlines():
        sample = json.loads(x)
        if sample['question_type'] == 'yes_no':
            now_id = sample['question_id']
            quas.append(sample['question'])
    return quas
        
        

def run():
    answers, ids = gen_data()
    quas = find_qua(ids)
    pass

if __name__ == '__main__':
    run()

    
