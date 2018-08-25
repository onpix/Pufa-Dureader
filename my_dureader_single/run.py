# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module prepares and runs the whole system.
"""
import pickle
import argparse
import os
from dataset import BRCDataset
from vocab import Vocab
from my_rc_model import RCModel
import logging
import torch
import datetime

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(
        'Reading Comprehension on BaiduRC dataset')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_prob', type=float, default=0.5,
                                help='probability of an element to be zeroed')
    train_settings.add_argument('--batch_size', type=int, default=16,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')
    train_settings.add_argument('--use_pre_train', type=bool, default=True,
                                help='use pre_train vec')


    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', choices=['BIDAF', 'MLSTM'], default='BIDAF',
                                help='choose the algorithm to use')
    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=150,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_p_num', type=int, default=5,
                                help='max passage num in one sample')
    model_settings.add_argument('--max_p_len', type=int, default=500,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=60,
                                help='max length of question')
    model_settings.add_argument('--max_a_len', type=int, default=200,
                                help='max length of answer')
    # [WHY edit] add continue training.
    # '../data/models_search_pretrain/BIDAF_10'
    train_settings.add_argument('--pretrain_model_path', default=False,
                                help='input the path of pretrianed model to continue train.')
    train_settings.add_argument('--rm_vocab_path', default=False,
                                help='delete vocab data to solve VocabSize Error.')
    

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', nargs='+',
                               # default=['./data/demo/trainset/search.train.json'],
                               # default=['./data/preprocessed/trainset/search.train.json',
                               #          './data/preprocessed/trainset/zhidao.train.json'],
                               # default=['./data/preprocessed/trainset/search.train.json'],
                               default=[
                                   '../data/data_train_preprocessed.json'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               # default=['./data/demo/devset/search.dev.json'],
                               # default=['./data/preprocessed/devset/search.dev.json',
                               #          './data/preprocessed/devset/zhidao.dev.json'],
                               # default=['./data/preprocessed/devset/search.dev.json'],
                               default=[
                                   '../data/data_dev_preprocessed.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               # default=['./data/demo/testset/search.test.json'],
                               # default=['./data/preprocessed/testset/search.test.json',
                               #          './data/preprocessed/testset/zhidao.test.json'],
                               # default=['./data/preprocessed/testset/search.test.json'],
                               default=[
                                   '../data/data_test_preprocessed.json'],
                               help='list of files that contain the preprocessed test data')
    path_settings.add_argument('--pre_train_file', type=str,
                               default='../data/mrcnew.vec', help='pre_train files')
    path_settings.add_argument('--vocab_dir', default='../data/vocab_search_pretrain/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='../data/models_search_pretrain/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='../data/results_demo/',
                               help='the dir to output the results')

    return parser.parse_args()


def prepare(args):
    """
    checks data, creates the directories, prepare the vocabulary and embeddings
    """
    #--------------------------------
    if args.rm_vocab_path:
        try:
            os.remove(args.rm_vocab_path)
        except FileNotFoundError:
            print('[Warning] File not exsit.')    
            pass

    for data_path in args.train_files + args.dev_files + args.test_files:
        assert os.path.exists(
            data_path), '{} file does not exist.'.format(data_path)

    for dir_path in [args.vocab_dir, args.model_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    brc_data = BRCDataset(args.max_p_num, args.max_p_len,
                          args.max_q_len, args.train_files)
    vocab = Vocab(lower=True)
    for word in brc_data.word_iter('train'):
        vocab.add(word)

    # unfiltered_vocab_size = vocab.size()
    print("vocab size is ", vocab.size())
    vocab.filter_tokens_by_cnt(min_cnt=2)
    print("after filtered vocab size is ", vocab.size())
    # filtered_num = unfiltered_vocab_size - vocab.size()

    vocab.randomly_init_embeddings(args.embed_size)
    if args.use_pre_train:
        vocab.load_pretrained_embeddings(args.pre_train_file)

    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)


def train(args, MODE_YESNO=False):
    """
    trains the reading comprehension model
    """
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          args.train_files, args.dev_files)
    brc_data.convert_to_ids(vocab)
    rc_model = RCModel(vocab, args)
    if args.pretrain_model_path:
        rc_model.load_state_dict(torch.load(args.pretrain_model_path))
    rc_model.train(brc_data, args.epochs, args.batch_size, save_dir=args.model_dir,
                   save_prefix=args.algo, MODE_YESNO=MODE_YESNO)


def evaluate(args):
    """
    evaluate the trained model on dev files
    """
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.dev_files) > 0, 'No dev files are provided.'
    brc_data = BRCDataset(args.max_p_num, args.max_p_len,
                          args.max_q_len, dev_files=args.dev_files)
    brc_data.convert_to_ids(vocab)
    rc_model = RCModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo+'_7')
    dev_batches = brc_data.gen_mini_batches('dev', args.batch_size,
                                            pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    bleu_rouge = rc_model.evaluate(dev_batches)
    # bleu_rouge = rc_model.evaluate(dev_batches,
    #                                result_dir=args.result_dir, result_prefix='dev.predicted')


def predict(args):
    """
    predicts answers for test files
    """
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.test_files) > 0, 'No test files are provided.'
    brc_data = BRCDataset(args.max_p_num, args.max_p_len, args.max_q_len,
                          test_files=args.test_files)
    brc_data.convert_to_ids(vocab)
    rc_model = RCModel(vocab, args)
    model_name = args.algo + '_{}'.format(args.epochs)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo + '_{}'.format(args.epochs))
    test_batches = brc_data.gen_mini_batches('test', args.batch_size,
                                             pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    rc_model.evaluate(test_batches,
                      result_dir=args.result_dir, result_prefix='dev.predicted.' + model_name)


def run():
    """
    Prepares and runs the whole system.
    """
    time1 = datetime.datetime.now()
    print('[START TIME] ',time1 )
    logging.basicConfig(level=logging.DEBUG,
                        filename='./LSTM.log',
                        filemode='a+',
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    args = parse_args()
    # prepare(args)
    #print("\033[0;30;46m WHY Info: Prepare complete. \033[0m ")
    
    # train(args, MODE_YESNO=False)
    #print("\033[0;30;46m WHY Info: Data train complete. \033[0m ")
    #evaluate(args)
    #print("\033[0;30;46m WHY Info: Dev data evaluate complete. \033[0m ")
    predict(args)
    print("\033[0;30;46m WHY Info: Test data predict complete. Everything done. \033[0m ")
    time2 = datetime.datetime.now()
    print('[END TIME] ', time2, '\n[TOTAL USE] ', (time2-time1).seconds/60, 'min')


def gen_yesno_vec():
    args = parse_args()
    prepare(args)
    train(args, MODE_YESNO=True)



if __name__ == '__main__':
    run()
