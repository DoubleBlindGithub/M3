import numpy as np
import pickle
import tensorflow as tf
import os
import sys
sys.path.append("../../..")
sys.path.append("../..")
sys.path.append("..")
import utils
import random
import math
from mimic3models.multitask import utils as mt_utils
from mimic3benchmark.readers import MultitaskReader
from waveform.WaveformLoader import WaveformDataset
from mimic3models.preprocessing import Discretizer, Normalizer
from text_utils import  avg_emb
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from models.multi_modality_model_hy import Text_CNN, Text_RNN,LSTMModel, ChannelWiseLSTM, \
     Waveform_Pretrained, MultiModal_Multitask_Model_4_task, Text_AVG
from models.loss import masked_weighted_cross_entropy_loss, get_class_weight, masked_mse_loss
from dataloaders import MultiModal_Dataset, custom_collate_fn
import functools
import json
from tqdm import tqdm


conf = utils.get_config()
args = utils.get_args()
vectors, w2i_lookup = utils.get_embedding_dict(conf)

if conf.padding_type == 'Zero':
    vectors[utils.lookup(w2i_lookup, '<pad>')] = 0
train_ts_root_dir = '/home/yong/mutiltasking-for-mimic3/data/multitask_updated/train'
test_ts_root_dir = '/home/yong/mutiltasking-for-mimic3/data/multitask_updated/test'
train_text_root_dir = '/home/yong/mutiltasking-for-mimic3/data/root/text_fixed_train/'
test_text_root_dir = '/home/yong/mutiltasking-for-mimic3/data/root/text_fixed_test/'
wf_root_dir = '/home/yong/mutiltasking-for-mimic3/data/waveform_embeddings'
ihm_pos = 48
los_pos = 24
use_ts = True
use_wf = False
use_text = True
wf_dim = 100
decay = 0.1
max_text_length = 500
max_num_notes = 150
regression = False
bin_type = 'coarse'
train_starttime_path = conf.starttime_path_train
test_starttime_path = conf.starttime_path_test
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 5
#ts_model = LSTMModel(input_dim = 76, hidden_dim = 256, layers=2)


# prepare discretizer and normalizer
conf = utils.get_config()
train_reader = MultitaskReader(dataset_dir=os.path.join(
conf.multitask_path, 'train'), listfile=os.path.join(conf.multitask_path, 'train','listfile.csv'))
test_reader = MultitaskReader(dataset_dir=os.path.join(
    conf.multitask_path, 'test'), listfile=os.path.join(conf.multitask_path, 'test','listfile.csv'))
discretizer = Discretizer(timestep=conf.timestep,
                        store_masks=True,
                        impute_strategy='previous',
                        start_time='zero')
discretizer_header = discretizer.transform(
    train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(
    discretizer_header) if x.find("->") == -1]
normalizer = Normalizer(fields=cont_channels)
normalizer_state = conf.normalizer_state
if normalizer_state is None:
    normalizer_state = 'mult_ts{}.input_str:previous.start_time:zero.n5e4.normalizer'.format(
        conf.timestep)
normalizer.load_params(normalizer_state)



train_mm_dataset = MultiModal_Dataset(train_ts_root_dir, train_text_root_dir, wf_root_dir, discretizer, train_starttime_path,\
        regression, bin_type, normalizer, ihm_pos, los_pos,  use_wf, use_text, use_ts, wf_dim, decay, w2i_lookup, max_text_length, max_num_notes)
test_mm_dataset = MultiModal_Dataset(test_ts_root_dir, test_text_root_dir, wf_root_dir, discretizer, test_starttime_path,\
        regression, bin_type, normalizer, ihm_pos, los_pos,  use_wf, use_text, use_ts, wf_dim, decay, w2i_lookup, max_text_length, max_num_notes)

collate_fn_train = functools.partial(custom_collate_fn, union = True)
collate_fn_test = functools.partial(custom_collate_fn, union = True)
train_data_loader = torch.utils.data.DataLoader(dataset=train_mm_dataset, 
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=5,
                                            collate_fn = collate_fn_train)

test_data_loader = torch.utils.data.DataLoader(dataset=test_mm_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=5,
                                            collate_fn = collate_fn_test)



def n_gram_stats(dataloader, topk, n_gram = 1):
    vocab_1, vocab_2, vocab_3 = {}, {}, {}

    for i, data in enumerate(dataloader):
        los_mask = data['los mask']
        los_label = data['los label']
        raw_text = data['raw text']
        for j in range(len(los_label)):
            if los_mask[j] == 0:
                continue
            valid_docs = [texts[1] for texts in raw_text[j] if texts[0]<=24]
            for docs in valid_docs:
                for word in docs:
                    if los_label[j] == 0:
                        if word not in vocab_1:
                            vocab_1[word] =1
                        else:
                            vocab_1[word]+=1
                    elif los_label[j] == 1:
                        if word not in vocab_2:
                            vocab_2[word] =1
                        else:
                            vocab_2[word]+=1
                    else:
                        if word not in vocab_3:
                            vocab_3[word] =1
                        else:
                            vocab_3[word]+=1
    
    topk_1 = sorted(vocab1.items(), key=lambda x: x[1], reverse=True)[:topk]
    topk_2 = sorted(vocab_2.items(), key=lambda x: x[1], reverse=True)[:topk]
    topk_3 = sorted(vocab_3.items(), key=lambda x: x[1], reverse=True)[:top3]
        
    return topk_1, topk_2, topk_3





    

    


