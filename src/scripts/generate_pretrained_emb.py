import numpy as np 
import sys
import os
from models.text_only_model import Text_Model
from models.ts_only_model import TS_Ihm_Model
import models.utils as utils
import argparse
import torch
import torch.nn as nn


# parser = argparse.ArgumentParser()
# parser.add_argument('--max_seqlen', default = 500, type=int, 
#                     help='The max seq length of a single doc')
# parser.add_argument('--max_numdocs', default = 50, type=int, 
#                     help='The max number of notes in one batch of icu stay, one icu stay usually contain multiple notes')
# parser.add_argument('--ckpt_path', default ='./ckpt')


# model
lstm = LSTMModel(input_dim = 76, hidden_dim = 128, layers=1, bidirectional = True)
ts_model = TS_Ihm_Model(lstm, bidirectional=True)


def generate_idx_matrix(patient_text_list, w2i_lookup, max_seqlen):
    """
    args: 
    patient_text_list: a 2d list of raw text tokens, in shape [batch_size, num_notes]
    w2i_lookip: the lookup table

    return: a numpy matrix contains all the word idx
    """
    patient_list_of_indices = []
    max_seqlen_found = -1 # this is the actual max seqlen we found in a batch
    max_numnotes = -1
    num_notes = []

    for patient_text in patient_text_list:
        # each patient_text is a list of text
        list_of_indices = []
        num_notes.append(len(patient_text))
        for sentence in patient_text:
            # each sentence is a list of word
            indices = list(map(lambda x: lookup(
                w2i_lookup, x), str(sentence).split()))
            if max_seqlen > 0:
                indices = indices[:max_seqlen]
            list_of_indices.append(indices)
            max_seqlen_found = max(len(indices), max_seqlen_found)
        patient_list_of_indices.append(list_of_indices)
        max_numnotes = max(len(list_of_indices), max_numnotes)

    pad_token = w2i_lookup['<pad>']

    def extra_pad_tokens(cnt): return [pad_token]*cnt

    padded_patient_list_of_indices = []

    for pt in patient_list_of_indices:
        padded_pt = []
        if len(pt) < max_numnotes:
            pt = pt + [[]]*(max_numnotes-len(pt))
        for note in pt:
            note = note + extra_pad_tokens(max_seqlen - len(note))
            padded_pt.append(note)
        padded_patient_list_of_indices.append(padded_pt)

    x = np.array(padded_patient_list_of_indices)
    n = np.array(num_notes)

    assert len(x.shape) == 3 # [batch_size, max_numnotes, max_seqlen]
    assert x.shape[0] == n.shape[0]
    assert x.shape[0] == len(patient_text_list), "x: {}, l: {}".format(
        str(x.shape), str(len(patient_text_list)))
    return x, l


def generate_emb_tensor(vectors, data_loader, ts_model, text_model, ts_model_ckpt, text_model_ckpt, device):
    """
    args:
    vectors: pretrained emb

    """
    if os.path.exists(ts_model_ckpt):
            model.load_state_dict(torch.load(ts_model_ckpt))
    if os.path.exists(text_model_ckpt):
            model.load_state_dict(torch.load(text_model_ckpt))

    embedding_layer = nn.Embedding(vectors.shape[0], vectors.shape[1])
    embedding_layer.weight.data.copy_(torch.from_numpy(vectors))
    embedding_layer.weight.requires_grad = False

    for i, batch in enumerate(data_loader):
        texts =  torch.from_numpy(batch['Texts']) 
        ts = torch.from_numpy(batch['ts'])

        ts_emb = ts_model.lstm_model(X)(ts)
        
        texts = embedding_layer(texts).to(device)
        text_emb = text_model.cnn_model(text)

        ### your code ###
    
    return




