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
from waveform.WaveformLoader import WaveformDataset
from mimic3models.preprocessing import Discretizer, Normalizer
from text_utils import  avg_emb
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from models.multi_modality_model_hy import Text_CNN, Text_RNN,LSTMModel, ChannelWiseLSTM, \
     Waveform_Pretrained, Text_Only_DS, Text_AVG, LSTMAttentionModel
from models.loss import masked_weighted_cross_entropy_loss, masked_mse_loss
from readmit_dataloaders import MultiModal_Dataset, custom_collate_fn
import functools
import json
from tqdm import tqdm
from sklearn import metrics
from utils import BootStrap, BootStrapDecomp, BootStrapLos, BootStrapIhm, BootStrapPheno, BootStrapReadmit

#======================================Hyperparameters======================================#
# decomp_weight = 5.0
# los_weight = 3.0
# ihm_weight = 3.0
# pheno_weight = 2.0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# class frequency for each task

readmit_class_weight = torch.FloatTensor([1.0599 ,17.5807])
readmit_class_weight = readmit_class_weight.to(device)




version = 1
experiment = "readmit_dev"
while os.path.exists(os.path.join('runs', experiment+"_v{}".format(version))):
    version += 1
experiment = experiment + "_v{}".format(version)
print("Starting run {}".format(experiment))

writer = SummaryWriter(os.path.join('runs', experiment))
conf = utils.get_config()
args = utils.get_args()
vectors, w2i_lookup = utils.get_embedding_dict(conf)

if conf.padding_type == 'Zero':
    vectors[utils.lookup(w2i_lookup, '<pad>')] = 0
train_val_ts_root_dir = '/home/yong/mutiltasking-for-mimic3/data/multitask_2/train'
test_ts_root_dir = '/home/yong/mutiltasking-for-mimic3/data/multitask_2/test'
train_val_text_root_dir = '/home/yong/mutiltasking-for-mimic3/data/root_2/train_text_ds'
test_text_root_dir = '/home/yong/mutiltasking-for-mimic3/data/root_2/test_text_ds'
train_listfile = 'listfile.csv'
val_listfile = '4k_val_listfile.csv'
test_listfile ='listfile.csv'
ihm_pos = 48
los_pos = 24
use_ts = False
use_text = True
decay = 0.1
max_text_length = 500
max_num_notes = 1
regression = False
discharge_summary_only = False
bin_type = 'coarse'
train_val_starttime_path = conf.starttime_path_train_val
test_starttime_path = conf.starttime_path_test

epochs = 50
learning_rate = 3e-4
batch_size = 8
bootstrap_decomp = BootStrapDecomp(k=1000, experiment_name = experiment)
bootstrap_los = BootStrapLos(experiment_name = experiment)
bootstrap_ihm = BootStrapIhm(experiment_name = experiment)
bootstrap_pheno = BootStrapPheno(experiment_name = experiment)
bootstrap_readmit = BootStrapReadmit(experiment_name = experiment)


# prepare discretizer and normalizer
conf = utils.get_config()

discretizer = Discretizer(timestep=conf.timestep,
                        store_masks=True,
                        impute_strategy='previous',
                        start_time='zero')



cont_channels = [2, 3, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]

normalizer = Normalizer(fields=cont_channels)
normalizer_state = conf.normalizer_state
if normalizer_state is None:
    normalizer_state = 'mult_ts{}.input_str:previous.start_time:zero.n5e4.normalizer'.format(
        conf.timestep)
normalizer.load_params(normalizer_state)

# Model

text_model = Text_CNN(in_channels=1, out_channels=128, kernel_heights =[2,3,4], embedding_length =200, name ='cnn')
#text_model = Text_RNN(embedding_length =200, hidden_size =32, name = 'rnn')
#text_model = Text_AVG()
#text_model = LSTMAttentionModel(hidden_size =128,embedding_length =200, name = 'lstm attn')

model = Text_Only_DS(text_model= text_model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=0.3)
early_stopper = utils.EarlyStopping(experiment_name = experiment)
embedding_layer = nn.Embedding(vectors.shape[0], vectors.shape[1])
embedding_layer.weight.data.copy_(torch.from_numpy(vectors))
embedding_layer.weight.requires_grad = False


train_mm_dataset = MultiModal_Dataset(train_val_ts_root_dir, train_val_text_root_dir, train_listfile, discretizer, train_val_starttime_path,\
        regression, bin_type, normalizer, ihm_pos, los_pos,  use_text, use_ts, decay, w2i_lookup, max_text_length, max_num_notes, discharge_summary_only, True)
#train_mm_dataset = subsampling(train_mm_dataset)
#print(len(train_mm_dataset))
# val_mm_dataset = MultiModal_Dataset(train_val_ts_root_dir, train_val_text_root_dir, wf_root_dir, val_listfile, discretizer, train_val_starttime_path,\
#         regression, bin_type, normalizer, ihm_pos, los_pos,  use_wf, use_text, use_ts, wf_dim, decay, w2i_lookup, max_text_length, max_num_notes)
test_mm_dataset = MultiModal_Dataset(test_ts_root_dir, test_text_root_dir,test_listfile, discretizer, test_starttime_path,\
        regression, bin_type, normalizer, ihm_pos, los_pos,  use_text, use_ts, decay, w2i_lookup, max_text_length, max_num_notes, discharge_summary_only, True)

collate_fn_train = functools.partial(custom_collate_fn)
collate_fn_val = functools.partial(custom_collate_fn)
collate_fn_test = functools.partial(custom_collate_fn)
train_data_loader = torch.utils.data.DataLoader(dataset=train_mm_dataset, 
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=5,
                                            collate_fn = collate_fn_train)
# val_data_loader = torch.utils.data.DataLoader(dataset=val_mm_dataset, 
#                                             batch_size=batch_size,
#                                             shuffle=True,
#                                             num_workers=5,
#                                             collate_fn = collate_fn_val)

test_data_loader = torch.utils.data.DataLoader(dataset=test_mm_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=5,
                                            collate_fn = collate_fn_test)

def text_embedding(embedding_layer,data, device):
    
    texts = torch.from_numpy(data['texts']).to(torch.int64)
    texts = embedding_layer(texts) # [batch_size, num_docs, seq_len, emb_dim]
    texts = texts.to(device)
    if text_model.name == 'avg':
        texts = avg_emb(texts, texts_weight_mat)
    return texts

def retrieve_data(data, device):
    """
    retrieve data from data loaders and reorganize its shape and obejects into desired forms
    """
    ihm_mask = torch.from_numpy(np.array(data['ihm mask']))
    ihm_mask = ihm_mask.to(device)
    ihm_label = torch.from_numpy(np.array(data['ihm label'])).long()
    ihm_label = ihm_label.reshape(-1,1).squeeze(1)
    ihm_label = ihm_label.to(device)

    decomp_mask = torch.from_numpy(data['decomp mask'])
    decomp_mask = decomp_mask.to(device)
    decomp_label = torch.from_numpy(data['decomp label']).long()
    # the num valid data is used in case the last batch is smaller than batch size
    num_valid_data = decomp_label.shape[0]
    decomp_label = decomp_label.reshape(-1,1).squeeze(1) # (b*t,)
    decomp_label = decomp_label.to(device)

    los_mask = torch.from_numpy(np.array(data['los mask']))
    los_mask = los_mask.to(device)
    los_label = torch.from_numpy(np.array(data['los label']))
    los_label = los_label.reshape(-1,1).squeeze(1)
    los_label = los_label.to(device)

    pheno_label = torch.from_numpy(np.array(data['pheno label'])).float()
    pheno_label = pheno_label.to(device)

    readmit_mask = torch.from_numpy(np.array(data['readmit mask']))
    readmit_mask = readmit_mask.to(device)
    readmit_label = torch.from_numpy(np.array(data['readmit label']))
    readmit_label = readmit_label.reshape(-1,1).squeeze(1).long()
    readmit_label = readmit_label.to(device) 
    return decomp_label, decomp_mask, los_label, los_mask, ihm_label, ihm_mask, pheno_label, readmit_label, readmit_mask, num_valid_data         


def train(epochs, train_data_loader, test_data_loader, early_stopper, model, optimizer, scheduler, device):
    
    criterion = nn.BCEWithLogitsLoss()
    aucroc_readmit = utils.AUCROCREADMIT()
    aucpr_readmit = utils.AUCPRREADMIT()
    cfm_readmit = utils.ConfusionMatrixReadmit()
    
    model.to(device)
    train_b = 0
    

    
    for epoch in range(epochs):
       
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 50)
        model.train()
        
        running_loss =0.0
        

        epoch_metrics = utils.EpochWriter("Train", regression, experiment)

        tk0 = tqdm(train_data_loader, total=int(len(train_data_loader)))

        for i, data in enumerate(tk0):
            if data is None:
                continue
            decomp_label, decomp_mask, los_label, los_mask, ihm_label, ihm_mask,\
                 pheno_label, readmit_label, readmit_mask, num_valid_data = retrieve_data(data, device)
            
          
            if use_text:
                texts = text_embedding(embedding_layer, data, device)
            else:
                texts = None
                

            
            readmit_logits = model(texts = texts)
            
            loss = masked_weighted_cross_entropy_loss(None, readmit_logits, readmit_label, readmit_mask )
            
            train_b+=1
        
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
           

            m = nn.Softmax(dim=1)
            sig = nn.Sigmoid()

            readmit_pred = (sig(readmit_logits)[:,1]).cpu().detach().numpy()
            readmit_label = readmit_label.cpu().detach().numpy()
            if readmit_label is None:
                print('bad')
            readmit_mask = readmit_mask.cpu().detach().numpy()
            aucpr_readmit.add(readmit_pred, readmit_label, readmit_mask)
            aucroc_readmit.add(readmit_pred, readmit_label, readmit_mask)
            cfm_readmit.add(readmit_pred, readmit_label, readmit_mask)
            interval = 50
            

            if i %interval == interval-1:
                writer.add_scalar('training loss',
                            running_loss/(interval-1),
                            train_b)
                
        print('readmission aucpr is {}'.format(aucpr_readmit.get()))
        print('readmission aucroc is {}'.format(aucroc_readmit.get()))
        print('readmission cfm is {}'.format(cfm_readmit.get()))



        #scheduler.step()
        #evaluate(epoch, val_data_loader, model, 'val', early_stopper, device, train_b)
        evaluate(epoch, test_data_loader, model, 'test', early_stopper, device, train_b)
        
       
        # if early_stopper.early_stop:
        #     evaluate(epoch, test_data_loader, model, 'test', early_stopper, device, train_b)
        #     bootstrap_pheno.get()
        #     print("Early stopping")
        #     break
    

def evaluate(epoch, data_loader, model, split, early_stopper, device,  train_step=None):

    aucroc_readmit = utils.AUCROCREADMIT()
    aucpr_readmit = utils.AUCPRREADMIT()
    cfm_readmit = utils.ConfusionMatrixReadmit()
    
    if split == 'val':
        epoch_metrics = utils.EpochWriter("Val", regression, experiment)

    else:
        epoch_metrics = utils.EpochWriter("Test", regression, experiment)

    
    model.to(device)
    model.eval()
    

    running_loss = 0.0
    
    tk = tqdm(data_loader, total=int(len(data_loader)))
    criterion = nn.BCEWithLogitsLoss()

   
    for i, data in enumerate(tk):
        if data is None:
            continue
        
        
        decomp_label, decomp_mask, los_label, los_mask, ihm_label, ihm_mask,\
                 pheno_label, readmit_label, readmit_mask, num_valid_data = retrieve_data(data, device)
        
        if use_ts:
                ts = torch.from_numpy(data['time series'])
                ts = ts.permute(1,0,2).float().to(device)
        else:
            ts = None

        
        if use_text:
            texts = text_embedding(embedding_layer, data, device)
        else:
            texts = None
        
        readmit_logits = model(texts = texts)

        
        loss = masked_weighted_cross_entropy_loss(None, readmit_logits, readmit_label, readmit_mask)
        
        running_loss += loss.item()

        sigmoid = nn.Sigmoid()
        readmit_pred = (sigmoid(readmit_logits)[:,1]).cpu().detach().numpy()
        readmit_label = readmit_label.cpu().detach().numpy()
        readmit_mask = readmit_mask.cpu().detach().numpy()

        aucpr_readmit.add(readmit_pred, readmit_label, readmit_mask)
        aucroc_readmit.add(readmit_pred, readmit_label, readmit_mask)
        cfm_readmit.add(readmit_pred, readmit_label, readmit_mask)
            
            
                
    print('readmission aucpr is {}'.format(aucpr_readmit.get()))
    print('readmission aucroc is {}'.format(aucroc_readmit.get()))
    print('readmission cfm is {}'.format(cfm_readmit.get()))


    if train_step is not None:
        xpoint = train_step
    else:
        xpoint = epoch+1


    writer.add_scalar('{} readmit loss'.format(split),
                running_loss/ (i),
                xpoint)

    if split == 'val':
        early_stopper(running_loss_pheno/(i), model)
 

train(epochs, train_data_loader,  test_data_loader, early_stopper, model, optimizer, scheduler, device)
#bootstrap_pheno.get()
evaluate(0, test_data_loader, model, 'test', early_stopper, device, None)
#bootstrap_los.get()