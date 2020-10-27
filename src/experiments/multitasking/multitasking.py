import numpy as np
import pickle
import os
import sys
sys.path.append("../../..")
sys.path.append("../..")
sys.path.append("..")
import utils
import random
import math

from har_code.readers import MultitaskReader
from har_code.mimic3models.multitask import utils as mt_utils
from har_code.readers import MultitaskReader
from har_code.mimic3models.preprocessing import Discretizer, Normalizer

from text_utils import  avg_emb
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from models.multi_modality_model_hy import TextCNN, LSTMModel, ChannelWiseLSTM, \
     TabularEmbedding, MultiModalMultiTaskWrapper, FCTaskComponent, \
     MultiModalEncoder
from models.loss import masked_weighted_cross_entropy_loss, masked_mse_loss
from dataloaders import MultiModal_Dataset, custom_collate_fn
import functools
import json
from tqdm import tqdm
from sklearn import metrics
from utils import BootStrap, BootStrapDecomp, BootStrapLos, BootStrapIhm, BootStrapPheno, BootStrapLtm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D      

import torch.autograd.profiler as profiler

#-----------------------Data locations ---------------------------------------------$
conf = utils.get_config()
args = utils.get_args()
vectors, w2i_lookup = utils.get_embedding_dict(conf)
#Note that some more paths are in conf
#TODO: Move all path defentions here
if conf.padding_type == 'Zero':
    vectors[utils.lookup(w2i_lookup, '<pad>')] = 0
train_val_ts_root_dir = '/home/luca/mutiltasking-for-mimic3/data/expanded_multitask/train'
test_ts_root_dir = '/home/luca/mutiltasking-for-mimic3/data/expanded_multitask/test'
train_val_text_root_dir = '/home/luca/mutiltasking-for-mimic3/data/root/train_text_ds/'
test_text_root_dir = '/home/luca/mutiltasking-for-mimic3/data/root/test_text_ds/'
train_val_tab_root_dir = '/home/luca/MultiModal-EHR/data/root/train/'
test_tab_root_dir = '/home/luca/MultiModal-EHR/data/root/test/'
train_listfile = '4k_train_listfile.csv'
val_listfile = '4k_val_listfile.csv'
test_listfile ='test_listfile.csv'
train_val_starttime_path = conf.starttime_path_train_val
test_starttime_path = conf.starttime_path_test

#======================================Hyperparameters======================================#
# decomp_weight = 5.0
# los_weight = 3.0
# ihm_weight = 3.0
# pheno_weight = 2.0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#If we don't care about a task, set it's weight to 0
decomp_weight = 1.0
los_weight = 1.0
ihm_weight = 1.0
pheno_weight = 1.0
readmit_weight = 1.0
ltm_weight = 1.0
target_task = 'los'
task_weight = {
    'decomp': decomp_weight,
    'los':    los_weight,
    'ihm':    ihm_weight,
    'pheno':  pheno_weight,
    'readmit': readmit_weight,
    'ltm': ltm_weight,
}

# class frequency for each task
decomp_class_weight = torch.FloatTensor([1.0214, 47.6688])
decomp_class_weight = decomp_class_weight.to(device)
ihm_class_weight = torch.FloatTensor([1.1565, 7.3888])
ihm_class_weight = ihm_class_weight.to(device)
los_class_weight = torch.FloatTensor([1.6047, 3.8934, 8.3376])
los_class_weight = los_class_weight.to(device)
pheno_class_weight = torch.FloatTensor([19.2544,  55.1893,  40.1445,  12.8604,  30.7595,  31.4979,  19.9768,
        57.2309,  15.4088,  12.8200,  43.2644,  21.3991,  14.2026,   9.8531,
        15.3284,  57.1641,  31.0782,  46.4064,  81.0640, 102.7755,  47.5936,
        29.6070,  22.7682,  28.8175,  52.8856])
pheno_class_weight = pheno_class_weight.to(device)
readmit_class_weight = torch.FloatTensor([1.1932 ,43.8267, 24.2904, 23.4624, 17.9707])
readmit_class_weight = readmit_class_weight.to(device)

# def subsampling(dataset):
#     positive_idx, negative_idx = [],[]
#     for i in range(len(dataset)):
#         if i%1000 == 0:
#             print('finished {}'.format(i))
#         data = dataset[i]
#         if data['readmit label'] == 1:
#             positive_idx.append(i)
#         else:
#             negative_idx.append(i)
#     subset_neg = negative_idx[:len(positive_idx)+500]
#     combined_idx = subset_neg + positive_idx
#     subset = torch.utils.data.Subset(dataset, combined_idx)
#     return subset

version = 1
experiment = 'refactor_experiment'#"channelwise_text_cnn_uniform_task_weights_64_hidden_lstm"
while os.path.exists(os.path.join('runs', experiment+"_v{}".format(version))):
    version += 1
experiment = experiment + "_v{}".format(version)
print("Starting run {}".format(experiment))

writer = SummaryWriter(os.path.join('runs', experiment))
ihm_pos = 48
los_pos = 24
use_ts = True
use_text = True
use_tab = True
decay = 0.1
max_text_length = 120
max_num_notes =  10
regression = False
bin_type = 'coarse'
epochs = 500
learning_rate = 1e-4
batch_size = 1
bootstrap_decomp = BootStrapDecomp(k=1000, experiment_name = experiment)
bootstrap_los = BootStrapLos(experiment_name = experiment)
bootstrap_ihm = BootStrapIhm(experiment_name = experiment)
bootstrap_pheno = BootStrapPheno(experiment_name = experiment)
bootstrap_ltm = BootStrapLtm(experiment_name = experiment)
#bootstrap_readmit = BootStrapReadmit(experiment_name = experiment)


# prepare discretizer and normalizer
conf = utils.get_config()

discretizer = Discretizer(timestep=conf.timestep,
                        store_masks=True,
                        impute_strategy='previous',
                        start_time='zero')
discretizer_header = discretizer.get_header()


cont_channels = [2, 3, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
train_reader = MultitaskReader(dataset_dir=os.path.join(
train_val_ts_root_dir), listfile=os.path.join(conf.multitask_path, 'train','4k_train_listfile.csv'))

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

#------------------------------- Define The Encoders per modality --------------------------# 
#ts_model = ChannelWiseLSTM(preprocess_dim =59, hidden_dim =64, layers =1, header = discretizer_header, bidirectional=False)
ts_model = LSTMModel(input_dim = 160, hidden_dim = 64, layers =1, dropout=0.0, bidirectional= False)
text_model = TextCNN(in_channels=1, out_channels=16, kernel_heights =[2,3,4], embedding_length =200, name ='cnn')
#text_model = Text_RNN(embedding_length =200, hidden_size =32, name = 'rnn')
#text_model = Text_AVG()
#text_model = LSTMAttentionModel(hidden_size =128,embedding_length =200, name = 'lstm attn')


### tab_inputs dict defines what tabular features get used
### Only keys in this dict will be used as inputs to the model
### Values are a tuple of (number_of_classes, embedding_size)
### Embeddings sizes need not be the same
tab_inputs = {
    "careunit": (8, 32), 
    "dbsource": (4, 16), 
#    "ethnicity": (40, 32), 
    "gender": (4, 32), 
    "age": (120, 32), # Need to bin?
    "height": (100, 32), #Need to bin?
    "weight": (40, 32) #Binned by 5 kgs, should change
    }


tab_model =  TabularEmbedding(tab_inputs, device)

#------------------------------- Declare the MultiModal Encoder --------------------------------#

encoder = MultiModalEncoder(ts_model = ts_model, text_model = text_model, tab_model = tab_model)


#------------------------------- Declare every task specific module ---------------------------#
#TODO: Switch the losses from being computed in train to the task specific module
global_dim = encoder.global_dim
decomp_model = FCTaskComponent("decomp", None, global_dim, 2, hidden_size = 128)
ihm_model = FCTaskComponent("ihm", None, global_dim, 2, hidden_size = 128)
los_model = FCTaskComponent("los", None, global_dim, 3, hidden_size = 32)
pheno_model = FCTaskComponent("pheno", None, global_dim, 25, hidden_size = 128)
readmit_model = FCTaskComponent("readmit", None, global_dim, 5, hidden_size = 128)
ltm_model = FCTaskComponent("ltm", None, global_dim, 2, hidden_size = 128)

#model = MultiModal_Multitask_Model(ts_model= ts_model, text_model= text_model, tab_model=tab_model, \
#     use_ts =use_ts, use_text = use_text, use_tab = use_tab)
#-------------------------- The Entire MM-MT model ----------------------------#
model = MultiModalMultiTaskWrapper(mm_encoder = encoder, \
    decomp_model = decomp_model, ihm_model = ihm_model, los_model = los_model,
    pheno_model = pheno_model, readmit_model = readmit_model, ltm_model = ltm_model)
log_var = {
    'decomp' : torch.zeros((1,), requires_grad=True, device=device),
    'ihm'    : torch.zeros((1,), requires_grad=True, device=device),
    'los'    : torch.zeros((1,), requires_grad=True, device=device),
    'pheno'  : torch.zeros((1,), requires_grad=True, device=device),
    'readmit': torch.zeros((1,), requires_grad=True, device=device),
    'ltm': torch.zeros((1,), requires_grad=True, device=device),

}
 
#optimizer = torch.optim.Adam(([p for p in model.parameters()] + [log_var[t] for t in log_var if t != 'readmit']), lr=learning_rate) #for uncertainty weighting
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=0.3)
early_stopper = utils.EarlyStopping(experiment_name = experiment)
embedding_layer = nn.Embedding(vectors.shape[0], vectors.shape[1])
embedding_layer.weight.data.copy_(torch.from_numpy(vectors))
embedding_layer.weight.requires_grad = False


#-------------------------- Define the train/val/test dataloaders ------------#
train_mm_dataset = MultiModal_Dataset(train_val_ts_root_dir, train_val_text_root_dir,train_val_tab_root_dir, train_listfile, discretizer, train_val_starttime_path,\
        regression, bin_type, None, ihm_pos, los_pos,  use_text, use_ts, use_tab, decay, w2i_lookup, max_text_length, max_num_notes)
#train_mm_dataset = subsampling(train_mm_dataset)
#print(len(train_mm_dataset))
val_mm_dataset = MultiModal_Dataset(train_val_ts_root_dir, train_val_text_root_dir, train_val_tab_root_dir, val_listfile, discretizer, train_val_starttime_path,\
        regression, bin_type, None, ihm_pos, los_pos,  use_text, use_ts, use_tab, decay, w2i_lookup, max_text_length, max_num_notes)
test_mm_dataset = MultiModal_Dataset(test_ts_root_dir, test_text_root_dir,test_tab_root_dir, test_listfile, discretizer, test_starttime_path,\
        regression, bin_type, None, ihm_pos, los_pos,  use_text, use_ts, use_tab, decay, w2i_lookup, max_text_length, max_num_notes)





#---------------------------- Load the dataset for each split ---------------#
collate_fn_train = functools.partial(custom_collate_fn, union = True)
collate_fn_val = functools.partial(custom_collate_fn, union = True)
collate_fn_test = functools.partial(custom_collate_fn, union = True)
#train_mm_dataset = torch.utils.data.Subset(train_mm_dataset, list(range(20, 40)))
train_data_loader = torch.utils.data.DataLoader(dataset=train_mm_dataset, 
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=5,
                                            collate_fn = collate_fn_train)
val_data_loader = torch.utils.data.DataLoader(dataset=val_mm_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=5,
                                            collate_fn = collate_fn_val)

test_data_loader = torch.utils.data.DataLoader(dataset=test_mm_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=5,
                                            collate_fn = collate_fn_test)


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        # if(p.requires_grad) and ("bias" not in n):
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    #print('avg grad is {:.5f}'.format(ave_grads))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=1.0) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                ], ['max-gradient', 'mean-gradient'])
    plt.show()




def text_embedding(embedding_layer,data, device):
    """
    Embed the notes using predifed embeddings
    """
    texts = torch.from_numpy(data['texts']).to(torch.int64)
    texts_weight_mat = torch.from_numpy(data['texts weight mat']).float()
    texts_weight_mat = texts_weight_mat.to(device)
    texts = embedding_layer(texts) # [batch_size, num_docs, seq_len, emb_dim]
    texts = texts.to(device)
    if text_model.name == 'avg':
        texts = avg_emb(texts, texts_weight_mat)
    return texts, texts_weight_mat

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
    los_label = torch.from_numpy(np.array(data['los label'])).long()
    los_label = los_label.reshape(-1,1).squeeze(1)
    los_label = los_label.to(device)

    pheno_label = torch.from_numpy(np.array(data['pheno label'])).float()
    pheno_label = pheno_label.to(device)

    readmit_mask = torch.from_numpy(np.array(data['readmit mask']))
    readmit_mask = readmit_mask.to(device)
    readmit_label = torch.from_numpy(np.array(data['readmit label'])).long()
    readmit_label = readmit_label.reshape(-1,1).squeeze(1)
    readmit_label = readmit_label.to(device)

    ltm_mask = torch.from_numpy(np.array(data['ltm mask']))
    ltm_mask = ltm_mask.to(device)
    ltm_label = torch.from_numpy(np.array(data['ltm label'])).long()
    ltm_label = ltm_label.reshape(-1,1).squeeze(1)
    ltm_label = ltm_label.to(device) 
    return decomp_label, decomp_mask, los_label, los_mask, ihm_label, ihm_mask,\
         pheno_label, readmit_label, readmit_mask, ltm_label, ltm_mask, num_valid_data         


def train(epochs, train_data_loader, test_data_loader, early_stopper, model, optimizer, scheduler, device):
    
    criterion = nn.BCEWithLogitsLoss()
    crossentropyloss = nn.CrossEntropyLoss()
    
    
    model.to(device)
    train_b = 0
    
    
    
    for epoch in range(epochs):
       
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 50)
        model.train()
        
        running_loss =0.0
        running_loss_decomp = 0.0
        running_loss_los = 0.0
        running_loss_ihm = 0.0
        running_loss_pheno = 0.0
        running_loss_readmit = 0.0
        running_loss_ltm = 0.0

        epoch_metrics = utils.EpochWriter("Train", regression, experiment)

        tk0 = tqdm(train_data_loader, total=int(len(train_data_loader)))

        for i, data in enumerate(tk0):
            

            decomp_label, decomp_mask, los_label, los_mask, ihm_label, ihm_mask,\
                 pheno_label, readmit_label, readmit_mask, ltm_label, ltm_mask, num_valid_data = retrieve_data(data, device)
            
            if use_ts:
                ts = torch.from_numpy(data['time series'])
                ts = ts.permute(1,0,2).float().to(device)
            else:
                ts = None
          
            
            if use_text:
                texts, texts_weight_mat = text_embedding(embedding_layer, data, device)
            else:
                texts = None
                texts_weight_mat = None
            if use_tab:
                tab_dict = data['tab']
                for cat in tab_dict:
                    tab_dict[cat] = torch.from_numpy(tab_dict[cat]).long().to(device)
            else:
                tab_dict = None
            decomp_logits, los_logits, ihm_logits, pheno_logits, readmit_logits, ltm_logits = model(ts = ts, texts = texts,\
             texts_weight_mat = texts_weight_mat, tab_dict = tab_dict)
            loss_decomp = masked_weighted_cross_entropy_loss(None, decomp_logits, decomp_label, decomp_mask)
            loss_los = masked_weighted_cross_entropy_loss(los_class_weight, los_logits, los_label, los_mask)
            loss_ihm = masked_weighted_cross_entropy_loss(None,ihm_logits, ihm_label, ihm_mask)
            loss_pheno = criterion(pheno_logits, pheno_label)
            loss_readmit = masked_weighted_cross_entropy_loss(None, readmit_logits, readmit_label, readmit_mask)
            loss_ltm = masked_weighted_cross_entropy_loss(None, ltm_logits, ltm_label, ltm_mask)

            losses = {
                'decomp' : loss_decomp,
                'ihm'    : loss_ihm,
                'los'    : loss_los,
                'pheno'  : loss_pheno,
                'readmit': loss_readmit,
                'ltm'    : loss_ltm, 
            }

            loss = 0.0

            for task in losses:
                #-------- uncertainty weighting -----------------#
                #prec = torch.exp(-log_var[task])
                #losses[task] = torch.sum(losses[task] * prec + log_var[task], -1)
                #loss += torch.sum(losses[task] * prec + log_var[task], -1)
                #-------- end uncertainty weighting stuff -------#
 
                loss += losses[task] * task_weight[task]
            
            train_b+=1

            
            
            optimizer.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # if i % 300 == 0:
            #     plot_grad_flow(model.named_parameters())
            
            optimizer.step()

            running_loss += loss.item()
            running_loss_decomp += loss_decomp.item() * task_weight['decomp']
            running_loss_los +=loss_los.item()* task_weight['los']
            running_loss_ihm +=loss_ihm.item()* task_weight['ihm']
            running_loss_pheno += loss_pheno.item()* task_weight['pheno']
            running_loss_readmit += loss_readmit.item()*task_weight['readmit']
            running_loss_ltm += loss_ltm.item()*task_weight['ltm']


            m = nn.Softmax(dim=1)
            sig = nn.Sigmoid()

            decomp_pred = (sig(decomp_logits)[:, 1]).cpu().detach().numpy()
            los_pred = m(los_logits).cpu().detach().numpy()
            ihm_pred = (sig(ihm_logits)[:, 1]).cpu().detach().numpy()
            pheno_pred = sig(pheno_logits).cpu().detach().numpy()
            readmit_pred = m(readmit_logits).cpu().detach().numpy()
            ltm_pred = (sig(ltm_logits)[:,1]).cpu().detach().numpy()
            

            outputs = {
                'decomp': {'pred': decomp_pred,
                           'label': decomp_label.cpu().detach().numpy(),
                           'mask': decomp_mask.cpu().detach().numpy()},
                'ihm': {'pred': ihm_pred,
                           'label': ihm_label.cpu().detach().numpy(),
                           'mask': ihm_mask.cpu().detach().numpy()},
                'los': {'pred': los_pred,
                           'label': los_label.cpu().detach().numpy(),
                           'mask': los_mask.cpu().detach().numpy()},
                'pheno': {'pred': pheno_pred,
                           'label': pheno_label.cpu().detach().numpy(),
                           'mask': None},
                'readmit': {'pred': readmit_pred,
                            'label':readmit_label.cpu().detach().numpy(),
                            'mask': readmit_mask.cpu().detach().numpy()},
                'ltm': {'pred': ltm_pred,
                        'label': ltm_label.cpu().detach().numpy(),
                        'mask': ltm_mask.cpu().detach().numpy()},
            }
            epoch_metrics.cache(outputs, num_valid_data)
            interval = 500
            

            if i %interval == interval-1:
                
                writer.add_scalar('training loss',
                            running_loss/(interval -1),
                            train_b)
                writer.add_scalar('decomp loss',
                            running_loss_decomp / (interval -1),
                            train_b)
                writer.add_scalar('los loss',
                            running_loss_los / (interval -1),
                            train_b)
                writer.add_scalar('ihm loss',
                            running_loss_ihm / (interval -1),
                            train_b)
                writer.add_scalar('pheno loss',
                            running_loss_pheno / (interval-1),
                            train_b)
                writer.add_scalar('readmit loss',
                            running_loss_readmit / (interval -1),
                            train_b)

                writer.add_scalar('ltm loss',
                            running_loss_ltm / (interval-1),
                            train_b)
                
                
                running_loss_decomp = 0.0
                running_loss_los = 0.0
                running_loss_ihm = 0.0
                running_loss_pheno = 0.0
                running_loss_readmit = 0.0
                running_loss_ltm = 0.0
                running_loss = 0.0
                epoch_metrics.add()

        epoch_metrics.write(writer,train_b)


        #scheduler.step()
        evaluate(epoch, val_data_loader, model, 'val', early_stopper, device, train_b)
        #evaluate(epoch, test_data_loader, model, 'test', early_stopper, device, train_b)
        
       
        if early_stopper.early_stop:
            evaluate(epoch, test_data_loader, model, 'test', early_stopper, device, train_b)
        #     #bootstrap_ltm.get()
        #     print("Early stopping")
        #     break
    

def evaluate(epoch, data_loader, model, split, early_stopper, device,  train_step=None):
    
    if split == 'val':
        epoch_metrics = utils.EpochWriter("Val", regression, experiment)

    else:
        epoch_metrics = utils.EpochWriter("Test", regression, experiment)
    '''
    if split == 'test':
        if os.path.exists('./ckpt/{}.ckpt'.format(experiment)):
            model.load_state_dict(torch.load('./ckpt/{}.ckpt'.format(experiment)))
        model.to(device)

    '''


    model.eval()
    

    running_loss = 0.0
    running_loss_decomp = 0.0
    running_loss_los = 0.0
    running_loss_ihm = 0.0
    running_loss_pheno = 0.0
    running_loss_readmit = 0.0
    running_loss_ltm = 0.0
    tk = tqdm(data_loader, total=int(len(data_loader)))
    criterion = nn.BCEWithLogitsLoss()

   
    for i, data in enumerate(tk):

        if data is None:
                continue
        
        decomp_label, decomp_mask, los_label, los_mask, ihm_label, ihm_mask,\
                 pheno_label, readmit_label, readmit_mask, ltm_label, ltm_mask, num_valid_data = retrieve_data(data, device)
        
        if use_ts:
                ts = torch.from_numpy(data['time series'])
                ts = ts.permute(1,0,2).float().to(device)
        else:
            ts = None

        
        if use_text:
            texts, texts_weight_mat = text_embedding(embedding_layer, data, device)
        else:
            texts = None
            texts_weight_mat = None
        
        if use_tab:
            tab_dict = data['tab']
            for cat in tab_dict:
                tab_dict[cat] = torch.from_numpy(tab_dict[cat]).long().to(device)
        else:
            tab_dict = None

        decomp_logits, los_logits, ihm_logits, pheno_logits, readmit_logits, ltm_logits = model(ts = ts, texts = texts, texts_weight_mat = texts_weight_mat,\
            tab_dict = tab_dict
        )

        loss_decomp = masked_weighted_cross_entropy_loss(None, 
                                                        decomp_logits, 
                                                        decomp_label, 
                                                        decomp_mask)
        loss_los = masked_weighted_cross_entropy_loss(los_class_weight, 
                                                      los_logits,
                                                      los_label, 
                                                      los_mask)

        loss_ihm = masked_weighted_cross_entropy_loss(None,ihm_logits, 
                                                      ihm_label, 
                                                      ihm_mask)
        loss_pheno = criterion(pheno_logits, pheno_label)
        loss_readmit = masked_weighted_cross_entropy_loss(readmit_class_weight, readmit_logits, readmit_label, readmit_mask)
        loss_ltm = masked_weighted_cross_entropy_loss(None, ltm_logits, ltm_label, ltm_mask)
        
        losses = {
            'decomp' : loss_decomp,
            'ihm'    : loss_ihm,
            'los'    : loss_los,
            'pheno'  : loss_pheno,
            'readmit': loss_readmit,
            'ltm'    : loss_ltm,
        }
        loss = 0.0

        
        for task in losses:
            loss += losses[task] * task_weight[task]
    
        running_loss += losses[target_task] * task_weight[target_task]
        running_loss_decomp += loss_decomp.item() * task_weight['decomp']
        running_loss_los +=loss_los.item()* task_weight['los']
        running_loss_ihm +=loss_ihm.item()* task_weight['ihm']
        running_loss_pheno += loss_pheno.item()* task_weight['pheno']
        running_loss_readmit += loss_readmit.item()*task_weight['readmit']
        running_loss_ltm += loss_ltm.item()*task_weight['ltm']


        m = nn.Softmax(dim=1)
        sigmoid = nn.Sigmoid()
        
        decomp_pred = (sigmoid(decomp_logits)[:,1]).cpu().detach().numpy()
        los_pred = m(los_logits).cpu().detach().numpy()
        ihm_pred = (sigmoid(ihm_logits)[:,1]).cpu().detach().numpy()
        pheno_pred = sigmoid(pheno_logits).cpu().detach().numpy()
        readmit_pred = m(readmit_logits).cpu().detach().numpy()
        ltm_pred = (sigmoid(ltm_logits)[:,1]).cpu().detach().numpy()
        #print(sigmoid(readmit_logits))

        outputs = {
            'decomp': {'pred': decomp_pred,
                        'label': decomp_label.cpu().detach().numpy(),
                        'mask': decomp_mask.cpu().detach().numpy()},
            'ihm': {'pred': ihm_pred,
                        'label': ihm_label.cpu().detach().numpy(),
                        'mask': ihm_mask.cpu().detach().numpy()},
            'los': {'pred': los_pred,
                        'label': los_label.cpu().detach().numpy(),
                        'mask': los_mask.cpu().detach().numpy()},
            'pheno': {'pred': pheno_pred,
                        'label': pheno_label.cpu().detach().numpy(),
                        'mask': None},
            'readmit': {'pred': readmit_pred,
                            'label':readmit_label.cpu().detach().numpy(),
                            'mask': readmit_mask.cpu().detach().numpy()},
            'ltm': {'pred': ltm_pred,
                        'label': ltm_label.cpu().detach().numpy(),
                        'mask': ltm_mask.cpu().detach().numpy()},
            
        }

        epoch_metrics.cache(outputs, num_valid_data)
   


    if train_step is not None:
        xpoint = train_step
    else:
        xpoint = epoch+1


    epoch_metrics.write(writer, xpoint)
    writer.add_scalar('{} loss'.format(split),
                running_loss/(i),
                xpoint)
    writer.add_scalar('{} decomp loss'.format(split),
                running_loss_decomp / (i),
                xpoint)
    writer.add_scalar('{} los loss'.format(split),
                running_loss_los / (i),
                xpoint)
    writer.add_scalar('{} ihm loss'.format(split),
                running_loss_ihm / (i),
                xpoint)
    writer.add_scalar('{} pheno loss'.format(split),
                running_loss_pheno / (i),
                xpoint)
    writer.add_scalar('{} readmit loss'.format(split),
                running_loss_readmit/ (i),
                xpoint)
    writer.add_scalar('{} ltm loss'.format(split),
                running_loss_ltm/ (i),
                xpoint)

    if split == 'val':
        early_stopper(running_loss/(i), model)
 

train(epochs, train_data_loader,  test_data_loader, early_stopper, model, optimizer, scheduler, device)
#bootstrap_pheno.get()
#evaluate(0, test_data_loader, model, 'test', early_stopper, device, None)
#bootstrap_los.get()