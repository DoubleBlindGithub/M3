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

decomp_weight = 1.0#4.0
los_weight = 1.5#0.3
ihm_weight = 0.2#2.5
pheno_weight = 1.0#1.0
task_weight = {
    'decomp': decomp_weight,
    'los':    los_weight,
    'ihm':    ihm_weight,
    'pheno':  pheno_weight
}


version = 1
#experiment = "Check_channelwise_lstm_no_text_ihm_singletask_learned_weighted_loss_lr_1e-4_larger_hidden"
#experiment = 'channelwise_lstm_ts_text_cnn_learned_weights_multitask_lr1e-4_16_hidden'
experiment = 'dev_discretizer'
while os.path.exists(os.path.join('new_runs', experiment+"_v{}".format(version))):
    version += 1
experiment = experiment + "_v{}".format(version)
print("Starting run {}".format(experiment))

writer = SummaryWriter(os.path.join('new_runs', experiment))
model_weights = experiment + '.ckpt'

conf = utils.get_config()
args = utils.get_args()
vectors, w2i_lookup = utils.get_embedding_dict(conf)

#print(mean_vector)
if conf.padding_type == 'Zero':
    vectors[utils.lookup(w2i_lookup, '<pad>')] = 0
train_ts_root_dir = '/home/luca/mutiltasking-for-mimic3/data/expanded_multitask/train'
test_ts_root_dir = '/home/luca/mutiltasking-for-mimic3/data/expanded_multitask/test'
train_text_root_dir = '/home/luca/mutiltasking-for-mimic3/data/root/text_fixed_train/'
test_text_root_dir = '/home/luca/mutiltasking-for-mimic3/data/root/text_fixed_test/'
wf_root_dir = '/home/luca/mutiltasking-for-mimic3/data/waveform_embeddings'
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
epochs = 40
learning_rate = 1e-4#1e-4 for decomp
batch_size = 5


# prepare discretizer and normalizer
conf = utils.get_config()
train_reader = MultitaskReader(dataset_dir=os.path.join(
train_ts_root_dir), listfile=os.path.join(conf.multitask_path, 'train','listfile.csv'))
test_reader = MultitaskReader(dataset_dir=os.path.join(
    test_ts_root_dir), listfile=os.path.join(conf.multitask_path, 'test','listfile.csv'))
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

#ts_model = ChannelWiseLSTM(preprocess_dim =59, hidden_dim = 16 , layers =1, header = discretizer_header, bidirectional=False)
ts_model = LSTMModel(input_dim = 160, hidden_dim= 64, layers =1)
text_model = Text_CNN(in_channels=1, out_channels=12, kernel_heights =[2,3,4], embedding_length =200, name ='cnn')
#text_model = Text_RNN(embedding_length =200, hidden_size =256, name = 'rnn')
#text_model = Text_AVG()
#text_model = LSTMAttentionModel(hidden_size =128,embedding_length =200, name = 'lstm attn')
wf_model = Waveform_Pretrained()
model = MultiModal_Multitask_Model_4_task(ts_model= ts_model, text_model= text_model, wf_model= wf_model,\
     use_ts =use_ts, use_text = use_text, use_wf = use_wf)

log_var = {
    'decomp' : torch.zeros((1,), requires_grad=True, device=device),
    'ihm'    : torch.zeros((1,), requires_grad=True, device=device),
    'los'    : torch.zeros((1,), requires_grad=True, device=device),
    'pheno'  : torch.zeros((1,), requires_grad=True, device=device)
}
 
optimizer = torch.optim.Adam(([p for p in model.parameters()] + [log_var[t] for t in log_var]), lr=learning_rate)



train_mm_dataset = MultiModal_Dataset(train_ts_root_dir, train_text_root_dir, wf_root_dir, discretizer, train_starttime_path,\
        regression, bin_type, None, ihm_pos, los_pos,  use_wf, use_text, use_ts, wf_dim, decay, w2i_lookup, max_text_length, max_num_notes,
        os.path.join(train_ts_root_dir, '4kval_train_feature_subset_listfile.csv'))

val_mm_dataset = MultiModal_Dataset(train_ts_root_dir, train_text_root_dir, wf_root_dir, discretizer, train_starttime_path,\
        regression, bin_type, None, ihm_pos, los_pos,  use_wf, use_text, use_ts, wf_dim, decay, w2i_lookup, max_text_length, max_num_notes,
        os.path.join(train_ts_root_dir, '4k_val_feature_subset_listfile.csv'))

test_mm_dataset = MultiModal_Dataset(test_ts_root_dir, test_text_root_dir, wf_root_dir, discretizer, test_starttime_path,\
        regression, bin_type, None, ihm_pos, los_pos,  use_wf, use_text, use_ts, wf_dim, decay, w2i_lookup, max_text_length, max_num_notes,
        os.path.join(test_ts_root_dir, 'test_feature_subset_listfile.csv'))

collate_fn_train = functools.partial(custom_collate_fn, union = True)
collate_fn_test = functools.partial(custom_collate_fn, union = True)
train_data_loader = torch.utils.data.DataLoader(dataset=train_mm_dataset, 
                                            batch_size=5,
                                            shuffle=True,
                                            num_workers=5,
                                            collate_fn = collate_fn_train)

val_data_loader = torch.utils.data.DataLoader(dataset=val_mm_dataset, 
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=5,
                                            collate_fn = collate_fn_train)


test_data_loader = torch.utils.data.DataLoader(dataset=test_mm_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=5,
                                            collate_fn = collate_fn_test)



def train(epochs, train_data_loader, test_data_loader, model, optimizer, device):
    # class frequency for each
    decomp_weight = torch.FloatTensor([1.0214, 47.6688])
    decomp_weight = decomp_weight/decomp_weight.sum()
    ihm_weight = torch.FloatTensor([1.1565, 7.3888])
    ihm_weight = ihm_weight/ihm_weight.sum()
    # los_weight = torch.FloatTensor([ 66.9758,  30.3148,  13.7411,   6.8861,   4.8724,   4.8037,   5.7935,
    #       8.9295,  29.8249, 391.6768])
    los_weight = torch.FloatTensor([1.6047, 3.8934, 8.3376])
    #los_weight = los_weight/los_weight.sum()
    pheno_weight = torch.FloatTensor([19.2544,  55.1893,  40.1445,  12.8604,  30.7595,  31.4979,  19.9768,
         57.2309,  15.4088,  12.8200,  43.2644,  21.3991,  14.2026,   9.8531,
         15.3284,  57.1641,  31.0782,  46.4064,  81.0640, 102.7755,  47.5936,
         29.6070,  22.7682,  28.8175,  52.8856])
    pheno_weight = pheno_weight/pheno_weight.sum()
    pheno_weight = pheno_weight.to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    embedding_layer = nn.Embedding(vectors.shape[0], vectors.shape[1])
    embedding_layer.weight.data.copy_(torch.from_numpy(vectors))
    embedding_layer.weight.requires_grad = False
    
    model.to(device)
    train_b = 0
    
    running_loss_decomp = 0.0
    running_loss_los = 0.0
    running_loss_ihm = 0.0
    running_loss_pheno = 0.0
    running_loss = 0.0
    
    
    
    for epoch in range(epochs):
        total_data_points = 0
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 50)
        model.train()
        
        running_loss =0.0
        running_loss_decomp = 0.0
        running_loss_los = 0.0
        running_loss_ihm = 0.0
        running_loss_pheno = 0.0

        epoch_metrics = utils.EpochWriter("Train", regression)

        tk0 = tqdm(train_data_loader, total=int(len(train_data_loader)))

        
        '''
        if os.path.exists('./ckpt/multitasking_experiment_ts_text_union_cw.ckpt'):
            model.load_state_dict(torch.load('./ckpt/multitasking_experiment_ts_text_union_cw.ckpt'))
        '''

           
        for i, data in enumerate(tk0):
            if data is None:
                continue
            
            
            
            ihm_mask = torch.from_numpy(np.array(data['ihm mask']))
            ihm_mask = ihm_mask.to(device)
            ihm_label = torch.from_numpy(np.array(data['ihm label'])).long()
            ihm_label = ihm_label.reshape(-1,1).squeeze(1)
            ihm_label = ihm_label.to(device)
            ihm_weight = ihm_weight.to(device)
            decomp_mask = torch.from_numpy(data['decomp mask'])
            decomp_mask = decomp_mask.to(device)
            decomp_label = torch.from_numpy(data['decomp label']).long()
            # the num valid data is used in case the last batch is smaller than batch size
            num_valid_data = decomp_label.shape[0]
            total_data_points+= num_valid_data
            decomp_label = decomp_label.reshape(-1,1).squeeze(1) # (b*t,)
            decomp_label = decomp_label.to(device)
            decomp_weight = decomp_weight.to(device)
            
            los_mask = torch.from_numpy(np.array(data['los mask']))
            los_mask = los_mask.to(device)
            los_label = torch.from_numpy(np.array(data['los label']))
            
            los_label = los_label.reshape(-1,1).squeeze(1)
            los_label = los_label.to(device)
            los_weight = los_weight.to(device)
            pheno_label = torch.from_numpy(np.array(data['pheno label'])).float()
            pheno_label = pheno_label.to(device)
            
            if use_ts:
                ts = torch.from_numpy(data['time series'])
                ts = ts.permute(1,0,2).float().to(device)
            else:
                ts = None
            if use_wf:
                waveforms = torch.from_numpy(data['waveforms']).float()
                waveforms_weight_mat = torch.from_numpy(data['waveforms weight mat']).float()
                waveforms_weight_mat = waveforms_weight_mat.to(device)
                waveforms = waveforms.to(device)
            else:
                waveforms = None
                waveforms_weight_mat = None
            
            if use_text:
                texts = torch.from_numpy(data['texts']).to(torch.int64)
                texts_weight_mat = torch.from_numpy(data['texts weight mat']).float()
                texts_weight_mat = texts_weight_mat.to(device)
                
                texts = embedding_layer(texts) # [batch_size, num_docs, seq_len, emb_dim]
                #print(texts.shape)
                texts = texts.to(device)
                #print(texts.shape)
                if text_model.name == 'avg':
                    texts = avg_emb(texts, texts_weight_mat)
                # t = ts.shape[0]
                # b = ts.shape[1]
                # texts = torch.rand(b*t, 768).float().to(device)
                # texts_weight_mat = None
            else:
                texts = None
                texts_weight_mat = None

            
            decomp_logits, los_logits, ihm_logits, pheno_logits = model(ts = ts, texts = texts,\
             texts_weight_mat = texts_weight_mat, waveforms = waveforms, waveforms_weight_mat =waveforms_weight_mat)
            loss_decomp = masked_weighted_cross_entropy_loss(None, decomp_logits, decomp_label, decomp_mask)
            #loss_los = masked_weighted_cross_entropy_loss(None, los_logits,los_label, los_mask)
            loss_los = masked_weighted_cross_entropy_loss(los_weight, los_logits, los_label, los_mask)
            #print(loss_los.item())
            loss_ihm = masked_weighted_cross_entropy_loss(None,ihm_logits, ihm_label, ihm_mask)
            loss_pheno = criterion(pheno_logits, pheno_label)


            losses = {
                'decomp' : loss_decomp,
                'ihm'    : loss_ihm,
                'los'    : loss_los,
                'pheno'  : loss_pheno
            }
            loss = 0.0

            for task in losses:
                #loss += losses[task]
                
                prec = torch.exp(-log_var[task])
                #losses[task] = torch.sum(losses[task] * prec + log_var[task], -1)


                loss += torch.sum(losses[task] * prec + log_var[task], -1)
                #loss += losses[task] * task_weight[task]



                #loss += losses[task]
            #loss = torch.mean(loss)

                            

            #loss = loss_decomp*5+loss_ihm*2+loss_los*1+loss_pheno*2
            #loss = loss_los
            train_b+=1
        
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_loss_decomp += loss_decomp.item() * task_weight['decomp']
            running_loss_los +=loss_los.item()* task_weight['los']
            running_loss_ihm +=loss_ihm.item()* task_weight['ihm']
            running_loss_pheno += loss_pheno.item()* task_weight['pheno']


            m = nn.Softmax(dim=1)
            sig = nn.Sigmoid()

            decomp_pred = (m(decomp_logits)[:, 1]).cpu().detach().numpy()
            #los_pred = torch.argmax(m(los_logits), dim=1).cpu().detach().numpy()
            los_pred = m(los_logits).cpu().detach().numpy()
            #los_pred = los_logits.cpu().detach().numpy()
            ihm_pred = (m(ihm_logits)[:, 1]).cpu().detach().numpy()
            pheno_pred = sig(pheno_logits).cpu().detach().numpy()

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
            }
            epoch_metrics.cache(outputs, num_valid_data)
            interval = 500
            

            if i %interval == interval-1:
                writer.add_scalar('training loss',
                            running_loss/(interval-1),
                            train_b)
                writer.add_scalar('decomp loss',
                            running_loss_decomp / (interval-1),
                            train_b)
                writer.add_scalar('los loss',
                            running_loss_los / (interval -1),
                            train_b)
                writer.add_scalar('ihm loss',
                            running_loss_ihm / (interval -1),
                            train_b)
                writer.add_scalar('pheno loss',
                            running_loss_pheno / (interval -1),
                            train_b)
                '''
                print('loss decomp', running_loss_decomp/interval)
                print('loss ihm', running_loss_ihm/interval)
                print('loss los', running_loss_los/interval)
                print('loss pheno', running_loss_pheno/interval)
                print('epoch {} , training loss is {:.3f}'.format(epoch+1, running_loss_los/interval))
                '''
                
                running_loss_decomp = 0.0
                running_loss_los = 0.0
                running_loss_ihm = 0.0
                running_loss_pheno = 0.0
                running_loss = 0.0
                epoch_metrics.add()
        for task in losses:
            print(task, torch.exp(-log_var[task]))
        epoch_metrics.write(writer,train_b)

        torch.save(model.state_dict(), os.path.join('./ckpt/','epoch{0}'.format(epoch) + model_weights))

        evaluate(epoch, val_data_loader, model, device, train_b, "Val")
        evaluate(epoch, test_data_loader, model, device, train_b, "Test")


def evaluate(epoch, data_loader, model, device, train_step=None, split="Test"):
    embedding_layer = nn.Embedding(vectors.shape[0], vectors.shape[1])
    embedding_layer.weight.data.copy_(torch.from_numpy(vectors))
    embedding_layer.weight.requires_grad = False
    '''
    if os.path.exists('./ckpt/multitasking_experiment_ts_text_union_cw.ckpt'):
        model.load_state_dict(torch.load('./ckpt/multitasking_experiment_ts_text_union_cw.ckpt'))
    '''

    epoch_metrics = utils.EpochWriter(split, regression)
    model.to(device)
    model.eval()
    total_data_points = 0

    running_loss = 0.0
    running_loss_decomp = 0.0
    running_loss_los = 0.0
    running_loss_ihm = 0.0
    running_loss_pheno = 0.0
    tk = tqdm(data_loader, total=int(len(data_loader)))
    criterion = nn.BCEWithLogitsLoss()

   
    for i, data in enumerate(tk):
        
        # if i>=50:
        #     break
        if data is None:
            continue
        num_valid_data = data['decomp label'].shape[0]
        total_data_points+= num_valid_data
        name = data['name'][0]
        # print(name)

        ihm_mask = torch.from_numpy(np.array(data['ihm mask'])).long()
        ihm_mask = ihm_mask.to(device)

        ihm_label = np.array(data['ihm label'])
        ihm_label = ihm_label.reshape(-1,1).squeeze(1)
        ihm_label = torch.from_numpy(ihm_label).long().to(device)

        decomp_mask = (torch.from_numpy(data['decomp mask'])).long().to(device)
        decomp_label = data['decomp label']
        #num_valid_data = decomp_label.shape[0]
        decomp_label = decomp_label.reshape(-1,1).squeeze(1) # (b*t,)
        decomp_label = torch.from_numpy(decomp_label).long().to(device)

        los_mask = torch.from_numpy(np.array(data['los mask'])).long().to(device)
        #print('los mask',los_mask)
        #np.save('./res/mask/los_mask_{}.npy'.format(name), los_mask)
        los_label = np.array(data['los label'])
        #np.save('./res/label/los_label_{}.npy'.format(name), los_label)
        los_label = los_label.reshape(-1,1).squeeze(1)
        
        los_label = torch.from_numpy(los_label).to(device)
        pheno_label = torch.from_numpy(np.array(data['pheno label'])).float().to(device)
        
        if use_ts:
                ts = torch.from_numpy(data['time series'])
                ts = ts.permute(1,0,2).float().to(device)
        else:
            ts = None

        if use_wf:
            waveforms = torch.from_numpy(data['waveforms']).float()
            waveforms_weight_mat = torch.from_numpy(data['waveforms weight mat']).float()
            waveforms_weight_mat = waveforms_weight_mat.to(device)
            waveforms = waveforms.to(device)
        else:
            waveforms = None
            waveforms_weight_mat = None
        
        if use_text:
            texts = torch.from_numpy(data['texts']).to(torch.int64)
            texts_weight_mat = torch.from_numpy(data['texts weight mat']).float()
            texts_weight_mat = texts_weight_mat.to(device)
            
            texts = embedding_layer(texts) # [batch_size, num_docs, seq_len, emb_dim]
            
            texts = texts.to(device)
            if text_model.name == 'avg':
                texts = avg_emb(texts, texts_weight_mat)
            # t = ts.shape[0]
            # b = ts.shape[1]
            # texts = torch.rand(b*t, 768).float().to(device)
            # texts_weight_mat = None
        else:
            texts = None
            texts_weight_mat = None
        
        decomp_logits, los_logits, ihm_logits, pheno_logits = model(ts = ts, texts = texts,texts_weight_mat = texts_weight_mat,\
         waveforms = waveforms, waveforms_weight_mat = waveforms_weight_mat)

        loss_decomp = masked_weighted_cross_entropy_loss(None, 
                                                        decomp_logits, 
                                                        decomp_label, 
                                                        decomp_mask)
        loss_los = masked_weighted_cross_entropy_loss(None, 
                                                      los_logits,
                                                      los_label, 
                                                      los_mask)
        # loss_los = masked_mse_loss(los_logits,
        #                            los_label, 
        #                            los_mask)
        loss_ihm = masked_weighted_cross_entropy_loss(None,ihm_logits, 
                                                      ihm_label, 
                                                      ihm_mask)
        loss_pheno = criterion(pheno_logits, pheno_label)

        
        losses = {
            'decomp' : loss_decomp,
            'ihm'    : loss_ihm,
            'los'    : loss_los,
            'pheno'  : loss_pheno
        }
        loss = 0.0

        for task in losses:
            loss += losses[task] * task_weight[task]
    
        running_loss += loss.item()
        running_loss_decomp += loss_decomp.item() * task_weight['decomp']
        running_loss_los +=loss_los.item()* task_weight['los']
        running_loss_ihm +=loss_ihm.item()* task_weight['ihm']
        running_loss_pheno += loss_pheno.item()* task_weight['pheno']



        m = nn.Softmax(dim=1)
        sigmoid = nn.Sigmoid()
        
        decomp_pred = (sigmoid(decomp_logits)[:,1]).cpu().detach().numpy()
        #los_pred = torch.argmax(m(los_logits), dim=1).cpu().detach().numpy()
        los_pred = m(los_logits).cpu().detach().numpy()
        #np.save('./res/pred/los_pred_{}.npy'.format(name), los_pred)
        #los_pred = los_logits.cpu().detach().numpy()
        ihm_pred = (sigmoid(ihm_logits)[:,1]).cpu().detach().numpy()
        pheno_pred = sigmoid(pheno_logits).cpu().detach().numpy()
        #print(pheno_pred.sum(dim=1))

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
        }

        epoch_metrics.cache(outputs, num_valid_data)
   
    #unique_elements, counts_elements = np.unique(metric_los_kappa.y_pred, return_counts=True)
    #labels, counts = np.unique(metric_los_kappa.y_true, return_counts=True)

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
 
#evaluate(0, test_data_loader, model, device, None)
train(epochs, train_data_loader, test_data_loader, model, optimizer, device)