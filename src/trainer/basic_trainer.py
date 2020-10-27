# -*- coding: utf-8 -*-
# @Author       : Yong Huang
# @Time         : Created at 2020-07-24
# @Description  : 
# Copyrights (C) 2020. All Rights Reserved.

import numpy as np
import os
import torch
import torch.nn as nn

from utils import get_args, EpochWriter
from loss import masked_weighted_cross_entropy_loss
from dataloaders import custom_collate_fn, MultiModal_Dataset
from multi_modality_model_hy import LSTMModel, ChannelWiseLSTM
from multi_modality_model_hy import Text_AVG, Text_CNN
from multi_modality_model_hy import Waveform_CNN, Waveform_Pretrained
from multi_modality_model_hy import MultiModal_Multitask_Model_4_task


args = get_args()


class Trainer():
    def __init__(self, args):
        self.tasks = ['decomp', 'los', 'ihm', 'pheno']
        self.experiment = '_'.join(self.tasks)
        self.version = 1
        # tensorboard log path
        while os.path.exists(os.path.join('../runs', self.experiment+"_v{}".format(version))):
            self.version += 1
        self.writer = SummaryWriter(os.path.join('../runs', self.experiment))
        self.name = self.experiment + "_v{}".format(self.version)
        self.log = create_logger(name)
        self.args = args
        self.show_args()
        self.pheno_criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.task_weight = {
                'decomp': args.decomp_weight,
                'los':    args.los_weight,
                'ihm':    args.ihm_weight,
                'pheno':  args.pheno_weight
            }
        self.ts_model = ChannelWiseLSTM()
        
        if args.use_text:
            if args.text_model_name == 'text cnn':
                self.text_model = Text_CNN()
            elif args.text_model_name == 'text avg':
                self.text_model = Text_AVG()
            self.vectors, self.w2i_lookup = utils.get_embedding_dict(args)

        if args.use_ts:
            if args.ts_model_name == 'standard lstm':
                self.ts_model = LSTMModel()
            elif args.ts_model_name == 'channel-wise lstm':
                self.ts_model = ChannelWiseLSTM()

        if args.use_wf:
            self.wf_model = Waveform_CNN()
    
        # Dataloader
        self.train_dataset = MultiModal_Dataset(
            args.train_ts_root_dir, args.train_text_root_dir,\
            args.wf_root_dir, args.discretizer, args.train_starttime_path,\
            args.normalizer, args.ihm_pos, args.use_wf, args.use_text,\
            args.use_ts, args.wf_dim, args.decay, args.w2i_lookup,\
            args.max_text_length, args.max_num_notes
        ) 
        
        self.test_dataset = MultiModal_Dataset(
            args.test_ts_root_dir, args.test_text_root_dir,\
            args.wf_root_dir, args.discretizer, args.test_starttime_path,\
            args.normalizer, args.ihm_pos, args.use_wf, args.use_text,\
            args.use_ts, args.wf_dim, args.decay, args.w2i_lookup,\
            args.max_text_length, args.max_num_notes
        )

        self.collate_function = functools.partial(custom_collate_fn, union = True)
        self.train_data_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=5,
                                            collate_fn = self.collate_function)
        self.test_data_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=5,
                                            collate_fn = self.collate_function)
        
        self.train_epoch_metrics = EpochWriter("Train")
        self.test_epoch_metrics = EpochWriter("Test")


    def _get_cross_entropy_weights(self):
        # TODO: update los weight based on binning strategy
        decomp_weight = torch.FloatTensor([1.0214, 47.6688])
        decomp_weight = decomp_weight/decomp_weight.sum()
        ihm_weight = torch.FloatTensor([1.1565, 7.3888])
        ihm_weight = ihm_weight/ihm_weight.sum()
        los_weight = torch.FloatTensor([ 66.9758,  30.3148,  13.7411,   6.8861,   4.8724,   4.8037,   5.7935,
            8.9295,  29.8249, 391.6768])
        #los_weight = los_weight/los_weight.sum()
        pheno_weight = torch.FloatTensor([19.2544,  55.1893,  40.1445,  12.8604,  30.7595,  31.4979,  19.9768,
            57.2309,  15.4088,  12.8200,  43.2644,  21.3991,  14.2026,   9.8531,
            15.3284,  57.1641,  31.0782,  46.4064,  81.0640, 102.7755,  47.5936,
            29.6070,  22.7682,  28.8175,  52.8856])
        pheno_weight = pheno_weight/pheno_weight.sum()
        pheno_weight = pheno_weight.to(device)
        return {
            'decomp ce weight': decomp_weight,
            'los ce weight': los_weight,
            'ihm ce weight': ihm_weight,
            'pheno ce weight': pheno_weight,
        }

    def _unpack_data_batch(self, data):
        ihm_mask = torch.from_numpy(np.array(data['ihm mask']))
        ihm_label = torch.from_numpy(np.array(data['ihm label'])).long()
        ihm_label = ihm_label.reshape(-1,1).squeeze(1)
        
        decomp_mask = torch.from_numpy(data['decomp mask'])
        decomp_label = torch.from_numpy(data['decomp label']).long()
        decomp_label = decomp_label.reshape(-1,1).squeeze(1) # (b*t,)
        num_valid_data = decomp_label.shape[0]

        los_mask = torch.from_numpy(np.array(data['los mask']))
        los_mask = los_mask.to(device)
        los_label = torch.from_numpy(np.array(data['los label']))
        
        los_label = los_label.reshape(-1,1).squeeze(1)
        los_label = los_label.to(device)
        los_weight = los_weight.to(device)
        pheno_label = torch.from_numpy(np.array(data['pheno label'])).float()
        pheno_label = pheno_label.to(device)





    def train(self):
        self._init_text_emb_layer()
        self.model.to(self.device)
        train_b = 0
    
        for epoch in range(epochs):
            total_data_points = 0
            self.log.info('Epoch {}/{}'.format(epoch+1, epochs))
            self.log.info('-' * 50)
            model.train()
            
            running_loss =0.0
            running_loss_decomp = 0.0
            running_loss_los = 0.0
            running_loss_ihm = 0.0
            running_loss_pheno = 0.0

            tk0 = tqdm(self.train_data_loader, total=int(len(self.train_data_loader)))
            for i, data in enumerate(tk0):
                
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
                    texts = texts.to(device)

                    if text_model.name == 'avg':
                        texts = avg_emb(texts, texts_weight_mat)
                else:
                    texts = None
                    texts_weight_mat = None

                
                decomp_logits, los_logits, ihm_logits, pheno_logits = model(ts = ts, texts = texts,\
                texts_weight_mat = texts_weight_mat, waveforms = waveforms, waveforms_weight_mat =waveforms_weight_mat)
                loss_decomp = masked_weighted_cross_entropy_loss(None, decomp_logits, decomp_label, decomp_mask)
                #loss_los = masked_weighted_cross_entropy_loss(None, los_logits,los_label, los_mask)
                loss_los = masked_weighted_cross_entropy_loss(None, los_logits, los_label, los_mask)
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
                    loss += losses[task] * task_weight[task]
                

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
                los_pred = torch.argmax(m(los_logits), dim=1).cpu().detach().numpy()
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
                interval = 50
                

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
                    
                    running_loss_decomp = 0.0
                    running_loss_los = 0.0
                    running_loss_ihm = 0.0
                    running_loss_pheno = 0.0
                    running_loss = 0.0
                    epoch_metrics.add()

            epoch_metrics.write(writer,train_b)


        
    def _update_writer(self):

    def evaluation(self):
        pass

    
    def _run(self):
        self.show_args()
        self.train()


    def _init_text_emb_layer(self):
        self.embedding_layer = nn.Embedding(vectors.shape[0], vectors.shape[1])
        self.embedding_layer.weight.data.copy_(torch.from_numpy(vectors))
        self.embedding_layer.weight.requires_grad = False


    def create_logger(self, name, silent=False, to_disk=False, log_file=None):
        """Create a new logger"""
        # setup logger
        log = logging.getLogger(name)
        log.setLevel(logging.DEBUG)
        log.propagate = False
        formatter = logging.Formatter(fmt='%(message)s', datefmt='%Y/%m/%d %I:%M:%S')
        if not silent:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            log.addHandler(ch)
        if to_disk:
            log_file = log_file if log_file is not None else strftime("log/log_%m%d_%H%M.txt", gmtime())
            if type(log_file) == list:
                for filename in log_file:
                    fh = logging.FileHandler(filename, mode='w')
                    fh.setLevel(logging.INFO)
                    fh.setFormatter(formatter)
                    log.addHandler(fh)
            if type(log_file) == str:
                fh = logging.FileHandler(log_file, mode='w')
                fh.setLevel(logging.INFO)
                fh.setFormatter(formatter)
                log.addHandler(fh)
        return log


    def show_args(self):
        logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
        self.log.info(100 * '=')
        self.log.info('> training arguments:')
        for arg in vars(self.args):
            self.log.info('>>> {0}: {1}'.format(arg, getattr(self.args, arg)))
        self.log.info(100 * '=')

    
    def save_ckpt(self, epoch, model_weights):
        if not os.path.exists(self.args.checkpoint_path):
            os.makedirs(self.args.checkpoint_path)
        torch.save(model.state_dict(), os.path.join('./ckpt/',self.name,'epoch{0}'.format(epoch) + model_weights))

        






    
        

        



