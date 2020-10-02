import json
import pandas as pd
import os
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import sklearn.utils as sk_utils
import config
import tensorflow as tf
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
import logging
import pickle
import argparse
import sys
import numpy as np
import torch
from matplotlib import pyplot
import time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, '..')
from copy import deepcopy
from numpy.random import rand
from numpy.random import randint
from numpy import mean
from numpy import median
from numpy import percentile


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_file", default = '../experiment.log')
    parser.add_argument(
        "--checkpoint_path", default ='/home/yong/mutiltasking-for-mimic3/multimodal/experiments/multitasking/ckpt', help="Path for checkpointing")
    parser.add_argument(
        "--basepath", default ='/home/luca/mutiltasking-for-mimic3/data/', help='root path of data')
    parser.add_argument(
        "--train_text_root_dir", default = '/home/yong/mutiltasking-for-mimic3/data/root/text_fixed_train/')
    parser.add_argument(
        "--test_text_root_dir", default = '/home/yong/mutiltasking-for-mimic3/data/root/text_fixed_test/')
    parser.add_argument(
        "--train_ts_root_dir", default = '/home/yong/mutiltasking-for-mimic3/data/root/text_fixed_train/')
    parser.add_argument(
        "--test_ts_root_dir", default = '/home/yong/mutiltasking-for-mimic3/data/root/text_fixed_test/')
    parser.add_argument(
        "--embeddingspath", default = "./embeds/BioWordVec_PubMed_MIMICIII_d200.vec.bin")
    parser.add_argument(
        "--reduced_emb_model_path", default = '/home/luca/mutiltasking-for-mimic3/data/embeds/wv.pkl',
        help = "pretrained embedding model weights")
    parser.add_argument(
        "--normalizer_state", default = 'mult_ts1.0.input_str:previous.start_time:zero.normalizer',
        help='presaved normalizer, see scientific report paper for more details')
    parser.add_argument(
        "--multitask_path", default = "/home/luca/mutiltasking-for-mimic3/data/multitask/")
    parser.add_argument(
        "--train_starttime_path", default = "/home/luca/mutiltasking-for-mimic3/data/root/train_starttime.pkl",
        help = "files used to help align time for text data")
    parser.add_argument(
        "--test_starttime_path", default = "/home/luca/mutiltasking-for-mimic3/data/root/test_starttime.pkl",
        help = "files used to help align time for text data")
    parser.add_argument(
        "--los_pos", default=48, type = int)
    parser.add_argument(
        "--ihm_pos", default=48, type = int)  
    parser.add_argument(
        "--epochs", default=20, type = int)
    parser.add_argument(
        "--batch_size", default=5, type = int)
    parser.add_argument(
        "--decay", default=0.1, help='decay term used in weight matrix', type = float)
    parser.add_argument(
        "--timestep", default =1.0, help ='in hours, we discretize the data in default 1 hour', type = float)
    parser.add_argument(
        "--imputation", default = 'previous', help='imputation stratagy for missing time series data')
    parser.add_argument(
        "--small_part", default = "False", help= 'training on a small subset of training data')
    parser.add_argument(
        "--use_ts", help="whether to use time series data, default is true", action="store_false")
    parser.add_argument(
        "--use_text", help="whether to use time series data, default is false", action="store_true")
    parser.add_argument(
        "--use_wf", help="whether to use time series data, default is false", action="store_true")
    parser.add_argument(
        "--regression", help="whether to do regression on los task, default is true ", action="store_false")
    parser.add_argument(
        "--wf_dim", default = 100, type = int)
    parser.add_argument(
        "--learning_rate", default = '1e-4')
    parser.add_argument(
        "--max_text_length", default = 500, help = "maximum notes length", type= int)
    parser.add_argument(
        "--max_num_notes", default = 150, help = "maximum number of notes per icu stay", type= int)
    parser.add_argument(
        "--padding_type", default = "Zero")
    parser.add_argument(
        "--bin_type", default = "custom")
    
    args = vars(parser.parse_args())
    return args


def get_config():
    return config.Config()


def get_embedding_dict(conf):
    with open(conf.model_path, 'rb') as f:
        data = pickle.load(f)

    index2word_tensor = data["model"]["index2word"]
    index2word_tensor.pop()
    index2word_tensor.append('<pad>')
    word2index_lookup = {word: index for index,
                         word in enumerate(index2word_tensor)}
    vectors = data["model"]["vectors"]

    return vectors, word2index_lookup


def lookup(w2i_lookup, x):
    if x in w2i_lookup:
        return w2i_lookup[x]
    else:
        return len(w2i_lookup)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, experiment_name, patience=4, verbose=True, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 4
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = os.path.join('./ckpt/', experiment_name + '.ckpt')
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class AUCROC():
    def __init__(self, *args, **kwargs):
        self.y_true = None
        self.y_pred = None

    def add(self, pred, y_true):
        if self.y_pred is None:
            self.y_pred = pred
        else:
            self.y_pred = np.concatenate([self.y_pred, pred])
        if self.y_true is None:
            self.y_true = y_true
        else:
            self.y_true = np.concatenate([self.y_true, y_true])

    def get(self):
        roc_auc = dict() 
        roc_auc["macro"] = metrics.roc_auc_score(self.y_true, self.y_pred, average ='macro')
        roc_auc["micro"] = metrics.roc_auc_score(self.y_true.flatten(), self.y_pred.flatten(), average ='micro')
        return roc_auc

    def save(self, name):
        path = './res'
        fname = os.path.join(path,name + ".pkl")
        with open(fname, 'wb') as f:
            pickle.dump((self.y_pred, self.y_true), f, pickle.HIGHEST_PROTOCOL)


class AUCPR():
    def __init__(self, *args, **kwargs):
        self.y_true = None
        self.y_pred = None
        self.y_mask = None

    def add(self, pred, y_true):
        if self.y_pred is None:
            self.y_pred = pred
        else:
            self.y_pred = np.concatenate([self.y_pred, pred])
        if self.y_true is None:
            self.y_true = y_true
        else:
            self.y_true = np.concatenate([self.y_true, y_true])

    def get(self):
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            self.y_true, self.y_pred)
        aucpr = metrics.auc(recalls, precisions)
        return aucpr

    def save(self, name):
        path = './res'
        fname = os.path.join(path,name + ".pkl")
        with open(fname, 'wb') as f:
            pickle.dump((self.y_pred, self.y_true), f, pickle.HIGHEST_PROTOCOL)


class ConfusionMatrix():
    def __init__(self, *args, **kwargs):
        self.y_true = None
        self.y_pred = None

    def add(self, pred, y_true, y_mask):
        y_mask = y_mask[:, 0]
        y_true = y_true[y_mask == 1]
        pred = pred[y_mask == 1]
        if self.y_pred is None:
            self.y_pred = pred
        else:
            self.y_pred = np.concatenate([self.y_pred, pred])
        if self.y_true is None:
            self.y_true = y_true
        else:
            self.y_true = np.concatenate([self.y_true, y_true])
    
    def get(self):
        print(self.y_pred.shape)
        self.y_pred = np.argmax(self.y_pred, axis=1)
        cfm = metrics.confusion_matrix(self.y_true, self.y_pred)
        return cfm

    

class AUCROCLOS(AUCROC):
    def add(self, pred, y_true, y_mask):
        y_mask = y_mask[:, 0]
        y_true = y_true[y_mask == 1]
        pred = pred[y_mask == 1]
        super().add(pred, y_true)

    def binarilize(self, day):
        y_pred = deepcopy(self.y_pred)
        y_true = deepcopy(self.y_true)
        if day == 3:
            y_pred = y_pred[:,1]+ y_pred[:,2]
            y_true = np.where(y_true >= 1, 1, 0)
        elif day == 7:
            y_pred = y_pred[:,-1]
            y_true = np.where(y_true == 2, 1, 0)
        return y_pred, y_true

    def get(self):
        roc_auc_ovr = metrics.roc_auc_score(self.y_true, self.y_pred, multi_class ='ovr')
        roc_auc_ovo = metrics.roc_auc_score(self.y_true, self.y_pred, multi_class ='ovo')
        print(metrics.confusion_matrix(self.y_true, np.argmax(self.y_pred, axis=1)))

        # 3 days aucroc
        y_pred_3_days, y_true_3_days = self.binarilize(3)
        roc_auc_3_days = metrics.roc_auc_score(y_true_3_days, y_pred_3_days)
        # 7 days aucroc
        y_pred_7_days, y_true_7_days = self.binarilize(7)
        roc_auc_7_days = metrics.roc_auc_score(y_true_7_days, y_pred_7_days)
        return roc_auc_ovo, roc_auc_ovr, roc_auc_3_days, roc_auc_7_days



class AUCPRLOS(AUCPR):
    def add(self, pred, y_true, y_mask):
        y_mask = y_mask[:, 0]
        y_true = y_true[y_mask == 1]
        pred = pred[y_mask == 1]
        super().add(pred, y_true)

    def binarilize(self, day):
        y_pred = deepcopy(self.y_pred)
        y_true = deepcopy(self.y_true)
        if day == 3:
            y_pred = y_pred[:,1]+ y_pred[:,2]
            y_true = np.where(y_true >= 1, 1, 0)
        elif day == 7:
            y_pred = y_pred[:,-1]
            y_true = np.where(y_true == 2, 1, 0)
        return y_pred, y_true

    def get(self):
        # 3 days aucpr
        y_pred_3_days, y_true_3_days = self.binarilize(3)
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            y_true_3_days, y_pred_3_days)
        aucpr_3_days = metrics.auc(recalls, precisions)
        
        # 7 days aucpr
        y_pred_7_days, y_true_7_days = self.binarilize(7)
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            y_true_7_days, y_pred_7_days)
        aucpr_7_days = metrics.auc(recalls, precisions)
        return aucpr_3_days, aucpr_7_days


class AUCROCIHM(AUCROC):
    def add(self, pred, y_true, y_mask):
        y_mask = y_mask[:, 0]
        y_true = y_true[y_mask == 1]
        pred = pred[y_mask == 1]
        super().add(pred, y_true)
    def get(self):
        fpr, tpr, _ =  metrics.roc_curve(self.y_true, self.y_pred)
        roc_auc = metrics.auc(fpr, tpr)
        return roc_auc


class AUCPRIHM(AUCPR):
    def add(self, pred, y_true, y_mask):
        y_mask = y_mask[:, 0]
        y_true = y_true[y_mask == 1]
        pred = pred[y_mask == 1]
        super().add(pred, y_true)

    def get(self):
        aucpr = super().get()
        return aucpr

class AUCROCLTM(AUCROC):
    def add(self, pred, y_true, y_mask):
        y_mask = y_mask[:, 0]
        y_true = y_true[y_mask == 1]
        pred = pred[y_mask == 1]
        super().add(pred, y_true)
    def get(self):
        fpr, tpr, _ =  metrics.roc_curve(self.y_true, self.y_pred)
        roc_auc = metrics.auc(fpr, tpr)
        return roc_auc


class AUCPRLTM(AUCPR):
    def add(self, pred, y_true, y_mask):
        y_mask = y_mask[:, 0]
        y_true = y_true[y_mask == 1]
        pred = pred[y_mask == 1]
        super().add(pred, y_true)

    def get(self):
        aucpr = super().get()
        return aucpr

class AUCROCREADMIT(AUCROC):
    def add(self, pred, y_true, y_mask):
        y_mask = y_mask[:, 0]
        y_true = y_true[y_mask == 1]
        pred = pred[y_mask == 1]
        super().add(pred, y_true)
    def get(self):
        roc_auc_ovr = metrics.roc_auc_score(self.y_true, self.y_pred, multi_class ='ovr')
        roc_auc_ovo = metrics.roc_auc_score(self.y_true, self.y_pred, multi_class ='ovo')
        return roc_auc_ovr, roc_auc_ovo


class ACCREADMIT():
    def __init__(self, *args, **kwargs):
        self.y_true = None
        self.y_pred = None
        self.y_mask = None

    def add(self, pred, y_true, y_mask):
        y_mask = y_mask[:, 0]
        y_true = y_true[y_mask == 1]
        pred = pred[y_mask == 1]
        pred = np.argmax(pred, axis=1)
        if self.y_pred is None:
            self.y_pred = pred
        else:
            self.y_pred = np.concatenate([self.y_pred, pred])
        if self.y_true is None:
            self.y_true = y_true
        else:
            self.y_true = np.concatenate([self.y_true, y_true])

    def get(self):
        acc = metrics.accuracy_score(self.y_true, self.y_pred)
        return acc

    

class MetricPerHour():
    def __init__(self):
        self.y_true_hr = {}
        self.pred_hr = {}
        self.aucpr = {}
        self.y_true = None
        self.y_pred = None
        self.metric_type = 'aucpr'

    def add(self, pred, y_true, mask, num_valid_data):
        pred = pred.reshape(num_valid_data, -1)
        y_true = y_true.reshape(num_valid_data, -1)
        mask = mask.reshape(num_valid_data, -1)
        assert len(pred.shape) == 2, "Pred: {} Y: {} Mask:{}".format(
            str(pred.shape), str(y_true.shape), str(mask.shape))
        assert len(y_true.shape) == 2
        assert len(mask.shape) == 2
        y_true_list = []
        pred_list = []
        for hour in range(y_true.shape[1]):
            mask_h = mask[:, hour]
            if mask_h.sum()< 1:
                continue
            y_true_h = y_true[:, hour]
            pred_h = pred[:, hour]
            mask_h = mask_h.astype(np.bool)
            y_true_h = y_true_h[mask_h]
            pred_h = pred_h[mask_h]

            if hour not in self.y_true_hr:
                self.y_true_hr[hour] = y_true_h
                self.pred_hr[hour] = pred_h
            else:
                self.y_true_hr[hour] = np.concatenate(
                    [self.y_true_hr[hour], y_true_h])
                self.pred_hr[hour] = np.concatenate(
                    [self.pred_hr[hour], pred_h])

            if self.y_true is None:
                self.y_true = [y_true_h]
                self.y_pred = [pred_h]
            else:
                self.y_true.append(y_true_h) #= np.concatenate([self.y_true, y_true_h])
                self.y_pred.append(pred_h) #= np.concatenate([self.y_pred, pred_h])
                assert len(self.y_true) == len(self.y_pred)

    def get_metric(self, y_true, pred):
        if self.metric_type == 'aucpr':
            (precisions, recalls, thresholds) = metrics.precision_recall_curve(
                y_true, pred)
            value = metrics.auc(recalls, precisions)
        elif self.metric_type == 'kappa':
            value = metrics.cohen_kappa_score(y_true.flatten(), pred.flatten(), weights='linear')
        elif self.metric_type == 'mad':
            value = metrics.mean_absolute_error(y_true.flatten(), pred.flatten())
        elif self.metric_type == 'mae':
            value = metrics.median_absolute_error(y_true.flatten(), pred.flatten())
        elif self.metric_type == 'ev_score':
            value = metrics.explained_variance_score(y_true.flatten(), pred.flatten())
        elif self.metric_type == 'aucroc':
            fpr, tpr, _ = metrics.roc_curve(y_true.ravel(), pred.ravel())
            value = metrics.auc(fpr, tpr) 
        elif self.metric_type == 'macro_aucroc':
            value = metrics.roc_auc_score(y_true.flatten(), pred.flatten())
        else:
            value = metrics.accuracy_score(y_true.flatten(), pred.flatten())

        return value

    def get(self):
        self.aucpr = {}  
        self.y_true = np.concatenate(self.y_true)
        self.y_pred = np.concatenate(self.y_pred)
        aggregated = self.get_metric(self.y_true, self.y_pred)
        return self.y_true_hr, self.pred_hr, self.aucpr, aggregated
    
    def save(self, name):
        path = './res'
        fname = os.path.join(path,name + ".pkl")
        with open(fname, 'wb') as f:
            pickle.dump((self.y_pred, self.y_true), f, pickle.HIGHEST_PROTOCOL)


class AUCPRperHour(MetricPerHour):
    def __init__(self):
        super().__init__()
        self.metric_type = 'aucpr'
    

class KappaPerHour(MetricPerHour):
    def __init__(self):
        super().__init__()
        self.metric_type = 'kappa'


class AUCROCPerHour(MetricPerHour):
    def __init__(self):
        super().__init__()
        self.metric_type = 'aucroc'


class AUCROCMacroPerHour(MetricPerHour):
    def __init__(self):
        super().__init__()
        self.metric_type = 'macro_aucroc'

class ACCPerHour(MetricPerHour):
    def __init__(self):
        super().__init__()
        self.metric_type = 'acc'


class MADPerHour(MetricPerHour):
    def __init__(self):
        super().__init__()
        self.metric_type = 'mad'


class EVPerHour(MetricPerHour):
    def __init__(self):
        super().__init__()
        self.metric_type = 'ev_score'


class MAEPerHour(MetricPerHour):
    def __init__(self):
        super().__init__()
        self.metric_type = 'mae'

class EpochWriter:
    """
    Hold all metrics for an entire epoch of training
    """

    def __init__(self, split, regression, experiment_name):
        self.split = split
        self.regression = regression
        self.experiment_name = experiment_name
        self.metric_decomp_aucpr = AUCPRperHour()
        self.metric_decomp_aucroc = AUCROCPerHour()
        if not self.regression:
            self.metric_los_aucroc = AUCROCLOS()
            self.metric_los_aucpr = AUCPRLOS()
            self.metric_los_cfm = ConfusionMatrix()
        else:
            self.metric_los_mad = MADPerHour()
            self.metric_los_mae = MAEPerHour()
            self.metric_los_ev = EVPerHour()
        
        self.metric_ihm_aucpr = AUCPRIHM()
        self.metric_ihm_aucroc = AUCROCIHM()
        self.metric_pheno_aucroc = AUCROC() 
        self.metric_ltm_aucroc = AUCROCLTM()
        self.metric_ltm_aucpr = AUCPRLTM()
        self.metric_readmit_aucroc = AUCROCREADMIT()
        self.metric_readmit_acc = ACCREADMIT()
        self.metric_readmit_cfm = ConfusionMatrix()
        self.metric_ltm_cfm = ConfusionMatrix()
        self.outputs_list = []
        self.num_valid_data_list = []
        self.exec = ThreadPoolExecutor(max_workers = 1)
        self.lock = None
    
    def _add(self, outputs, num_valid_data):
        """
        Add the preds and labels from outputs to their respective 
        metric classes

        :param: outputs, dict of tasks, each element is a dict of preds/labels/
                masks for that given task
        """
        start = time.time()
        decomp_pred = outputs['decomp']['pred']
        decomp_label = outputs['decomp']['label']
        decomp_mask = outputs['decomp']['mask']

        ihm_pred = outputs['ihm']['pred']
        ihm_label = outputs['ihm']['label']
        ihm_mask = outputs['ihm']['mask']

        los_pred = outputs['los']['pred']
        los_label = outputs['los']['label']
        los_mask = outputs['los']['mask']

        pheno_pred = outputs['pheno']['pred']
        pheno_label = outputs['pheno']['label']
        pheno_mask = outputs['pheno']['mask']

        ltm_pred = outputs['ltm']['pred']
        ltm_label = outputs['ltm']['label']
        ltm_mask = outputs['ltm']['mask']

        readmit_pred = outputs['readmit']['pred']
        readmit_label = outputs['readmit']['label']
        readmit_mask = outputs['readmit']['mask']

        self.metric_decomp_aucpr.add(decomp_pred, decomp_label,
                                decomp_mask, num_valid_data)
        self.metric_decomp_aucroc.add(decomp_pred, decomp_label,
                                    decomp_mask, num_valid_data)
        if not self.regression:
            self.metric_los_aucroc.add(los_pred, los_label, los_mask)
            self.metric_los_aucpr.add(los_pred, los_label, los_mask)
            self.metric_los_cfm.add(los_pred, los_label, los_mask)
        else:
            self.metric_los_mad.add(los_pred, los_label, los_mask, num_valid_data)
            self.metric_los_mae.add(los_pred, los_label, los_mask, num_valid_data)
            self.metric_los_ev.add(los_pred, los_label, los_mask, num_valid_data)

        self.metric_ihm_aucpr.add(ihm_pred, ihm_label, ihm_mask)
        self.metric_ihm_aucroc.add(ihm_pred, ihm_label, ihm_mask)

        self.metric_pheno_aucroc.add(pheno_pred, pheno_label)
        self.metric_ltm_aucpr.add(ltm_pred, ltm_label, ltm_mask)
        self.metric_ltm_aucroc.add(ltm_pred, ltm_label, ltm_mask)
        

        self.metric_readmit_acc.add(readmit_pred, readmit_label, readmit_mask)
        self.metric_readmit_aucroc.add(readmit_pred, readmit_label, readmit_mask)
        self.metric_readmit_cfm.add(readmit_pred, readmit_label, readmit_mask)
        end = time.time()
        #print("Time size frac {0} seconds".format((end-start)/los_pred.shape[0]), los_pred.shape)
    
    def cache(self, outputs, num_valid_data):
        '''
        Cache the outpus and num_valid_data for later use, adds for 
        MetricPerHour taking increasingly long, so we cache data and
        dispatch an async thread to deal with cleanup later.
        TODO: Fix the add in MetriPerHour
        '''
        self.outputs_list.append(outputs)
        self.num_valid_data_list.append(num_valid_data)
    
    def execute_add(self):
        outputs_list = self.outputs_list.copy()
        num_valid_data_list = self.num_valid_data_list.copy()
        self.outputs_list = []
        self.num_valid_data_list = []
        start = time.time()
        for outputs, num_valid_data in zip(outputs_list, num_valid_data_list):
            self._add(outputs, num_valid_data)
        #print("Exec took {0} seconds".format(time.time() - start))
        return None

    def add(self):
        '''
        Execute all the chached adds and reset the cache
        '''
        if (len(self.outputs_list) > 0) and (len(self.num_valid_data_list) > 0):
            if self.lock is not None:#If anyone is executing add, wait
                print("Waiting on previous add")
                self.lock.result()
            #self.lock = self.exec.submit(self.execute_add)
            self.execute_add()
   
    def write(self, writer, xpoint):
        """
        Write all metrics in the given writer
        """
        self.add()
        _, _, _, aucpr_decomp = self.metric_decomp_aucpr.get()
        _, _, _, aucroc_decomp = self.metric_decomp_aucroc.get()
        
        if not self.regression:
            roc_auc_ovo, roc_auc_ovr, roc_auc_3_days, roc_auc_7_days = self.metric_los_aucroc.get()
            aucpr_3_days, aucpr_7_days = self.metric_los_aucpr.get()
            cfm = self.metric_los_cfm.get()
            
            
            
        else:
            _, _, _, los_mad = self.metric_los_mad.get()
            _, _, _, los_mae = self.metric_los_mae.get()
            _, _, _, los_ev = self.metric_los_ev.get()

        aucpr_ihm = self.metric_ihm_aucpr.get()
        aucroc_ihm = self.metric_ihm_aucroc.get()
        
        aucroc_micro_pheno = self.metric_pheno_aucroc.get()["micro"]
        aucroc_macro_pheno = self.metric_pheno_aucroc.get()["macro"]

        aucpr_ltm = self.metric_ltm_aucpr.get()
        aucroc_ltm = self.metric_ltm_aucroc.get()
        

        aucroc_ovr_readmit, aucroc_ovo_readmit = self.metric_readmit_aucroc.get()
        acc_readmit = self.metric_readmit_acc.get()
        cfm_readmit = self.metric_readmit_cfm.get()


        if self.split == 'Test':
            self.metric_los_aucroc.save(self.experiment_name+'_los')
            self.metric_decomp_aucpr.save(self.experiment_name+'_decomp')
            self.metric_ihm_aucpr.save(self.experiment_name+'_ihm')
            self.metric_pheno_aucroc.save(self.experiment_name+'_pheno')
            self.metric_ltm_aucpr.save(self.experiment_name+'_ltm')
            self.metric_readmit_aucroc.save(self.experiment_name +'readmit')


        print(self.split)

        print('decomp task aucpr is {:.3f}'.format(aucpr_decomp))
        print('decomp task aucroc is {:.3f}'.format(aucroc_decomp))
        if not self.regression:
            print('los task ovo aucroc is {:.3f}'.format(roc_auc_ovo))
            print('los task ovr aucroc is {:.3f}'.format(roc_auc_ovr))
            print('los task 3 day aucroc is {:.3f}'.format(roc_auc_3_days))
            print('los task 7 day aucroc is {:.3f}'.format(roc_auc_7_days))
            print('los task 3 day aucpr is {:.3f}'.format(aucpr_3_days))
            print('los task 7 day aucpr is {:.3f}'.format(aucpr_7_days))
            print('los confusion matrix is')
            # print(cfm)

            
        else:
            print('los task mad is {:.3f}'.format(los_mad))
            print('los task mae is {:.3f}'.format(los_mae))
            print('los task ev is {:.3f}'.format(los_ev))

        print('ihm task aucroc is {:.3f}'.format(aucroc_ihm))
        print('ihm task aucpr is {:.3f}'.format(aucpr_ihm))
        print('pheno task micro aucroc is {:.3f}'.format(aucroc_micro_pheno))
        print('pheno task macro aucroc is {:.3f}'.format(aucroc_macro_pheno))
        print('ltm task aucpr is {:.3f}'.format(aucpr_ltm))
        print('ltm task aucroc is {:.3f}'.format(aucroc_ltm))
        print('readmit task ovr aucroc is {:.3f}'.format(aucroc_ovr_readmit))
        print('readmit task ovo aucroc is {:.3f}'.format(aucroc_ovo_readmit))
        print('readmit task accuracy is {:.3f}'.format(acc_readmit))
        print(cfm_readmit)
        # print(cfm_ltm)
        


        writer.add_scalar('{0} decomp aucpr'.format(self.split),
                        aucpr_decomp,
                        xpoint)
        writer.add_scalar('{0} decomp aucroc'.format(self.split),
                            aucroc_decomp,
                            xpoint)
        

        if not self.regression:
            writer.add_scalar('{0} los aucroc ovo'.format(self.split),
                                roc_auc_ovo,
                                xpoint)
            writer.add_scalar('{0} los aucroc ovr'.format(self.split),
                                roc_auc_ovr,
                                xpoint)
            writer.add_scalar('{0} los aucroc 3 days'.format(self.split),
                                roc_auc_3_days,
                                xpoint)
            writer.add_scalar('{0} los aucroc 7 days'.format(self.split),
                                roc_auc_7_days,
                                xpoint)
            writer.add_scalar('{0} los aucpr 3 days'.format(self.split),
                                aucpr_3_days,
                                xpoint)
            writer.add_scalar('{0} los aucpr 7 days'.format(self.split),
                                aucpr_7_days,
                                xpoint)
           
        else:
            writer.add_scalar('{0} los mad'.format(self.split),
                            los_mad,
                            xpoint)
            writer.add_scalar('{0} los mae'.format(self.split),
                            los_mae,
                            xpoint)
            writer.add_scalar('{0} los ev'.format(self.split),
                            los_ev,
                            xpoint)

        writer.add_scalar('{0} ihm aucroc'.format(self.split),
                          aucroc_ihm,
                          xpoint)

        writer.add_scalar('{0} ihm aucpr'.format(self.split),
                          aucpr_ihm,
                          xpoint)

        writer.add_scalar('{0} pheno micro aucroc'.format(self.split),
                          aucroc_micro_pheno,
                          xpoint)

        writer.add_scalar('{0} pheno macro aucroc'.format(self.split),
                          aucroc_macro_pheno,
                          xpoint)

        writer.add_scalar('{0} ltm aucroc'.format(self.split),
                          aucroc_ltm,
                          xpoint)

        writer.add_scalar('{0} ltm aucpr'.format(self.split),
                          aucpr_ltm,
                          xpoint)

        writer.add_scalar('{0} readmit aucroc ovr'.format(self.split),
                          aucroc_ovr_readmit,
                          xpoint)
        
        writer.add_scalar('{0} readmit aucroc ovo'.format(self.split),
                          aucroc_ovo_readmit,
                          xpoint)

        writer.add_scalar('{0} readmit acc'.format(self.split),
                          acc_readmit,
                          xpoint)


class BootStrap():
    def __init__(self, experiment_name, seed =1, k=10000, alpha= 5):
        self.k = k
        self.alpha = alpha
        self.seed = seed
        np.random.seed(self.seed)
        self.experiment_name = experiment_name

    def get(self):
        raise NotImplementedError

    def show_res(self, scores, name):
        
        """
        show 2-sided symmetric confidence interval specified
        by p.
        """
        scores.sort()
        print(name+' 50th percentile (median) = %.3f' % median(scores))
        # calculate 95% confidence intervals (100 - alpha)
        # calculate lower percentile (e.g. 2.5)
        lower_p = (self.alpha / 2.0)*0.01
        # retrieve observation at lower percentile
        print(len(scores))
        print(int(np.floor(len(scores)*lower_p)))
        lower = scores[int(np.floor(len(scores)*lower_p))]
        print(name+ ' %.1fth percentile = %.3f' % (lower_p, lower))
        # calculate upper percentile (e.g. 97.5)
        upper_p = ((100 - self.alpha) + (self.alpha / 2.0))*0.01
        # retrieve observation at upper percentile
        upper = scores[int(np.floor(len(scores)*upper_p))]
        print(name+ ' %.1fth percentile = %.3f' % (upper_p, upper))


class BootStrapLos(BootStrap):
    def __init__(self, experiment_name):
        super().__init__(experiment_name = experiment_name)
        self.file_path = os.path.join('./res', self.experiment_name+'_los.pkl')

    def binarilize(self, day, y_true, y_pred):
        if day == 3:
            y_pred = y_pred[:,1]+ y_pred[:,2]
            y_true = np.where(y_true >= 1, 1, 0)
        elif day == 7:
            y_pred = y_pred[:,-1]
            y_true = np.where(y_true == 2, 1, 0)
        return  y_true, y_pred

    def get(self):
        file = open(self.file_path, 'rb')
        pred, label = pickle.load(file)
        file.close()

        # bootstrap
        aucrocs = list()
        aucrocs_3, aucrocs_7 = list(), list()
        aucprs_3, aucprs_7 = list(), list()
        data = np.zeros((label.shape[0], 4))
        data[:,:3] = pred
        data[:,3:] = label
        
        for _ in range(self.k):
            # bootstrap sampleprint(len(label))

            cur_data = sk_utils.resample(data, n_samples=len(data))
            pred, label = cur_data[:,:3], cur_data[:,3:]
            label_3, pred_3 = self.binarilize(3, label, pred)
            label_7, pred_7 = self.binarilize(7, label, pred)
        
            # calculate and store statistic
            (precisions_3, recalls_3, thresholds_3) = metrics.precision_recall_curve(
                label_3, pred_3)
            aucpr_3 = metrics.auc(recalls_3, precisions_3)
            aucprs_3.append(aucpr_3)

            (precisions_7, recalls_7, thresholds_7) = metrics.precision_recall_curve(
                label_7, pred_7)
            aucpr_7 = metrics.auc(recalls_7, precisions_7)
            aucprs_7.append(aucpr_7)

            aucroc_ovr = metrics.roc_auc_score(label, pred, multi_class ='ovr')
            aucroc_3 = metrics.roc_auc_score(label_3, pred_3)
            aucroc_7 = metrics.roc_auc_score(label_7, pred_7)

            aucrocs.append(aucroc_ovr)
            aucrocs_3.append(aucroc_3)
            aucrocs_7.append(aucroc_7)

        self.show_res(aucprs_3, 'aucpr 3 days')
        self.show_res(aucprs_7, 'aucpr 7 days')
        self.show_res(aucrocs, 'aucroc ovr')
        self.show_res(aucrocs_3, 'aucroc 3 days')
        self.show_res(aucrocs_7, 'aucroc 7 days')


class BootStrapDecomp(BootStrap):
    def __init__(self, k, experiment_name):
        super().__init__(experiment_name = experiment_name, k = k)
        self.file_path = os.path.join('./res', self.experiment_name+'_decomp.pkl')

    def get(self):
        file = open(self.file_path, 'rb')
        pred, label = pickle.load(file)
        file.close()

        # bootstrap
        aucrocs = list()
        aucprs = list()
        data = np.zeros((label.shape[0], 2))
        data[:,0] = pred
        data[:,1] = label
        
        for _ in range(self.k):
            # bootstrap sample
            cur_data = sk_utils.resample(data, n_samples=len(data))
            pred = cur_data[:,0]
            label = cur_data[:,1]
            # calculate and store statistic
            (precisions, recalls, thresholds) = metrics.precision_recall_curve(
                label, pred)
            aucpr = metrics.auc(recalls, precisions)
            aucprs.append(aucpr)

            aucroc = metrics.roc_auc_score(label, pred)
            aucrocs.append(aucroc)

        self.show_res(aucprs, 'aucpr decomp')
        self.show_res(aucrocs, 'aucroc decomp')


class BootStrapIhm(BootStrap):
    def __init__(self, experiment_name):
        super().__init__(experiment_name = experiment_name)
        self.file_path = os.path.join('./res', self.experiment_name+'_ihm.pkl')

    def get(self):
        file = open(self.file_path, 'rb')
        pred, label = pickle.load(file)
        file.close()

        # bootstrap
        aucrocs = list()
        aucprs = list()
        data = np.zeros((label.shape[0], 2))
        data[:,0] = pred
        data[:,1] = label
        
        for _ in range(self.k):
            # bootstrap sample
            cur_data = sk_utils.resample(data, n_samples=len(data))
            pred = cur_data[:,0]
            label = cur_data[:,1]
            # calculate and store statistic
            (precisions, recalls, thresholds) = metrics.precision_recall_curve(
                label, pred)
            aucpr = metrics.auc(recalls, precisions)
            aucprs.append(aucpr)

            aucroc = metrics.roc_auc_score(label, pred)
            aucrocs.append(aucroc)

        self.show_res(aucprs, 'aucpr ihm')
        self.show_res(aucrocs, 'aucroc ihm')

class BootStrapLtm(BootStrap):
    def __init__(self, experiment_name):
        super().__init__(experiment_name = experiment_name)
        self.file_path = os.path.join('./res', self.experiment_name+'_ltm.pkl')

    def get(self):
        file = open(self.file_path, 'rb')
        pred, label = pickle.load(file)
        file.close()

        # bootstrap
        aucrocs = list()
        aucprs = list()
        data = np.zeros((label.shape[0], 2))
        data[:,0] = pred
        data[:,1] = label
        
        for _ in range(self.k):
            # bootstrap sample
            cur_data = sk_utils.resample(data, n_samples=len(data))
            pred = cur_data[:,0]
            label = cur_data[:,1]
            # calculate and store statistic
            (precisions, recalls, thresholds) = metrics.precision_recall_curve(
                label, pred)
            aucpr = metrics.auc(recalls, precisions)
            aucprs.append(aucpr)

            aucroc = metrics.roc_auc_score(label, pred)
            aucrocs.append(aucroc)

        self.show_res(aucprs, 'aucpr ltm')
        self.show_res(aucrocs, 'aucroc ltm')

class BootStrapPheno(BootStrap):
    def __init__(self, experiment_name):
        super().__init__(experiment_name = experiment_name)
        self.file_path = os.path.join('./res', self.experiment_name+ '_pheno.pkl')


    def get(self):
        file = open(self.file_path, 'rb')
        pred, label = pickle.load(file)
        file.close()

        print(label.sum(axis=0)/len(label))

        # bootstrap
        macro_aucrocs = list()
        micro_aucrocs = list()

        data = np.zeros((label.shape[0], 50))
        for i in range(0, 25):
            data[:, i] = pred[:,i]
            data[:, 25 + i] = label[:,i]
        
        for _ in range(self.k):
            # bootstrap sampleprint(len(label))

            cur_data = sk_utils.resample(data, n_samples=len(data))
            
            # indices = np.random.choice(np.arange(len(label)),len(label))


            # pred = pred[indices]
            # label = label[indices]
            
            # calculate and store statistic
            
            try:    
                macro_aucroc = metrics.roc_auc_score(cur_data[:,25:], cur_data[:,:25], average = 'macro')

                micro_aucroc = metrics.roc_auc_score(cur_data[:,25:], cur_data[:,:25], average ='micro')
                   

                macro_aucrocs.append(macro_aucroc)
                micro_aucrocs.append(micro_aucroc)
            except:
                continue
                
        self.show_res(macro_aucrocs, 'macro auc roc pheno')
        self.show_res(micro_aucrocs, 'micro auc roc pheno')

