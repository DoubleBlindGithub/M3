# import json
# import pandas as pd
# import os
# import os,sys,inspect
# current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir) 
# import config
# import tensorflow as tf
# from sklearn import metrics
# import logging
# import pickle
# import argparse
# import sys
# import numpy as np
# from matplotlib import pyplot
# import time
# from concurrent.futures import ThreadPoolExecutor
# sys.path.insert(0, '..')


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--load_model", help="1/0 to specify whether to load the model", default="0")
#     parser.add_argument(
#         "--number_epoch", default=20, type = int)
#     parser.add_argument(
#         "--batch_size", default=100, type = int)
#     parser.add_argument(
#         "--log_file", default = '../experiment.log')
#     parser.add_argument(
#         "--checkpoint_path", default ='models/ckpts', help="Path for checkpointing")
#     parser.add_argument(
#         "--decay", default=0.1, help='decay term used in weight matrix', type = float)
#     parser.add_argument(
#         "--basepath", default ='/home/luca/mutiltasking-for-mimic3/data/', help='root path of data')
#     parser.add_argument(
#         "--timestep", default =1.0, help ='in hours, we discretize the data in default 1 hour', type = float)
#     parser.add_argument(
#         "--normalizer_state", default = 'mult_ts1.0.input_str:previous.start_time:zero.normalizer',
#         help='presaved normalizer, see scientific report paper for more details')
#     parser.add_argument(
#         "--imputation", default = 'previous', help='imputation stratagy for missing time series data')
#     parser.add_argument(
#         "--small_part", default = "False", help= 'training on a small subset of training data')
#     parser.add_argument(
#         "--textdata_train", default = '/home/luca/mutiltasking-for-mimic3/data/root/text_fixed_train/')
#     parser.add_argument(
#         "--textdata_test", default = '/home/luca/mutiltasking-for-mimic3/data/root/text_fixed_test/')
#     parser.add_argument(
#         "--embeddingspath", default = "./embeds/BioWordVec_PubMed_MIMICIII_d200.vec.bin")
#     parser.add_argument(
#         "--model_path", default = '/home/luca/mutiltasking-for-mimic3/data/embeds/wv.pkl',
#         help = "pretrained embedding model weights")
#     parser.add_argument(
#         "--learning_rate", default = '1e-4')
#     parser.add_argument(
#         "--max_len", default = 300, help = "maximum notes length", type= int)
#     parser.add_argument(
#         "--padding_type", default = "Zero")
#     parser.add_argument(
#         "--multitask_path", default = "/home/luca/mutiltasking-for-mimic3/data/multitask/")
#     parser.add_argument(
#         "--starttime_path_train", default = "/home/luca/mutiltasking-for-mimic3/data/root/train_starttime.pkl",
#         help = "files used to help align time for text data")
#     parser.add_argument(
#         "--starttime_path_test", default = "/home/luca/mutiltasking-for-mimic3/data/root/test_starttime.pkl",
#         help = "files used to help align time for text data")
       
#     args = vars(parser.parse_args())
#     return args


# def get_config():
#     return config.Config()


# def get_embedding_dict(args):
#     with open(args.model_path, 'rb') as f:
#         data = pickle.load(f)

#     index2word_tensor = data["model"]["index2word"]
#     index2word_tensor.pop()
#     index2word_tensor.append('<pad>')
#     word2index_lookup = {word: index for index,
#                          word in enumerate(index2word_tensor)}
#     vectors = data["model"]["vectors"]

#     return vectors, word2index_lookup


# def lookup(w2i_lookup, x):
#     if x in w2i_lookup:
#         return w2i_lookup[x]
#     else:
#         return len(w2i_lookup)


# class AUCROC():
#     def __init__(self, *args, **kwargs):
#         self.y_true = None
#         self.y_pred = None
#     def add(self, pred, y_true):
         
#         if self.y_pred is None:
#             self.y_pred = pred
#         else:
#             self.y_pred = np.concatenate([self.y_pred, pred])

#         if self.y_true is None:
#             self.y_true = y_true
#         else:
#             self.y_true = np.concatenate([self.y_true, y_true])

#     def get(self):
#         # fpr = dict() 
#         # tpr = dict() 
#         roc_auc = dict() 
#         # for i in range(25):
#         #     fpr[i], tpr[i], _ = metrics.roc_curve(self.y_true[:, i], self.y_pred[:, i])
#         #     roc_auc[i] = metrics.auc(fpr[i], tpr[i])
#         # Compute micro-average ROC curve and ROC area
#         #print(self.y_true.shape)
#         roc_auc["macro"] = metrics.roc_auc_score(self.y_true, self.y_pred, average ='macro')
#         roc_auc["micro"] = metrics.roc_auc_score(self.y_true.flatten(), self.y_pred.flatten(), average ='micro')
        
#         return roc_auc

        

# class AUCPR():
#     def __init__(self, *args, **kwargs):
#         self.y_true = None
#         self.y_pred = None
#         self.y_mask = None

#     def add(self, pred, y_true, y_mask):
#         #y_mask = y_mask.astype(np.bool)
#         y_mask = y_mask[:,0]
        
#         if np.sum(y_mask)==0:
#             pass
#         y_true = y_true[y_mask==1]  
#         pred = pred[y_mask==1]
#         if self.y_pred is None:
#             self.y_pred = pred
#         else:
#             self.y_pred = np.concatenate([self.y_pred, pred])

#         if self.y_true is None:
#             self.y_true = y_true
#         else:
#             self.y_true = np.concatenate([self.y_true, y_true])

#     def get(self):
#         (precisions, recalls, thresholds) = metrics.precision_recall_curve(
#             self.y_true, self.y_pred)
#         auprc = metrics.auc(recalls, precisions)
#         return auprc

#     def save(self, name):
#         fname = name + ".pkl"
#         with open(fname, 'wb') as f:
#             pickle.dump((self.y_pred, self.y_true), f, pickle.HIGHEST_PROTOCOL)


# # class MetricPerHour():
# #     def __init__(self):
# #         self.y_true_hr = {}
# #         self.pred_hr = {}
# #         self.aucpr = {}

# #         self.y_true = None
# #         self.y_pred = None

# #         self.metric_type = 'aucpr'

# #     def add(self, pred, y_true, mask, num_valid_data):
# #         # y_true shape (B*T, 1)
# #         # pred shape (B*T, )

# #         pred = pred.reshape(num_valid_data, -1)
# #         y_true = y_true.reshape(num_valid_data, -1)
# #         mask = mask.reshape(num_valid_data, -1)
# #         assert len(pred.shape) == 2, "Pred: {} Y: {} Mask:{}".format(
# #             str(pred.shape), str(y_true.shape), str(mask.shape))
# #         assert len(y_true.shape) == 2
# #         assert len(mask.shape) == 2
# #         y_true_list = []
# #         pred_list = []
# #         for hour in range(5, y_true.shape[1]):
# #             y_true_h = y_true[:, hour]
# #             pred_h = pred[:, hour]
# #             mask_h = mask[:, hour]
# #             if mask_h.sum()==0:
# #                 continue
            
            
# #             mask_h = mask_h.astype(np.bool)
            
# #             y_true_h = y_true_h[mask_h]
            
# #             pred_h = pred_h[mask_h]


# #             if hour not in self.y_true_hr:
# #                 self.y_true_hr[hour] = y_true_h
# #                 self.pred_hr[hour] = pred_h
# #             else:
# #                 self.y_true_hr[hour] = np.concatenate(
# #                     [self.y_true_hr[hour], y_true_h])
# #                 self.pred_hr[hour] = np.concatenate(
# #                     [self.pred_hr[hour], pred_h])

# #             if self.y_true is None:
# #                 self.y_true = y_true_h
# #                 self.y_pred = pred_h
# #             else:
# #                 self.y_true = np.concatenate([self.y_true, y_true_h])
# #                 self.y_pred = np.concatenate([self.y_pred, pred_h])

# #     def get_metric(self, y_true, pred):
# #         if self.metric_type == 'aucpr':
# #             (precisions, recalls, thresholds) = metrics.precision_recall_curve(
# #                 y_true, pred)
# #             value = metrics.auc(recalls, precisions)
# #         elif self.metric_type == 'kappa':
# #             value = metrics.cohen_kappa_score(y_true, pred, weights='linear')
# #         return value

# #     def get(self):
# #         self.aucpr = {}    
# #         aggregated = self.get_metric(self.y_true, self.y_pred)
# #         correct = self.y_pred == self.y_true
# #         print(correct.all())
# #         return self.y_true_hr, self.pred_hr, self.aucpr, aggregated

# #     def save(self, name):
# #         fname = name + ".pkl"
# #         with open(fname, 'wb') as f:
# #             pickle.dump({'aucpr': self.aucpr, 'predbyhr': self.pred_hr, 'truebyhr': self.y_true_hr},
# #                         f, pickle.HIGHEST_PROTOCOL)


# class AUCROCIHM(AUCROC):
#     def add(self, pred, y_true, y_mask):
#         y_mask = y_mask[:, 0]
#         y_true = y_true[y_mask == 1]
#         pred = pred[y_mask == 1]
#         super().add(pred, y_true)
#     def get(self):
#         fpr, tpr, _ =  metrics.roc_curve(self.y_true, self.y_pred)
#         roc_auc = metrics.auc(fpr, tpr)
#         return roc_auc


# class MetricPerHour():
#     def __init__(self):
#         self.y_true_hr = {}
#         self.pred_hr = {}
#         self.aucpr = {}
#         self.y_true = None
#         self.y_pred = None
#         self.metric_type = 'aucpr'

#     def add(self, pred, y_true, mask, num_valid_data):
#         pred = pred.reshape(num_valid_data, -1)
#         y_true = y_true.reshape(num_valid_data, -1)
#         mask = mask.reshape(num_valid_data, -1)
#         assert len(pred.shape) == 2, "Pred: {} Y: {} Mask:{}".format(
#             str(pred.shape), str(y_true.shape), str(mask.shape))
#         assert len(y_true.shape) == 2
#         assert len(mask.shape) == 2
#         y_true_list = []
#         pred_list = []
#         for hour in range(y_true.shape[1]):

#             mask_h = mask[:, hour]
#             if mask_h.sum()< 1:
#                 continue
#             y_true_h = y_true[:, hour]
#             pred_h = pred[:, hour]

#             mask_h = mask_h.astype(np.bool)
            
#             y_true_h = y_true_h[mask_h]
            
#             pred_h = pred_h[mask_h]


#             if hour not in self.y_true_hr:
#                 self.y_true_hr[hour] = y_true_h
#                 self.pred_hr[hour] = pred_h
#             else:
#                 self.y_true_hr[hour] = np.concatenate(
#                     [self.y_true_hr[hour], y_true_h])
#                 self.pred_hr[hour] = np.concatenate(
#                     [self.pred_hr[hour], pred_h])

#             if self.y_true is None:
#                 self.y_true = [y_true_h]
#                 self.y_pred = [pred_h]
#             else:
#                 self.y_true.append(y_true_h) #= np.concatenate([self.y_true, y_true_h])
#                 self.y_pred.append(pred_h) #= np.concatenate([self.y_pred, pred_h])
#                 assert len(self.y_true) == len(self.y_pred)
#                 #len_ytrue = sum(map(lambda x: x.shape[0], self.y_true))
#                 #len_ypred = sum(map(lambda x: x.shape[0], self.y_pred))
#                 #assert len_ytrue == len_ypred

#     def get_metric(self, y_true, pred):
#         #print(self.metric_type)
#         if self.metric_type == 'aucpr':
#             (precisions, recalls, thresholds) = metrics.precision_recall_curve(
#                 y_true, pred)
#             value = metrics.auc(recalls, precisions)
#         elif self.metric_type == 'kappa':
#             #pred = np.zeros_like(y_true)
#             value = metrics.cohen_kappa_score(y_true.flatten(), pred.flatten(), weights='linear')
#         elif self.metric_type == 'aucroc':
#             fpr, tpr, _ = metrics.roc_curve(y_true.ravel(), pred.ravel())
#             value = metrics.auc(fpr, tpr) 
#         elif self.metric_type == 'macro_aucroc':
#             value = metrics.roc_auc_score(y_true.flatten(), pred.flatten())
#         else:
#             value = metrics.accuracy_score(y_true.flatten(), pred.flatten())

#         return value

#     def get(self):
#         self.aucpr = {}  
#         self.y_true = np.concatenate(self.y_true)
#         self.y_pred = np.concatenate(self.y_pred)
#         aggregated = self.get_metric(self.y_true, self.y_pred)
#         return self.y_true_hr, self.pred_hr, self.aucpr, aggregated

# # class MetricPerHour():
# #     def __init__(self):
# #         self.y_true_hr = {}
# #         self.pred_hr = {}
# #         self.aucpr = {}
# #         self.aucroc = {}
# #         self.y_true = None
# #         self.y_pred = None

# #         self.metric_type = 'aucpr'

# #     def add(self, pred, y_true, mask, num_valid_data):
# #         # y_true shape (B*T, 1)
# #         # pred shape (B*T, )
       
# #         pred = pred.reshape(num_valid_data, -1)
        

# #         y_true = y_true.reshape(num_valid_data, -1)
# #         mask = mask.reshape(num_valid_data, -1)
# #         assert len(pred.shape) == 2, "Pred: {} Y: {} Mask:{}".format(
# #             str(pred.shape), str(y_true.shape), str(mask.shape))
# #         assert len(y_true.shape) == 2
# #         assert len(mask.shape) == 2
# #         y_true_list = []
# #         pred_list = []
        
# #         for hour in range(y_true.shape[1]):
# #             y_true_h = y_true[:, hour]
# #             pred_h = pred[:, hour]
# #             mask_h = mask[:, hour]
# #             if mask_h.sum()==0:
# #                 continue
# #             mask_h = mask_h.astype(np.bool)
            
# #             y_true_h = y_true_h[mask_h]
            
# #             pred_h = pred_h[mask_h]
# #             if hour not in self.y_true_hr:
# #                 self.y_true_hr[hour] = y_true_h
# #                 self.pred_hr[hour] = pred_h
# #             else:
# #                 self.y_true_hr[hour] = np.concatenate(
# #                     [self.y_true_hr[hour], y_true_h])
# #                 self.pred_hr[hour] = np.concatenate(
# #                     [self.pred_hr[hour], pred_h])

# #             if self.y_true is None:
# #                 self.y_true = y_true_h
# #                 self.y_pred = pred_h
# #             else:
# #                 self.y_true = np.concatenate([self.y_true, y_true_h])
# #                 self.y_pred = np.concatenate([self.y_pred, pred_h])

# #     def get_metric(self, y_true, pred):
# #         if self.metric_type == 'aucpr':
# #             (precisions, recalls, thresholds) = metrics.precision_recall_curve(
# #                 y_true, pred)
# #             value = metrics.auc(recalls, precisions)
# #         elif self.metric_type == 'kappa':
# #             value = metrics.cohen_kappa_score(y_true, pred, weights='linear')
# #         else:
# #             fpr, tpr, _ = metrics.roc_curve(y_true.ravel(), pred.ravel())
# #             value = metrics.auc(fpr, tpr) 
# #         return value

# #     def get(self):
# #         self.aucpr = {}
# #         aggregated = self.get_metric(self.y_true, self.y_pred)
# #         correct = self.y_pred == self.y_true
# #         return self.y_true_hr, self.pred_hr, self.aucpr, aggregated

# #     def save(self, name):
# #         fname = name + ".pkl"
# #         with open(fname, 'wb') as f:
# #             pickle.dump({'aucpr': self.aucpr, 'predbyhr': self.pred_hr, 'truebyhr': self.y_true_hr},
# #                         f, pickle.HIGHEST_PROTOCOL)




# class AUCPRperHour(MetricPerHour):
#     def __init__(self):
#         super().__init__()
#         self.metric_type = 'aucpr'


# class KappaPerHour(MetricPerHour):
#     def __init__(self):
#         super().__init__()
#         self.metric_type = 'kappa'


# class AUCROCPerHour(MetricPerHour):
#     def __init__(self):
#         super().__init__()
#         self.metric_type = 'aucroc'

# class AUCROCMacroPerHour(MetricPerHour):
#     def __init__(self):
#         super().__init__()
#         self.metric_type = 'macro_aucroc'

# class ACCPerHour(MetricPerHour):
#     def __init__(self):
#         super().__init__()
#         self.metric_type = 'acc'

# class EpochWriter:
#     """
#     Hold all metrics for an entire epoch of training
#     """

#     def __init__(self, split):
#         self.split = split
#         self.metric_decomp_aucpr = AUCPRperHour()
#         self.metric_decomp_aucroc = AUCROCPerHour()
#         self.metric_los_kappa = KappaPerHour()
#         self.metric_los_acc = ACCPerHour()
#         self.metric_ihm_aucpr = AUCPR()
#         self.metric_ihm_aucroc = AUCROCIHM()
#         self.metric_pheno_aucroc = AUCROC() 
#         self.outputs_list = []
#         self.num_valid_data_list = []
#         self.exec = ThreadPoolExecutor(max_workers = 1)
#         self.lock = None
    
#     def _add(self, outputs, num_valid_data):
#         """
#         Add the preds and labels from outputs to their respective 
#         metric classes

#         :param: outputs, dict of tasks, each element is a dict of preds/labels/
#                 masks for that given task
#         """
#         start = time.time()
#         decomp_pred = outputs['decomp']['pred']
#         decomp_label = outputs['decomp']['label']
#         decomp_mask = outputs['decomp']['mask']

#         ihm_pred = outputs['ihm']['pred']
#         ihm_label = outputs['ihm']['label']
#         ihm_mask = outputs['ihm']['mask']

#         los_pred = outputs['los']['pred']
#         los_label = outputs['los']['label']
#         los_mask = outputs['los']['mask']

#         pheno_pred = outputs['pheno']['pred']
#         pheno_label = outputs['pheno']['label']
#         pheno_mask = outputs['pheno']['mask']

#         self.metric_decomp_aucpr.add(decomp_pred, decomp_label,
#                                 decomp_mask, num_valid_data)
#         self.metric_decomp_aucroc.add(decomp_pred, decomp_label,
#                                     decomp_mask, num_valid_data)
#         self.metric_los_acc.add(los_pred, los_label, los_mask, num_valid_data)

#         self.metric_los_kappa.add(los_pred, los_label, los_mask, num_valid_data)


#         self.metric_ihm_aucpr.add(ihm_pred, ihm_label, ihm_mask)
#         self.metric_ihm_aucroc.add(ihm_pred, ihm_label, ihm_mask)

#         self.metric_pheno_aucroc.add(pheno_pred, pheno_label)
#         end = time.time()
#         #print("Time size frac {0} seconds".format((end-start)/los_pred.shape[0]), los_pred.shape)
    
#     def cache(self, outputs, num_valid_data):
#         '''
#         Cache the outpus and num_valid_data for later use, adds for 
#         MetricPerHour taking increasingly long, so we cache data and
#         dispatch an async thread to deal with cleanup later.
#         TODO: Fix the add in MetriPerHour
#         '''
#         self.outputs_list.append(outputs)
#         self.num_valid_data_list.append(num_valid_data)
    
#     def execute_add(self):
#         outputs_list = self.outputs_list.copy()
#         num_valid_data_list = self.num_valid_data_list.copy()
#         self.outputs_list = []
#         self.num_valid_data_list = []
#         start = time.time()
#         for outputs, num_valid_data in zip(outputs_list, num_valid_data_list):
#             self._add(outputs, num_valid_data)
#         #print("Exec took {0} seconds".format(time.time() - start))
#         return None

    
#     def add(self):
#         '''
#         Execute all the chached adds and reset the cache
#         '''
#         if (len(self.outputs_list) > 0) and (len(self.num_valid_data_list) > 0):
#             if self.lock is not None:#If anyone is executing add, wait
#                 print("Waiting on previous add")
#                 self.lock.result()
#             #self.lock = self.exec.submit(self.execute_add)
#             self.execute_add()
   
#     def write(self, writer, xpoint):
#         """
#         Write all metrics in the given writer
#         """
#         self.add()


#         _, _, _, aucpr_decomp = self.metric_decomp_aucpr.get()
#         _, _, _, aucroc_decomp = self.metric_decomp_aucroc.get()
#         _, _, _, los_acc = self.metric_los_acc.get()


#         _, _, _, kappa_score = self.metric_los_kappa.get()

#         aucpr_ihm = self.metric_ihm_aucpr.get()
#         aucroc_ihm = self.metric_ihm_aucroc.get()
#         aucroc_micro_pheno = self.metric_pheno_aucroc.get()["micro"]
#         aucroc_macro_pheno = self.metric_pheno_aucroc.get()["macro"]

#         print(self.split)

#         print('decomp task aucpr is {:.3f}'.format(aucpr_decomp))
#         print('decomp task aucroc is {:.3f}'.format(aucroc_decomp))
 
#         print('los task kappa is {:.3f}'.format(kappa_score))
#         print('ihm task aucroc is {:.3f}'.format(aucroc_ihm))
#         print('ihm task aucpr is {:.3f}'.format(aucpr_ihm))
#         print('pheno task micro aucroc is {:.3f}'.format(aucroc_micro_pheno))
#         print('pheno task macro aucroc is {:.3f}'.format(aucroc_macro_pheno))



#         writer.add_scalar('{0} decomp aucpr'.format(self.split),
#                         aucpr_decomp,
#                         xpoint)
#         writer.add_scalar('{0} decomp aucroc'.format(self.split),
#                             aucroc_decomp,
#                             xpoint)
#         writer.add_scalar('{0} los acc'.format(self.split),
#                             los_acc,
#                             xpoint)


#         writer.add_scalar('{0} los kappa'.format(self.split),
#                           kappa_score,
#                           xpoint)

#         writer.add_scalar('{0} ihm aucroc'.format(self.split),
#                           aucroc_ihm,
#                           xpoint)

#         writer.add_scalar('{0} ihm aucpr'.format(self.split),
#                           aucpr_ihm,
#                           xpoint)

#         writer.add_scalar('{0} pheno micro aucroc'.format(self.split),
#                           aucroc_micro_pheno,
#                           xpoint)

#         writer.add_scalar('{0} pheno macro aucroc'.format(self.split),
#                           aucroc_macro_pheno,
#                           xpoint)


# def create_logger(name, to_disk = True, log_file = None):
#     """Create a new logger"""
#     # setup logger
#     log = logging.getLogger(name)
#     log.setLevel(logging.DEBUG)
#     log.propagate = False
#     formatter = logging.Formatter(fmt='%(message)s', datefmt='%Y/%m/%d %I:%M:%S')
#     if to_disk:
#         log_file = log_file if log_file is not None else strftime("log/log_%m%d_%H%M.txt", gmtime())
#         if type(log_file) == list:
#             for filename in log_file:
#                 fh = logging.FileHandler(filename, mode='w')
#                 fh.setLevel(logging.INFO)
#                 fh.setFormatter(formatter)
#                 log.addHandler(fh)
#         if type(log_file) == str:
#             fh = logging.FileHandler(log_file, mode='w')
#             fh.setLevel(logging.INFO)
#             fh.setFormatter(formatter)
#             log.addHandler(fh)
#     return log