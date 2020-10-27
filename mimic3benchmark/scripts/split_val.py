from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import numpy as np
import pandas as pd
import random
random.seed(49297)

def read_listfile(path):
    with open(path, "r") as lfile:
        labels = lfile.readlines()
    data = [line.split(',') for line in labels]
    header = data[0]
    
    data = data[1:]
    return data, header

def split_val_from_train(args):
    val_data, _ = read_listfile(args.val_listfile_path)
    val_filenames = [val_data[i][0] for i in range(len(val_data))]
    
    new_val, new_train = [], []
    train_data, train_header = read_listfile(args.train_listfile_path)
    for i in range(len(train_data)):
        new_line = ','.join(train_data[i])
        #print(new_line)
        if train_data[i][0] in val_filenames:
            new_val.append(new_line)
        else:
            new_train.append(new_line)
    
    with open(args.output_val_listfile_path, "w") as listfile:
        header = ','.join(train_header)
        listfile.write(header)
        for i in range(len(new_val)):
            listfile.write(new_val[i])
    
    with open(args.output_train_listfile_path, "w") as listfile:
        header = ','.join(train_header)
        listfile.write(header)
        for i in range(len(new_train)):
            listfile.write(new_train[i])
    
    





parser = argparse.ArgumentParser(description="Create data for multitask prediction.")
parser.add_argument('--val_listfile_path',  default= '/home/yong/mutiltasking-for-mimic3/data/multitask_updated/train_val/4k_val_listfile.csv' , help="Path to root folder containing val list files.")
parser.add_argument('--train_listfile_path', default= '/home/yong/mutiltasking-for-mimic3/data/multitask_3/train/listfile.csv', help='path to the og train listfile containing val set')
parser.add_argument('--output_val_listfile_path', default = '/home/yong/mutiltasking-for-mimic3/data/multitask_3/train/4k_val_listfile.csv', help='path to the og train listfile containing val set')
parser.add_argument('--output_train_listfile_path',default = '/home/yong/mutiltasking-for-mimic3/data/multitask_3/train/4k_train_listfile.csv', help='path to the og train listfile containing val set')
args = parser.parse_args()
split_val_from_train(args)
