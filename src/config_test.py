import os


class Config():
    def __init__(self):
        self.basepath = '/home/yong/mutiltasking-for-mimic3/data/'
        self.data = '/home/yong/mutiltasking-for-mimic3/data/in-hospital-mortality/'
        self.timestep = 1.0
        self.normalizer_state = 'mult_ts1.0.input_str:previous.start_time:zero.normalizer'
        self.imputation = 'previous'
        self.small_part = True
        self.textdata_train = self.basepath + 'root/text_fixed_train/'
        self.textdata_test = self.basepath + 'root/text_fixed_test'
        self.embeddingspath = './embeds/BioWordVec_PubMed_MIMICIII_d200.vec.bin'
        self.buffer_size = 100
        self.model_path = '/home/yong/mutiltasking-for-mimic3/data/embeds/wv.pkl'
        self.learning_rate = 1e-4
        self.max_len = 300
        self.break_text_at = 300
        self.padding_type = 'Zero'
        self.los_path = '/home/yong/mutiltasking-for-mimic3/data/length-of-stay/'
        self.decompensation_path = '/home/yong/mutiltasking-for-mimic3/data/decompensation/'
        self.ihm_path = '/home/yong/mutiltasking-for-mimic3/data/in-hospital-mortality/'
        self.textdata_fixed_train = '/home/yong/mutiltasking-for-mimic3/data/root/text_fixed_train/'
        self.textdata_fixed_test = '/home/yong/mutiltasking-for-mimic3/data/root/text_fixed_test/'
        self.multitask_path = '/home/yong/mutiltasking-for-mimic3/data/multitask/'
        self.starttime_path_train = '/home/yong/mutiltasking-for-mimic3/data/root/train_starttime.pkl'
        self.starttime_path_test = '/home/yong/mutiltasking-for-mimic3/data/root/test_starttime.pkl'
        self.rnn_hidden_units = 128
        self.maximum_number_events = 150
        self.conv1d_channel_size = 256
        self.dropout = 0.7 
        