import os


class Config():
    def __init__(self):
        
        self.timestep = 1.0
        self.normalizer_state = 'mult_ts1.0.input_str:previous.start_time:zero.normalizer'
        self.imputation = 'previous'
        self.embeddingspath = './embeds/BioWordVec_PubMed_MIMICIII_d200.vec.bin'
        self.model_path = '/home/luca/mutiltasking-for-mimic3/data/embeds/wv.pkl'
        self.padding_type = 'Zero'
        self.textdata_fixed_train = '/home/luca/mutiltasking-for-mimic3/data/root/text_fixed_train/'
        self.textdata_fixed_test = '/home/luca/mutiltasking-for-mimic3/data/root/text_fixed_test/'
        self.multitask_path = '/home/luca/mutiltasking-for-mimic3/data/expanded_multitask/'
        self.starttime_path_train_val = '/home/luca/mutiltasking-for-mimic3/data/root/train_starttime.pkl'
        self.starttime_path_test = '/home/luca/mutiltasking-for-mimic3/data/root/test_starttime.pkl'


        