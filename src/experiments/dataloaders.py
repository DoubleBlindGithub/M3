import torch
import os
import sys
sys.path.append("../../..")
sys.path.append("../..")
sys.path.append("..")
import pandas as pd 
import numpy as np
from torch.utils.data import Dataset, DataLoader
import utils
import pickle
from mimic3benchmark.readers import MultitaskReader
from mimic3models.preprocessing import Discretizer, Normalizer
import time
import functools
import json
import matplotlib.pyplot as plt

def lookup(w2i_lookup, x):
    if x in w2i_lookup:
        return w2i_lookup[x]
    else:
        return len(w2i_lookup)


def diff(time1, time2):
    """
    compute time2-time1
    return difference in hours
    """
    a = np.datetime64(time1)
    b = np.datetime64(time2)
    h = (b-a).astype('timedelta64[h]').astype(int)
    
    return h


def get_bin_log(x, nbins, one_hot=False):
    """
    this function is used to generate los stay labels,
    in general, this function split continous los values into bins
    """
    binid = int(np.log(x + 1) / 8.0 * nbins)
    if binid < 0:
        binid = 0
    if binid >= nbins:
        binid = nbins - 1

    if one_hot:
        ret = np.zeros((LogBins.nbins,))
        ret[binid] = 1
        return ret
    return binid

class CustomBins:
    inf = 1e18
    bins = [(-inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, +inf)]
    nbins = len(bins)
    means = [11.450379, 35.070846, 59.206531, 83.382723, 107.487817,
             131.579534, 155.643957, 179.660558, 254.306624, 585.325890]


def get_bin_custom(x, nbins, one_hot=False):
    for i in range(nbins):
        a = CustomBins.bins[i][0] * 24.0
        b = CustomBins.bins[i][1] * 24.0
        if a <= x < b:
            if one_hot:
                ret = np.zeros((CustomBins.nbins,))
                ret[i] = 1
                return ret
            return i
    return None


class CoarseBins:
    inf = 1e18
    bins = [(-inf, 3), (3, 7), (7, +inf)]
    nbins = len(bins)
    

def get_bin_coarse(x, nbins, one_hot=False):
    for i in range(nbins):
        a = CoarseBins.bins[i][0] * 24.0
        b = CoarseBins.bins[i][1] * 24.0
        if a <= x < b:
            if one_hot:
                ret = np.zeros((CoarseBins.nbins,))
                ret[i] = 1
                return ret
            return i
    return None

class WeightBins:
    inf = 1e18
    step_size = 5
    start = 10
    stop = 200
    bins = [[x, x + 5] for x in range(start, stop, step_size)]
    bins.insert(0, (-inf, start))
    bins.append((stop+step_size, inf))
    nbins = len(bins)

def get_weight_bin(x, nbins=WeightBins.nbins):
    for i in range(nbins):
        a = WeightBins.bins[i][0]
        b = WeightBins.bins[i][1]
        if a <= x < b:
            return i
    return None



def pad_zeros(arr, min_length=None):
    """
    `arr` is an array of `np.array`s

    The function appends zeros to every `np.array` in `arr`
    to equalize their first axis lenghts.
    """
    dtype = arr[0].dtype
    ret = arr
    if (min_length is not None) and arr.shape[0] < min_length and len(arr.shape)==1:
        ret = np.concatenate([arr, np.zeros((min_length - arr.shape[0]),dtype=dtype)], axis=0)
    # pad ts data
    elif (min_length is not None) and arr.shape[0] < min_length and len(arr.shape)==2:
        ret = np.concatenate([arr, np.zeros((min_length - arr.shape[0], arr.shape[1]),dtype=dtype)], axis=0)
    return np.array(ret)


def assert_shapes(X, mask, output, TimeMask = None, Texts = None):
    assert output.shape[0] == X.shape[0]
    assert mask.shape[0] == X.shape[0]
    assert output.shape[1] == mask.shape[1]
    assert output.shape[1] == X.shape[1]
    if TimeMask is not None:
        assert TimeMask.shape[0] == X.shape[0]
        assert TimeMask.shape[1] == X.shape[1]
        assert Texts.shape[0] == X.shape[0]
        assert Texts.shape[1] <= X.shape[1], "Texts.shape:{}, X.shape: {}".format(
            str(Texts.shape), str(X.shape))
        assert Texts.shape[1] == TimeMask.shape[2], "Texts.shape:{}, TimeMask.shape: {}".format(
            str(Texts.shape), str(TimeMask.shape))


class MultiModal_Dataset(Dataset):
    def __init__(self, ts_root_dir, text_root_dir,tabular_root_dir, listfile, discretizer, starttime_path, regression, bin_type,\
         normalizer, ihm_pos, ihm_gap_time, los_pos, los_gap_time, use_text, use_ts, use_tab, decay, w2i_lookup, max_text_length, max_num_notes):
        self.ts_root_dir = ts_root_dir
        self.text_root_dir = text_root_dir
        self.tabular_root_dir = tabular_root_dir
        self.ihm_pos = ihm_pos
        self.los_pos = los_pos
        self.use_text = use_text
        self.use_ts = use_ts
        self.use_tab = use_tab
        self.decay = decay
        self.w2i_lookup = w2i_lookup
        self.max_text_length = max_text_length
        self.max_num_notes = max_num_notes
        self.regression = regression
        self.bin_type = bin_type
        self.ihm_gap_time = ihm_gap_time
        self.los_gap_time = los_gap_time
        with open(starttime_path, 'rb') as f:
            self.episodeToStartTime = pickle.load(f)
        
        ts_listfile_path = os.path.join(self.ts_root_dir, listfile)
        with open(ts_listfile_path, "r") as lfile:
            self._ts_data = lfile.readlines()
        self._ts_data = [line.split(',') for line in self._ts_data]
        self._listfile_header = self._ts_data[0]
        
        self._ts_data = self._ts_data[1:]
        self._text_data = os.listdir(self.text_root_dir)

        self.discretizer = discretizer
        self.normalizer = normalizer

        def process_ihm(x):
            return list(map(int, x.split(';')))

        def process_los(x):
            los = list(map(float, x.split(';')))
            return [float(x) for x in los]

        def process_ph(x):
            return list(map(int, x.split(';')))

        def process_decomp(x):
            x = x.split(';')
            if x[0] == '':
                return ([], [])
            return (list(map(int, x[:len(x)//2])), list(map(int, x[len(x)//2:])))

        def process_readmit(x):
            return list(map(int, x.split(';')))

        def process_ltm(x):
            return list(map(int, x.split(';')))

        self._ts_data = [(fname, float(t), process_ihm(ihm), process_los(los),
                       process_ph(pheno), process_decomp(decomp), process_readmit(readmit), process_ltm(ltm))
                      for fname, t, ihm, los, pheno, decomp, readmit, ltm in self._ts_data]
        
        self.map_ethnicity = {
        "AMERICAN INDIAN/ALASKA NATIVE": 1,
        "AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE": 2,
        "ASIAN": 3,
        "ASIAN - ASIAN INDIAN": 4,
        "ASIAN - CAMBODIAN": 5,
        "ASIAN - CHINESE": 6,
        "ASIAN - FILIPINO": 7,
        "ASIAN - JAPANESE": 8,
        "ASIAN - KOREAN": 9,
        "ASIAN - OTHER": 10,
        "ASIAN - THAI": 11,
        "ASIAN - VIETNAMESE": 12,
        "BLACK/AFRICAN": 13,
        "BLACK/AFRICAN AMERICAN": 14,
        "BLACK/CAPE VERDEAN": 15,
        "BLACK/HAITIAN": 16,
        "CARIBBEAN ISLAND": 17,
        "HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)": 18,
        "HISPANIC/LATINO - COLOMBIAN": 19,
        "HISPANIC/LATINO - CUBAN": 20,
        "HISPANIC/LATINO - DOMINICAN": 21,
        "HISPANIC/LATINO - GUATEMALAN": 22,
        "HISPANIC/LATINO - HONDURAN": 23,
        "HISPANIC/LATINO - MEXICAN": 24,
        "HISPANIC/LATINO - PUERTO RICAN": 25,
        "HISPANIC/LATINO - SALVADORAN": 26,
        "HISPANIC OR LATINO": 27,
        "MIDDLE EASTERN": 28,
        "MULTI RACE ETHNICITY": 29,
        "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER": 30,
        "OTHER": 0,
        "PATIENT DECLINED TO ANSWER": 0,
        "PORTUGUESE": 32,
        "SOUTH AMERICAN": 33,
        "UNABLE TO OBTAIN": 0,
        "UNKNOWN/NOT SPECIFIED": 0,
        "WHITE": 34,
        "WHITE - BRAZILIAN": 35,
        "WHITE - EASTERN EUROPEAN": 36,
        "WHITE - OTHER EUROPEAN": 37,
        "WHITE - RUSSIAN": 38
        }
        self.map_gender = {
            'F': 1,
            'M': 2,
            'OTHER': 3,
            '': 0
        }
        self.map_careunit = {
            'CCU': 1,
            'CSRU': 2,
            'MICU': 3,
            'NICU': 4,
            'NWARD': 5,
            'SICU': 6,
            'TSICU': 7
        }
        self.map_dbsource = {
            'both': 1,
            'carevue': 2,
            'metavision': 3
        }
        
    
    def read_ts(self, idx):
        """ Reads the example with given index.

        :args idx: Index of the line of the listfile to read (counting starts from 0).
        :return: Return dictionary with the following keys:
            X : np.array
                2D array containing all events. Each row corresponds to a moment.
                First column is the time and other columns correspond to different
                variables.
            t : float
                Length of the data in hours. Note, in general, it is not equal to the
                timestamp of last event.
            ihm : array
                Array of 3 integers: [pos, mask, label].
            los : array
                Array of 2 arrays: [masks, labels].
            pheno : array
                Array of 25 binary integers (phenotype labels).
            decomp : array
                Array of 2 arrays: [masks, labels].
            header : array of strings
                Names of the columns. The ordering of the columns is always the same.
            name: Name of the sample.
        """
        if idx < 0 or idx >= len(self._ts_data):
            raise ValueError("Index must be from 0 (inclusive) to number of lines (exclusive).")
        fname = self._ts_data[idx][0]
        ret = []
        with open(os.path.join(self.ts_root_dir, fname), "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            assert header[0] == "Hours"
            for line in tsfile:
                mas = line.strip().split(',')
                ret.append(np.array(mas))
         
        X, header = np.stack(ret), header
        ts_dict= {"X": X,
                "t": self._ts_data[idx][1],
                "ihm": self._ts_data[idx][2],
                "los": self._ts_data[idx][3],
                "pheno": self._ts_data[idx][4],
                "decomp": self._ts_data[idx][5],
                "readmit": self._ts_data[idx][6],
                "ltm": self._ts_data[idx][7],
                "header": header,
                "name": fname}
        
        return self.process_ts(ts_dict)


    def process_ts(self, ts_dict):
        max_time = ts_dict['t']
        X = ts_dict['X']
        ihm = ts_dict['ihm']
        los = ts_dict['los']
        pheno = ts_dict['pheno']
        decomp = ts_dict['decomp']
        readmit = ts_dict['readmit']
        ltm = ts_dict['ltm']
        name = ts_dict['name']

        timestep = self.discretizer._timestep
        eps = 1e-6

        def get_bin(t):
            return int(t / timestep - eps)

        n_steps = get_bin(max_time) + 1
        # X
        X = self.discretizer.transform(X, end=max_time)[0]
        if self.normalizer is not None:
            X = self.normalizer.transform(X)
        assert len(X) == n_steps
        X = pad_zeros(X, min_length=self.ihm_pos + 1)
        T = X.shape[0]

        # ihm
        # NOTE: when mask is 0, we set y to be 0. This is important
        #       because in the multitask networks when ihm_M = 0 we set
        #       our prediction thus the loss will be 0.

        if np.equal(ihm[1], 0):
            ihm[2] = 0
        ihm_M = np.int32(ihm[1])
        ihm_M = np.expand_dims(ihm_M, axis=-1)
        ihm_y = np.int32(ihm[2])  
        ihm_y = np.expand_dims(ihm_y, axis=-1)

        # decomp
        decomp_M = [0] * n_steps
        decomp_y = [0] * n_steps

        for i in range(len(decomp[0])):
            pos = get_bin(i)
            decomp_M[pos] = decomp[0][i]
            decomp_y[pos] = decomp[1][i]
        decomp_M = np.array(decomp_M, dtype=np.int32)
        decomp_M = pad_zeros(decomp_M, min_length=self.ihm_pos + 1)
        decomp_M = np.expand_dims(decomp_M, axis=-1)
        decomp_y = np.array(decomp_y, dtype=np.int32)
        decomp_y = pad_zeros(decomp_y, min_length=self.ihm_pos + 1)
        decomp_y = np.expand_dims(decomp_y, axis=-1)

        #los
        if np.equal(los[1], 0):
            los[2] = 0
        los_M = np.int32(los[1])
        los_M = np.expand_dims(los_M, axis=-1)
        los_y = np.int32(los[2])  
        los_y = np.expand_dims(los_y, axis=-1)
       
        if not self.regression:
            if self.bin_type == 'custom':
                los_y = np.array([get_bin_custom(los_y, 10)]) 
            elif self.bin_type == 'log':
                los_y = np.array([get_bin_log(los_y, 10)])
            elif self.bin_type == 'coarse':
                los_y = np.array([get_bin_coarse(los_y, 3)])
        else:
            los_y = np.array(float(los_y))
        
        # pheno
        pheno_y = np.array(pheno, dtype=np.int32)

        # readmit
        if np.equal(readmit[0], 0):
            readmit[1] = 0
       
        readmit_M = np.int32(readmit[0])
        readmit_M = np.expand_dims(readmit_M, axis=-1)
        readmit_y = np.int32(readmit[1])  
        readmit_y = np.expand_dims(readmit_y, axis=-1)

        # ltm
        if np.equal(ltm[0], 0):
            ltm[1] = 0
        
        ltm_M = np.int32(ltm[0])
        ltm_M = np.expand_dims(ltm_M, axis =-1)
        ltm_y = np.int32(ltm[1])
        ltm_y = np.expand_dims(ltm_y, axis =-1)


        return X, T, ihm_M, ihm_y, decomp_M, decomp_y, los_M, los_y, pheno, readmit_M, readmit_y, ltm_M, ltm_y, name


    def read_text_event_json(self, text_fname):
        """
        read one single icu stay notes text
        return: a list of absolute times and a list of the notes associated with the corrrespoding time stamp
        """
        filepath = os.path.join(self.text_root_dir, str(text_fname))
        with open(filepath, 'r') as f:
            d = json.load(f)
        times = sorted(d.keys())
        #print(times[-1])
        texts = []
        for t in times:
            texts.append(" ".join(d[t]))
        assert len(times) == len(texts)
        
        return times, texts


    def read_text(self, idx, T):
        """
        retrieve text based on the fname in ts data list, ts data is the sup set of wf and text data,
        so use ts data as golden reference
        return : 
        """

            
        fname = self._ts_data[idx][0]
        pid, eid = self.get_id_from_filename(fname)
        text_fname = pid + "_" + eid + '.json'
        
        if text_fname in self._text_data:
            times, texts = self.read_text_event_json(text_fname)
            hours, texts = self.process_text(times, texts, text_fname, T)
            return hours, texts
        else:
            return None, None


    def process_text(self, times, texts, text_fname, T):
        """
        convert absolute time to relative time in hours and validate the raw text, namely
        concat text in the same hour if there are multiple notes in a hour
        remove text taken before hour 5
        """
        text_fname = text_fname.replace('.json', '')
        start_time = self.episodeToStartTime[text_fname]
        
        if len(texts) == 0 or start_time == -1:
            return None, None
        prev_h = 5
        prev_text = ""
        out_hours, out_texts = [], []
        notes_count = 0
        for (t, txt) in zip(times, texts):
            if t.startswith("ds"):
                
                h = T
            else:
                h = diff(start_time, t)
            notes_count +=1
            if notes_count> self.max_num_notes:
                break
            if h < 5:
                continue
            if h == prev_h:
                prev_text += txt
            else:
                out_hours.append(prev_h)
                out_texts.append(prev_text)
                
                prev_h = h
                prev_text = txt
        out_hours.append(prev_h)
        out_texts.append(prev_text)
        return out_hours, out_texts


    def get_id_from_filename(self, fname):
        # '24610_episode1_timeseries.csv'
        tokens = fname.split('_')
        pid = tokens[0]
        episode_id = tokens[1].replace('episode', '').strip()
        return pid, episode_id


    def get_weight_matrix(self, total_time, hours, decay):
        """
        generate a weight matrix that will be used in weigthed combination of waveform features
        args:
        hours: when the notes are taken, a list
        total_time: all time points in hours for the icu stay
        --------------------------------------------------------
        return: 
        a weight matrix used for weighted combination of text/waveform features
        """
        if hours is None:
            return None
        
        weight_matrix = np.zeros((total_time, len(hours)))
        
        for t in range(0, total_time):
            for i, h in enumerate(hours):
                # print('h',h)
                # print(total_time)
                if h>t+1:
                    continue
                weight_matrix[t][i] = t-h+2
                assert weight_matrix[t][i] >= 0
        ones = np.ones_like(weight_matrix)
        binary_mask = np.where(weight_matrix==0, weight_matrix, ones)
        weight_decay = np.exp(-decay*weight_matrix)
        weight_decay_clipped = np.clip(weight_decay, a_min= 1e-10 , a_max=1)
        final_weight = weight_decay_clipped * binary_mask
        return final_weight
    


    def convert_text_2_idx(self, text_list, w2i_lookup, max_len):
        """
        convert raw text into text index
        -------------------------------------------------------------------------------------------
        args:
        text_list: 2d list of raw text, 1 dimesion: num of notes. 2 dimension: word token for each notes
        w2i_lookup: lookup table of words
        max_len: max_length of text, some notes are very long, we need to truncate these notes
        --------------------------------------------------------------------------------------------
        return:
        a numpy array of text idx
        """

        list_of_idx_all_notes = []
        max_words = 0

        if text_list is None or len(text_list) == 0:
            return None

        else:
            
            num_notes = len(text_list)
            for note in text_list:
                # each note is a list of word
                indices = list(map(lambda x: lookup(w2i_lookup, x), str(note).split()))
                if max_len > 0:
                    indices = indices[:max_len]
                list_of_idx_all_notes.append(indices)
                max_words = max(len(indices), max_words)
                
            pad_token = w2i_lookup['<pad>']
            
            if num_notes == 0 or max_words <= 4:
                # in case no valid text for a given icu stay all notes are too short to support bigram or trigam convolution
                max_notes =1
                max_words =max_len

            def extra_pad_tokens(cnt): return [pad_token]*cnt

            padded_list_of_indices = []
            
            for note in list_of_idx_all_notes:
                #note = note + extra_pad_tokens(max_words - len(note))
                note = note + extra_pad_tokens(max_len - len(note))
                padded_list_of_indices.append(note)
            
            x = np.array(padded_list_of_indices)
            
            try:
                assert len(x.shape) == 2
                return x
            except:
                print('bad shape of x', x.shape)
    
    
    def read_tab(self, fname):
        """
        Read tabular data for givenfilename
        """
        pid, episode_id = self.get_id_from_filename(fname)
        p_dir = os.path.join(self.tabular_root_dir, pid)
        stays_df = pd.read_csv(os.path.join(p_dir, 'stays.csv'))
        stays_data = stays_df.loc[int(episode_id)-1]#0 indexed
        careunit = self.map_careunit[stays_data['LAST_CAREUNIT']]#categorical
        dbsource = self.map_dbsource[stays_data['DBSOURCE']]#categorial
        ethnicity = self.map_ethnicity[stays_data['ETHNICITY']]#categorical
        gender = self.map_gender[stays_data['GENDER']]#categorical
        age = stays_data['AGE']#come back and bin?
        if age == '':
            age = 0
        else:
            age = int(age)
        supplemental = pd.read_csv(os.path.join(p_dir, 'episode'+episode_id+'.csv'))
        height = supplemental['Height']#Come back and bin?
        if type(height) is not float or np.isnan(height):
            height = 0
        else:
            height = int(height)
        init_weight = get_weight_bin(supplemental['Weight'][0])

        return {"careunit": careunit, 
                "dbsource": dbsource, 
                "ethnicity": ethnicity, 
                "gender": gender, 
                "age": age,
                "height": height,
                "weight": init_weight
                }



    def __len__(self):
        return len(self._ts_data)


    def __getitem__(self, idx):
        data = {}
        X, T, ihm_M, ihm_y, decomp_M, decomp_y, los_M, los_y, pheno_y, readmit_M, readmit_y, ltm_M, ltm_y, name = self.read_ts(idx)
        
        if self.use_ts:
            data['time series'] = X
        data['t'] = T
        data['ihm mask'] = ihm_M * (T >= self.ihm_pos + self.ihm_gap_time)
        data['ihm label'] = ihm_y
        data['decomp mask'] = decomp_M
        data['decomp label'] = decomp_y
        data['los mask'] = los_M * (T >= self.los_pos + self.los_gap_time)
        data['los label'] = los_y
        data['pheno label'] = pheno_y
        data['readmit mask'] = readmit_M
        data['readmit label'] = readmit_y
        data['ltm mask'] = ltm_M
        data['ltm label'] = ltm_y
        data['name'] = name
        
        
        if self.use_text:
            text_hours, texts = self.read_text(idx, T)
            #data['raw text'] = list(zip(text_hours, texts))
            texts_weight_mat = self.get_weight_matrix(T, text_hours, self.decay)
            texts_idx = self.convert_text_2_idx(texts, self.w2i_lookup, self.max_text_length)
            data['texts'] = texts_idx
            data['texts weight matrix'] = texts_weight_mat
        
        if self.use_tab:
            tab_data = self.read_tab(name)
            data['tab'] = tab_data

        return data


def padding_batch(data):
    """
    padding data in a batch with variable length, the input data is supposed to be a list
    the elment in data list is supposed to be a 2d numpy array, for example, for texts, it is upposed to be
    in shape num_note * max_note_len, for los label, it is supposed to be in shape T * 1, for ts data,
    the shape is T*76, for text weight matrix, the shape is T * num_notes
    """

    shape_0 = [i.shape[0] for i in data]
    shape_1 = [i.shape[1] for i in data]
    max_shape_0 = max(shape_0)
    max_shape_1 = max(shape_1)
    res = []
    for item in data:
        padded_item = np.zeros((max_shape_0, max_shape_1))
        padded_item[:item.shape[0], :item.shape[1]] = item
        res.append(padded_item)
    return np.stack(res, axis =0)




def custom_collate_fn(batch, union = True):
    """
    custom collate function, used for padding ts data, text data, wf data text weight matrix, waveform weight matrix
    labels, masks in a batch since it may be of different length
    """
    
    final_output = {}

    if not union:
        # return intersection of the modality
        new_batch = []
        for item in batch:
            if 'texts' in item:
                if item['texts'] is None:
                    continue
            if 'waveforms' in item:
                if item['waveforms'] is None:
                    continue
                 
            new_batch.append(item)
        if len(new_batch) == 0:
            # if no data in a batch that have all three modality, return None
            return None

    else:
        new_batch = []
        for item in batch:
            T = item['t']
            if 'texts' in item:
                if item['texts'] is None or item['texts weight matrix'] is None:
                    item['texts'] = np.ones((1, 300))
                    item['texts weight matrix'] = np.zeros((T,1))
            if 'waveforms' in item:
                if item['waveforms'] is None or item['waveforms weight matrix'] is None:
                    item['waveforms'] = np.zeros((1, 100))
                    item['waveforms weight matrix'] = np.zeros((T,1))
            new_batch.append(item)
    
           
    
    ihm_M = [item['ihm mask'] for item in new_batch]
    ihm_y = [item['ihm label'] for item in new_batch]
    readmit_M = [item['readmit mask'] for item in new_batch]
    readmit_y = [item['readmit label'] for item in new_batch]
    ltm_M = [item['ltm mask'] for item in new_batch]
    ltm_y = [item['ltm label'] for item in new_batch]
    decomp_M = [item['decomp mask'] for item in new_batch]
    decomp_y = [item['decomp label'] for item in new_batch]
    los_M = [item['los mask'] for item in new_batch]
    los_y = [item['los label'] for item in new_batch]
    pheno_y = [item['pheno label'] for item in new_batch]
    name = [item['name'] for item in new_batch]

    if 'time series' in new_batch[0]:
        X = [item['time series'] for item in new_batch]
        X_padded = padding_batch(X)
        T = X_padded.shape[1]
    if 'texts' in new_batch[0]:
        #raw_text = [item['raw text'] for item in new_batch]
        texts_weight_mat = [item['texts weight matrix'] for item in new_batch]
        texts_idx = [item['texts'] for item in new_batch]
        texts_idx = padding_batch(texts_idx)
        texts_weight_mat = padding_batch(texts_weight_mat)
    if 'tab' in new_batch[0]:
        tabs = [item['tab'] for item in new_batch]
        tabs_data = {cat: [] for cat in tabs[0]}
        for pat in tabs:
            for cat in pat:
                tabs_data[cat].append(pat[cat])
        for cat in tabs_data:
            tabs_data[cat] = np.array(tabs_data[cat], dtype=np.float64)
        tab_data = tabs_data #Not variable length, don't have to pad
    
    
    decomp_M = padding_batch(decomp_M)
    decomp_y = padding_batch(decomp_y)


    final_output['ihm mask'] = ihm_M
    final_output['ihm label'] = ihm_y
    final_output['readmit mask'] = readmit_M
    final_output['readmit label'] = readmit_y
    final_output['ltm mask'] = ltm_M
    final_output['ltm label'] = ltm_y
    final_output['decomp mask'] = decomp_M
    final_output['decomp label'] = decomp_y
    final_output['los mask'] = los_M
    final_output['los label'] = los_y
    final_output['pheno label'] = pheno_y
    final_output['name'] = name
    
    if 'time series' in new_batch[0]:
        final_output['time series'] = X_padded
        final_output['T'] = T
    if 'texts' in new_batch[0]:
        final_output['texts'] = texts_idx
        final_output['texts weight mat'] =texts_weight_mat
        #final_output['raw text'] = raw_text
    if 'tab' in new_batch[0]:
        final_output['tab'] = tab_data
 
    return final_output
    



# if __name__ == '__main__':
#     # prepare discretizer and normalizer
#     conf = utils.get_config()
#     train_reader = MultitaskReader(dataset_dir=os.path.join(
#     conf.multitask_path, 'train'), listfile=os.path.join(conf.multitask_path, 'train','listfile.csv'))
#     test_reader = MultitaskReader(dataset_dir=os.path.join(
#         conf.multitask_path, 'test'), listfile=os.path.join(conf.multitask_path, 'test','listfile.csv'))
#     discretizer = Discretizer(timestep=conf.timestep,
#                             store_masks=True,
#                             impute_strategy='previous',
#                             start_time='zero')
#     discretizer_header = discretizer.transform(
#         train_reader.read_example(0)["X"])[1].split(',')
#     cont_channels = [i for (i, x) in enumerate(
#         discretizer_header) if x.find("->") == -1]
#     normalizer = Normalizer(fields=cont_channels)
#     normalizer_state = conf.normalizer_state
#     if normalizer_state is None:
#         normalizer_state = 'mult_ts{}.input_str:previous.start_time:zero.n5e4.normalizer'.format(
#             conf.timestep)
#     normalizer.load_params(normalizer_state)

#     vectors, w2i_lookup = utils.get_embedding_dict(conf)
#     train_ts_root_dir = '/home/yong/mutiltasking-for-mimic3/data/multitask_2/train'
#     train_text_root_dir = '/home/yong/mutiltasking-for-mimic3/data/root/text_fixed_train_val/'
#     test_ts_root_dir = '/home/yong/mutiltasking-for-mimic3/data/multitask_2/test'
#     test_text_root_dir = '/home/yong/mutiltasking-for-mimic3/data/root/text_fixed_test/'
#     ihm_pos = 47
#     los_pos = 24
#     use_text = True
#     use_ts = True
#     wf_dim = 100
#     decay = 0.1
#     max_text_length = 300
#     max_num_notes = 150
#     regression = True
#     bin_type = 'custom'
#     train_starttime_path = conf.starttime_path_train_val
#     test_starttime_path = conf.starttime_path_test
#     # ts_root_dir, text_root_dir, wf_root_dir, listfile, discretizer, starttime_path, regression, bin_type,\
#     #      normalizer, ihm_pos, los_pos, use_wf, use_text, use_ts,  wf_dim, decay, w2i_lookup, max_text_length, max_num_notes
#     train_mm_dataset = MultiModal_Dataset(train_ts_root_dir, train_text_root_dir, 'listfile.csv', discretizer, train_starttime_path,\
#          regression, bin_type, normalizer, ihm_pos, los_pos, use_text, use_ts, decay, w2i_lookup, max_text_length, max_num_notes)
#     test_mm_dataset = MultiModal_Dataset(test_ts_root_dir, test_text_root_dir, 'listfile.csv', discretizer, test_starttime_path,\
#          regression, bin_type, normalizer, ihm_pos, los_pos, use_text, use_ts, decay, w2i_lookup, max_text_length, max_num_notes)
#     print('mm dataset ready')
#     start_time = time.time()
#     collate_fn = functools.partial(custom_collate_fn, union = True)
#     train_data_loader = torch.utils.data.DataLoader(dataset=train_mm_dataset, 
#                                               batch_size=1,
#                                               shuffle=True,
#                                               num_workers=5,
#                                               collate_fn = collate_fn)
#     test_data_loader = torch.utils.data.DataLoader(dataset=test_mm_dataset, 
#                                               batch_size=1,
#                                               shuffle=True,
#                                               num_workers=5,
#                                               collate_fn = collate_fn)
    
#     read_M_aggregated, read_y_aggregated = [],[]

#     for i, data in enumerate(train_data_loader):
#         # if i% 100 ==99:
#         #     break
#         read_M = np.array(data['readmit mask'])
#         read_y = np.array(data['readmit label'])
        
#         # assert read_M.shape == read_y.shape
#         read_M_aggregated.append(read_M.reshape(1,-1))
#         read_y_aggregated.append(read_y.reshape(1,-1))
#         name = data['name']
#         if np.equal(read_y, 1):
#             print(name)
    
#     read_M_aggregated = np.concatenate(read_M_aggregated, axis = 1)
#     print(read_M_aggregated.sum())
#     read_y_aggregated = np.concatenate(read_y_aggregated, axis = 1)
#     read_y_aggregated = read_y_aggregated[read_M_aggregated==1]
#     print(read_y_aggregated.shape)
#     print(read_y_aggregated.sum())
#     labels, counts = np.unique(read_y_aggregated, return_counts=True)
#     print(labels)
#     print(counts)
    
    
    
        
    
    
            
        
    
    
        
   