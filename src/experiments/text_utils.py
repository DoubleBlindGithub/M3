import os
import pickle
import numpy as np
import json
from utils import lookup
import pandas as pd
from models.multi_modality_model_hy import FeatureSpreadWTime
import torch
import time

def diff(time1, time2):
    # compute time2-time1
    # return difference in hours
    a = np.datetime64(time1)
    b = np.datetime64(time2)
    h = (b-a).astype('timedelta64[h]').astype(int)
    '''if h < -1e-6:
        print(h)
        assert h > 1e-6'''
    return h


class TextReader():
    def __init__(self, dbpath, starttime_path, max_number_events=-1):
        self.dbpath = dbpath
        self.all_files = set(os.listdir(dbpath))
        self.maximum_number_events = max_number_events
        with open(starttime_path, 'rb') as f:
            self.episodeToStartTime = pickle.load(f)

    def get_name_from_filename(self, fname):
        # '24610_episode1_timeseries.csv'
        tokens = fname.split('_')
        pid = tokens[0]
        episode_id = tokens[1].replace('episode', '').strip()
        return pid, episode_id


    def read_text_event_json(self, text_file_name):
        filepath = os.path.join(self.dbpath, str(text_file_name))
        with open(filepath, 'r') as f:
            d = json.load(f)
        time = sorted(d.keys())
        text = []
        for t in time:
            text.append(" ".join(d[t]))
        assert len(time) == len(text)
        return time, text

    def read_all_text_events_json(self, names):
        """
        retrieve raw text by file names, e.g. '24610_episode1_timeseries.csv'
        return type: a dictionary, key is the hours, value is a list containing 
        all text tokens of that note in that hour
        """
        text_event_dictionary = {}  # name -> [(hour, text)]
        for patient_id in names:
            pid, eid = self.get_name_from_filename(patient_id)
            text_file_name = pid + "_" + eid
            data_for_patient = []
            find = 0
            
            if text_file_name in self.all_files:
                find+=1
                
                time, texts = self.read_text_event_json(text_file_name)
                start_time = self.episodeToStartTime[text_file_name]
                if len(texts) == 0 or start_time == -1:
                    continue
                last_h = 5
                last_text = ""
                cnt = 0
                for (t, txt) in zip(time, texts):
                    h = diff(start_time, t)
                    cnt += 1
                    if self.maximum_number_events != -1 and cnt >= self.maximum_number_events:
                        break
                    if h < 5:
                        h = 5
                    if h == last_h:
                        last_text += txt
                    else:
                        data_for_patient.append((last_h, last_text))
                        last_h = h
                        last_text = txt
                data_for_patient.append((last_h, last_text))
                text_event_dictionary[patient_id] = data_for_patient
        return text_event_dictionary


class TextReader_Pretrained(TextReader):
    def __init__(self):
        super(TextReader_Pretrained, self).__init__()

    def read_text(self, name):
        raise NotImplementedError


class WaveformReader_Pretrained():
    def __init__(self, rootpath):
        self.rootpath = rootpath
        self.file_list = os.listdir(rootpath)
        #print('total_file', len(self.file_list))

    def get_name_from_filename(self, fname):
        # '24610_episode1_timeseries.csv'
        tokens = fname.split('_')
        pid = tokens[0]
        episode_id = tokens[1].replace('episode', '').strip()
        return pid, episode_id

    def read_waveform(self, name):
        candidates = []
        pid, eid = self.get_name_from_filename(name)

        for fname in self.file_list:
            if fname.startswith('{}_episode{}_'.format(pid, eid)):
            #if '{}_episode{}'.format(pid, eid) in fname:
                candidates.append(fname)

        # if not any('{}_episode{}'.format(pid, eid) in fname for fname in candidates):
        #     return None
        if len(candidates) == 0:
            return None


        hours, waveforms = [], []
        for fname in candidates:
            # 98994_episode3_108.0_waveform_embedding_100.pt
            #print(fname.split('_'))
            hour = fname.split('_')[2]
            if hour == 'waveform':
                continue
            wf_fname = '{}_episode{}_{}_waveform_embedding_100.pt'.format(pid, eid, hour)
            wf_emb = torch.load(os.path.join(self.rootpath, wf_fname)).detach().numpy()
            hours.append(float(hour))
            waveforms.append(wf_emb)
        
        return hours, waveforms



class WaveformReader():
    def __init__(self, dataset):
        self.dataset = dataset
        # add a new column to get the row_id, waveform is retrieved base on row id
        self.dataset.csv['row_id'] = np.arange(len(self.dataset.csv))

    def get_idx(self, name):
        """
        given a filename, retrive the row ids for that file, e.g. '24610_episode1_timeseries.csv'
        return type is a dictionary where key is the hour, value is the row id of waveforms in that hours
        """
        
        idx = {}
        if name not in self.dataset.csv.values:
            return None
        sliced = self.dataset.csv[self.dataset.csv['stay']==name]
        hours = sliced['period_length'].values.tolist()
        for h in hours:
            idx[h] = sliced[sliced['period_length']==h]['row_id'].values[0]
        return idx

    def read_waveform(self, name):
        idx = self.get_idx(name)
        # if there is no waveform for that ts filename, we return a None
        if not idx:
            return None
        hours, waveforms = [], []
        for h in idx.keys():
            hours.append(h)
            signal, _ = self.dataset[idx[h]]
            waveforms.append(signal)
        return hours, waveforms


class Tablular_reader():
    def __init__(self,rootpath):
        self.rootpath = rootpath
        self.genderset = set(['M', 'F'])
        self.ethnicityset = set(['ASIAN','WHITE','BLACK/AFRICAN AMERICAN','HISPANIC OR LATINO'])
        
    def readall(self, rootpath):
        subdirs = [ f.path for f in os.scandir(rootpath) if f.is_dir()]
        for subdir in subdirs:
            stay_df = pd.read_csv(os.path.join(subdir, 'stays.csv'))
            gender = stay_df['GENDER'].iloc[0].lower()
            age = stay_df['AGE'].iloc[0]
            ethnicity = stay_df['ETHNICITY'].iloc[0].lower()
            self.ageset.add(age)
            self.genderset.add(gender)
            self.ethnicityset.add(ethnicity) 
        
    def get_name_from_filename(self, fname):
        # '24610_episode1_timeseries.csv'
        fname = fname.split('.')[0]
        tokens = fname.split('_')
        pid = tokens[0]
        episode_id = tokens[1].replace('episode', '').strip()
        return pid, episode_id

    def read_tabular(self, name):
        res = {}
        pid, eid = self.get_name_from_filename(name)
        gender = pd.read_csv(os.path.join(self.rootpath, pid, 'stays.csv'))['GENDER'].iloc[0]
        age = pd.read_csv(os.path.join(self.rootpath, pid, 'stays.csv'))['AGE'].iloc[0]
        ethnicity = pd.read_csv(os.path.join(self.rootpath, pid, 'stays.csv'))['ETHNICITY'].iloc[0]
        res['GENDER']= gender
        res['AGE'] = age
        res['ETHNICITY'] = ethnicity
        return res


def generate_tensor_text(patient_text_list, w2i_lookup, conf_max_len):
    patient_list_of_indices = []
    max_words = 0
    max_notes = 0

    for patient_notes in patient_text_list:
        # each patient_text is a list of text
        list_of_word_idx = []
        for note in patient_notes:
            # each note is a list of word
            indices = list(map(lambda x: lookup(
                w2i_lookup, x), str(note).split()))
            if conf_max_len > 0:
                indices = indices[:conf_max_len]
            list_of_word_idx.append(indices)
            max_words = max(len(indices), max_words)
        patient_list_of_indices.append(list_of_word_idx)
        max_notes = max(len(list_of_word_idx), max_notes)

    pad_token = w2i_lookup['<pad>']


    if max_notes ==0 or max_words<=4:
        
        # in case all icu stay in a batch don't have text or all notes are too short to support bigram or trigam conv
        max_notes =1
        max_words =20


    # 3. 3d pad, padding token.
    # 4. convert to numpy tensor and return
    def extra_pad_tokens(cnt): return [pad_token]*cnt

    padded_patient_list_of_indices = []
    for pt in patient_list_of_indices:
        padded_pt = []
        if len(pt) < max_notes:
            pt = pt + [[]]*(max_notes-len(pt))
        for l in pt:
            l = l + extra_pad_tokens(max_words - len(l))
            padded_pt.append(l)
        padded_patient_list_of_indices.append(padded_pt)

    x = np.array(padded_patient_list_of_indices)
    
    try:
        assert len(x.shape) == 3
        assert x.shape[0] == len(patient_text_list), "x: {}, l: {}".format(
            str(x.shape), str(len(patient_text_list)))
        return x
    except:
        print('bad shape of x', x.shape)


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



def get_max_num_notes(batch, text_event_dictionary, X):
    max_num_notes = -1
    for i, name in enumerate(batch['names']):
        if name not in text_event_dictionary:
            continue
        text_events = text_event_dictionary[name]
        hours = map(lambda x: x[0], text_events)
        hours = list(filter(lambda h: h <= X.shape[1], hours))
        max_num_notes = max(max_num_notes, len(hours))
    return max_num_notes


def get_text_weight_matrix(total_time, max_num_notes, hours):
    # hours: when the notes are taken
    if max_num_notes == -1:
        max_num_notes = 1
    weight_matrix = np.zeros((total_time, max_num_notes))
    # when there is no notes
    if len(hours) == 0:
        return weight_matrix
    for t in range(total_time):
        for i in range(len(hours)):
            h = hours[i]
            if h>t:
                continue
            weight_matrix[t][i] = t-h+1
            assert weight_matrix[t][i] >= 0
    ones = np.ones_like(weight_matrix)
    binary_mask = np.where(weight_matrix==0, weight_matrix, ones)
    weight_decay = np.exp(-0.1*weight_matrix)
    weight_decay_clipped = np.clip(weight_decay, a_min= 1e-10 , a_max=1)
    final_weight = weight_decay_clipped * binary_mask
    return final_weight

def get_waveform_weight_matrix(total_time,max_num_waveform ,hours):
    """
    generate a weight matrix that will be used in weigthed combination of waveform features
    """
    # hours: when the notes are taken
    if max_num_waveform == -1:
        max_num_waveform = 1
    weight_matrix = np.zeros((total_time, max_num_waveform))
    # when there is no notes
    if len(hours) == 0:
        return weight_matrix
    for t in range(total_time):
        for i in range(len(hours)):
            h = hours[i]
            if h>t:
                continue
            weight_matrix[t][i] = t-h+1
            assert weight_matrix[t][i] >= 0
    ones = np.ones_like(weight_matrix)
    binary_mask = np.where(weight_matrix==0, weight_matrix, ones)
    weight_decay = np.exp(-0.1*weight_matrix)
    weight_decay_clipped = np.clip(weight_decay, a_min= 1e-10 , a_max=1)
    final_weight = weight_decay_clipped * binary_mask
    return final_weight


def convert_to_numpyarray(final_data_batch):
    """
    convert all data in a batch into numpy array
    """
    final_data_batch['ts'] = np.stack(final_data_batch['ts'])
    final_data_batch['decom_mask'] = np.stack(final_data_batch['decom_mask'])
    final_data_batch['los_mask'] = np.stack(final_data_batch['los_mask'])
    final_data_batch['decom_label'] = np.stack(final_data_batch['decom_label'])
    final_data_batch['los_label'] = np.stack(final_data_batch['los_label'])
    final_data_batch['ihm_mask'] = np.stack(final_data_batch['ihm_mask'])
    final_data_batch['ihm_label'] = np.stack(final_data_batch['ihm_label'])
    final_data_batch['text_weight_matrix'] = np.stack(final_data_batch['text_weight_matrix'])
    final_data_batch['waveform_weight_matrix'] = np.stack(final_data_batch['waveform_weight_matrix'])
    final_data_batch['pheno_label'] = np.stack(final_data_batch['pheno_label'])

    return final_data_batch


def pad_waveform(waveforms, max_num_waveform):
    
    if max_num_waveform == -1:
        return None
    paddings = np.zeros((1, 100))
    #print(max_num_waveform)
    for i in range(len(waveforms)):
        wf = waveforms[i]
        
        wf = wf +[paddings]* (max_num_waveform-len(wf))
        waveforms[i] = wf
        
    res = []
    for wf in waveforms:
        
        wf = np.vstack( wf)
        res.append(wf)
    res = np.stack(res, axis= 0)
    return res    
    

def onehot_enc(data, tab_reader):
    # data is a dict
    #ageset = tab_reader.ageset
    genderset = list(tab_reader.genderset)
    ethnicityset = list(tab_reader.ethnicityset)
    age_vec = np.zeros( 5)
    gender_vec = np.zeros(len(genderset)+1)
    ethnicity_vec = np.zeros(len(ethnicityset)+1)

    if int(data['AGE'])>=18 and int(data['AGE'])<30:
        age_vec[0] =1.0
    elif int(data['AGE'])>=30 and int(data['AGE'])<50:
        age_vec[1] =1.0
    elif int(data['AGE'])>=50 and int(data['AGE'])<70:
        age_vec[2] =1.0
    elif int(data['AGE'])>=70:
        age_vec[3] =1.0
    else:
        age_vec[4] =1.0
    
    if data['GENDER'] not in genderset:
        gender_vec[-1] =1.0
    for i in range(len(genderset)):
        if data['GENDER'] == genderset[i]:
            gender_vec[i] =1.0
    if data['ETHNICITY'] not in ethnicityset:
        ethnicity_vec[-1] =1.0
    for i in range(len(ethnicityset)):
        if data['ETHNICITY'] == ethnicityset[i]:
            ethnicity_vec[i] =1.0
    res = np.concatenate((age_vec, gender_vec, ethnicity_vec))
    return res
    


def avg_emb(text, weight_mat):
    # text: b * n * l * e
    text = torch.mean(text, dim=2)
    res = []
    for i in range(text.shape[0]):
        text_i = FeatureSpreadWTime(text[i,:,:], weight_mat[i,:,:])
        res.append(text_i)
    return torch.stack(res, dim = 0) # B * T * E


def retrieve_waveforms(batch, waveformreader):
    max_num_waveform = -1
    hs, wfs = [], []
    for i, name in enumerate(batch['names']):
        # if name not in waveformreader.dataset.csv.values:
        #     continue
        if waveformreader.read_waveform(name) is None:
            hs.append([])
            wfs.append([])
        else:
            hours, waveforms = waveformreader.read_waveform(name)
            hs.append(hours)
            wfs.append(waveforms)
            max_num_waveform = max(max_num_waveform, len(hours))
    return max_num_waveform, hs, wfs

def merge_text_ts_waveforms(batch, text_reader, waveform_reader, w2i_lookup, conf_max_len):
    #start_time = time.time()
    final_data_batch = {'ts':[], 'texts':[], 'decom_mask':[], 'ihm_mask':[], 'los_mask':[], 'text_weight_matrix':[], \
        'waveform_weight_matrix':[],'decom_label':[], 'los_label':[], 'ihm_label':[], 'waveforms':[], 'pheno_label':[]}

    ip, op, _ = batch['data']
    ts = ip[0] # [batch_size, max_len, 76]
    total_time = ts.shape[1]

    # decom mask from hour 0 to max recorded hour in that batch, hour less than 5 or the begining time will be 0.
    # the padding hour are also zero

    ihm_mask, decom_mask, los_mask = ip[1], ip[2], ip[3]
    ihm_label, decom_label, los_label, pheno_label = op[0], op[1], op[2], op[3]
    assert_shapes(ts, decom_mask, decom_label)
    assert_shapes(ts, los_mask, los_label)

    text_event_dictionary = text_reader.read_all_text_events_json(
        batch['names'])
    
    max_num_notes = get_max_num_notes(batch, text_event_dictionary, ts)
    #print('read ts', time.time()-start_time)
    max_num_waveform, hs, wfs = retrieve_waveforms(batch, waveform_reader)
    #print('read_max_waveform', time.time()-start_time)


    for i, name in enumerate(batch['names']):
        # timerow represents 1 patient.
        # first timestep is 5.
        # the number of final data in one batch may be less than batch size if we want the intersection of the ts and text
        # if not union:
        #     if name not in text_event_dictionary:
        #         continue

        final_data_batch['ts'].append(ts[i])
        final_data_batch['decom_mask'].append(decom_mask[i])
        final_data_batch['los_mask'].append(los_mask[i])
        final_data_batch['decom_label'].append(decom_label[i])
        final_data_batch['los_label'].append(los_label[i])
        final_data_batch['ihm_mask'].append(ihm_mask[i])
        final_data_batch['ihm_label'].append(ihm_label[i])
        final_data_batch['pheno_label'].append(pheno_label[i])

        # read waveform data
        # if waveform_reader.read_waveform(name) is not None:
        #     waveform_hours, waveforms = waveform_reader.read_waveform(name)
        # else:
        #     waveform_hours, waveforms = [], []

        waveform_hours, waveforms = hs[i], wfs[i]
        # read text data
        if name in text_event_dictionary:
            text_events = text_event_dictionary[name]
            assert len(text_events[0]) == 2
            text_hours = list(map(lambda x: x[0], text_events))[:max_num_notes]
            texts = list(map(lambda x: x[1], text_events))[:max_num_notes]
            assert len(text_hours) == len(texts)
        else:
            text_hours = []
            texts = []


        text_weight_matrix_i = get_text_weight_matrix(total_time, max_num_notes, text_hours) #[total_time, max_num_notes]
        waveform_weight_matrix_i = get_waveform_weight_matrix(total_time, max_num_waveform, waveform_hours)
        final_data_batch['text_weight_matrix'].append(text_weight_matrix_i)
        final_data_batch['waveform_weight_matrix'].append(waveform_weight_matrix_i)
        final_data_batch['texts'].append(texts)
        final_data_batch['waveforms'].append(waveforms)
        
   
    if not any(final_data_batch['texts']) or not any(final_data_batch['waveforms']):
        return None
    #print('read all', time.time()-start_time)
    # if not any(final_data_batch['waveforms']):
    #     return None

    final_data_batch = convert_to_numpyarray(final_data_batch)
    texts = generate_tensor_text(final_data_batch['texts'], w2i_lookup, conf_max_len)
    waveforms = pad_waveform(final_data_batch['waveforms'], max_num_waveform)
    
    final_data_batch['texts'] = texts
    final_data_batch['waveforms'] = waveforms
    #print('maerge all', time.time()-start_time)
    
    return final_data_batch




        



        
            


    





    

    
