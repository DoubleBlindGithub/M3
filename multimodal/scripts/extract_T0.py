import os
import pandas as pd
import numpy as np
import pickle

homedir = os.path.expanduser("~")
base_path = os.path.join(homedir, 'mutiltasking-for-mimic3')

rootpath_train = os.path.join(base_path, 'data/root_3/train')
rootpath_test = os.path.join(base_path, 'data/root_3/test')
train_starttime_path = os.path.join(base_path, 'data/root_3/train_starttime.pkl')
test_starttime_path = os.path.join(base_path, 'data/root_3/test_starttime.pkl')


def diff(time1, time2):
    # compute time2-time1
    # return difference in hours
    a = np.datetime64(time1)
    b = np.datetime64(time2)
    return (a-b).astype('timedelta64[h]').astype(int)

def time_mapping(rootpath, starttime_path, episodeToStartTimeMapping):
    for findex, folder in enumerate(os.listdir(rootpath)):
        events_path = os.path.join(rootpath, folder, 'events.csv')
        if not os.path.exists(events_path):
            continue
        events = pd.read_csv(events_path)

        stays_path = os.path.join(rootpath, folder, 'stays.csv')
        if not os.path.exists(stays_path):
            continue
        stays_df = pd.read_csv(stays_path)
        hadm_ids = list(stays_df.HADM_ID.values)
        intimes = stays_df.INTIME.values

        for ind, hid in enumerate(hadm_ids):
            sliced = events[events.HADM_ID == hid]
            chart_times = sliced['CHARTTIME']
            chart_times = chart_times.sort_values()
            intime = intimes[ind]
            # remove intime from charttime
            result = -1
            # pick the first charttime which is positive or > -eps (1e-6)
            for t in chart_times:
                # compute t-intime in hours
                if diff(t, intime) > 1e-6:
                    result = t
                    break
            name = folder + '_' + str(ind+1)
            episodeToStartTimeMapping[name] = result

        if findex % 100 == 0:
            print("Processed %d" % (findex + 1))

    with open(starttime_path, 'wb') as f:
        pickle.dump(episodeToStartTimeMapping, f, pickle.HIGHEST_PROTOCOL)

time_mapping(rootpath_train, train_starttime_path,{})
time_mapping(rootpath_test, test_starttime_path, {})

