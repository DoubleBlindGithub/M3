from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import pandas as pd
import random
random.seed(49297)
import numpy as np
import matplotlib.pyplot as plt



def process_partition(args, partition, eps=1e-6):
    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    xy_pairs = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    label_df = pd.read_csv(args.patients_path)
    all_stays_df = pd.read_csv(os.path.join(args.root_path, 'all_stays.csv'))
    
    all_survial_time, all_stays = [],0
    for (patient_index, patient) in enumerate(patients):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))
        
        for ts_filename in patient_ts_files:
            lb_filename = ts_filename.replace("_timeseries", "")
            lb_df = pd.read_csv(os.path.join(patient_folder, lb_filename))
            if lb_df.empty:
                continue
            all_stays+=1
            stays_df = pd.read_csv(os.path.join(patient_folder, 'stays.csv'))
            icu_stay_id = lb_df.Icustay.values[0]
            
            
            # hadm_id = stays_df.loc[stays_df.ICUSTAY_ID == icu_stay_id, 'HADM_ID'].values[0]
            
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                
                lb_filename = ts_filename.replace("_timeseries", "")
                
                
                dod_ssn = label_df.loc[label_df.SUBJECT_ID == int(patient), 'DOD_SSN'].values[0]
                if pd.isnull(dod_ssn):
                    
                    continue
                dod_ssn = pd.to_datetime(dod_ssn)
                dod_hosp = label_df['DOD_HOSP'].where(label_df['SUBJECT_ID'] == int(patient)).values[0]

                if not pd.isnull(dod_hosp):
                    
                    continue
                
                
                doa = all_stays_df['INTIME'].where(all_stays_df['ICUSTAY_ID'] == int(icu_stay_id)).dropna().values[0]
                if pd.isnull(doa):
                    continue
                doa = pd.to_datetime(doa) 
                survial_time = (dod_ssn-doa).days
                all_survial_time.append(survial_time)
                
                
                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                ts_lines = [line for (line, t) in zip(ts_lines, event_times)]

                # # no measurements in ICU
                if len(ts_lines) == 0:
                    print("\n\t(no events in ICU) ", patient, ts_filename)
                    continue

                output_ts_filename = patient + "_" + ts_filename
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)

                xy_pairs.append((output_ts_filename, survial_time))

        if (patient_index + 1) % 100 == 0:
            print("processed {} / {} patients".format(patient_index + 1, len(patients)), end='\r')

    print("\n", len(xy_pairs))
    if partition == "train":
        random.shuffle(xy_pairs)
    if partition == "test":
        xy_pairs = sorted(xy_pairs)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,y_true\n')
        for (x, y) in xy_pairs:
            listfile.write('{},{:d}\n'.format(x, y))
    print('the number of death is {}'.format(len(all_survial_time)))
    print('the number of stays {}'.format(all_stays))
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = [0, 30, 60, 90,120,150,180,210,240,270,300,330, 360, 720]
    _, bins, patches = plt.hist(np.clip(all_survial_time, bins[0], bins[-1]), bins=bins,label='days of survival in {} set'.format(partition))
    xlabels = [str(i) for i in bins]
    xlabels[-1] = '720+'
    N_labels = len(xlabels)
    
    plt.xticks(25 * np.arange(N_labels))
    ax.set_xticklabels(xlabels)
    plt.xlim([0, 360])
    plt.legend()
    plt.show()
    

def main():
    parser = argparse.ArgumentParser(description="Create data for length-of-survival prediction task.")
    
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    parser.add_argument('patients_path', default='data/PATIENTS.csv')
    parser.add_argument('admissions_path', default='data/ADMISSIONS.csv')
    

    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    process_partition(args, "test")
    process_partition(args, "train")


if __name__ == '__main__':
    main()
