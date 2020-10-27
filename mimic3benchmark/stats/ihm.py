from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import pandas as pd
import random
import csv
random.seed(49297)


def get_stats(args, partition, n_hours = 24):
    output_dir = os.path.join(args.output_path, str(n_hours), partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    icu_stay_positive, icu_stay_negative = [],[]
    patients_all = list(filter(str.isdigit, os.listdir(output_dir)))
    with open(os.path.join(output_dir,'listfile.csv')) as csvfile:
        f = csv.reader(csvfile)
        for row in f:
            icu_stay = row[0].split('.')[0]
            label = row[1]
            if label == '1':
                icu_stay_positive.append(icu_stay)
            elif label == '0':
                icu_stay_negative.append(icu_stay)
    print('Number of positive patients in {} is {}'.format(partition, len(icu_stay_positive))) 
    print('Number of positive icu stays in {} is {}'.format(partition,len(icu_stay_positive)))
    print('Number of negative patients in {} is {}'.format(partition, len(patients_all)-len(icu_stay_positive))) 
    print('Number of negative icu stays in {} is {}'.format(partition,len(icu_stay_negative)))


def get_detailed_stats(args, partition, n_hours = 24):
    output_dir = os.path.join(args.output_path, str(n_hours), partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    all_stays_df = pd.read_csv(os.path.join(args.root_path, 'all_stays.csv'))
    male_p, male_n, female_p, female_n, gender_all = 0,0,0,0,0
    ethinicity = all_stays_df['ETHNICITY'].value_counts
    gender = all_stays_df['GENDER'].value_counts()
    ax = ethinicity.plot.bar(x='ethinic groups', y='stays', rot=0)
    print(gender)

    patients_all = list(filter(str.isdigit, os.listdir(output_dir)))
    with open(os.path.join(output_dir,'listfile.csv')) as csvfile:
        f = csv.reader(csvfile)
        for row in f:
            icu_stay = row[0].split('.')[0]





    


def main():
    parser = argparse.ArgumentParser(description="Create data for in-hospital mortality prediction task.")
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    get_stats(args, "test")
    get_stats(args, "train")
    get_detailed_stats(args, "test")


if __name__ == '__main__':
    main()
