MIMIC-III Multimodal and Multitask
=========================


## Structure
```bash
.
├── data
├── mimic3benchmark
│   ├── evaluation
│   ├── resources
│   └── scripts
├── mimic3models
│   ├── decompensation
│   │   └── logistic
│   ├── in_hospital_mortality
│   │   └── logistic    
│   ├── keras_models
│   ├── length_of_stay
│   │   └── logistic
│   ├── multitask
│   ├── phenotyping
│   │   └── logistic
│   └── resources
└── multimodal
    ├── models
    │   └── ckpt
    └── scripts
```

1. data: this is the folder stores all data
2. mimic3benchmark: the folder contains the data preprocess pipeline and some other utilitis to read the data
3. mimic3benchmark: the folder contains the Harytyunyan benchmark models
4. multimodal: the folder contains scripts to preprocess the text data and multimodal-multitasking models


## Requirements


- numpy
- pandas

Note: Please use pandas==0.20.3 to avoid errors in preprocessing datetimes.


## Data
All mimic-3 raw data (text and time-series) is in [here]()

## Preprocessing the time-series data:
1. Download all data

2. The following command takes MIMIC-III CSVs, generates one directory per `SUBJECT_ID` and writes ICU stay information to `data/{SUBJECT_ID}/stays.csv`, diagnoses to `data/{SUBJECT_ID}/diagnoses.csv`, and events to `data/{SUBJECT_ID}/events.csv`. This step might take around an hour.

       python -m mimic3benchmark.scripts.extract_subjects {PATH TO MIMIC-III CSVs} data/root/

3. The following command attempts to fix some issues (ICU stay ID is missing) and removes the events that have missing information. About 80% of events remain after removing all suspicious rows (more information can be found in [`mimic3benchmark/scripts/more_on_validating_events.md`](mimic3benchmark/scripts/more_on_validating_events.md)).

       python -m mimic3benchmark.scripts.validate_events data/root/

4. The next command breaks up per-subject data into separate episodes (pertaining to ICU stays). Time series of events are stored in ```{SUBJECT_ID}/episode{#}_timeseries.csv``` (where # counts distinct episodes) while episode-level information (patient age, gender, ethnicity, height, weight) and outcomes (mortality, length of stay, diagnoses) are stores in ```{SUBJECT_ID}/episode{#}.csv```. This script requires two files, one that maps event ITEMIDs to clinical variables and another that defines valid ranges for clinical variables (for detecting outliers, etc.). **Outlier detection is disabled in the current version**.

       python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/

5. The next command splits the whole dataset into training and testing sets. Note that the train/test split is the same of all tasks.

       python -m mimic3benchmark.scripts.split_train_and_test data/root/
	
6. The following commands will generate task-specific datasets, which can later be used in models. These commands are independent, if you are going to work only on one benchmark task, you can run only the corresponding command.

       python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/
       python -m mimic3benchmark.scripts.create_decompensation data/root/ data/decompensation/
       python -m mimic3benchmark.scripts.create_length_of_stay data/root/ data/length-of-stay/
       python -m mimic3benchmark.scripts.create_phenotyping data/root/ data/phenotyping/
       python -m mimic3benchmark.scripts.create_multitask data/root/ data/multitask/

After the above commands are done, there will be a directory `data/{task}` for each created benchmark task.
These directories have two sub-directories: `train` and `test`.
Each of them contains bunch of ICU stays and one file with name `listfile.csv`, which lists all samples in that particular set.
Each row of `listfile.csv` has the following form: `icu_stay, period_length, label(s)`.
A row specifies a sample for which the input is the collection of ICU event of `icu_stay` that occurred in the first `period_length` hours of the stay and the target is/are `label(s)`.
In in-hospital mortality prediction task `period_length` is always 48 hours, so it is not listed in corresponding listfiles.

## Preprocess the text data

1. cd into multimodal folder
2. run extract_notes.py file under scripts folder. This will extract text data in each icu stay and save it in json files, {patient_id}_{# of their stay}.the key is the time point when notes are written and the value are the text, if a stay has no notes  taken, this step would be skipped.
```shell
python3 scripts/extract_notes.py 
```

3. run extract_T0.py file under scripts folder. This would extract the time point when time-series data start recording, this would be useful in later steps.
```shell
python3 scripts/extract_T0.py 
```

## Models
Models are defined under multimodal/models/multi_modality_model_hy.py. This files defines 
1. base class for `ModalityEncoder` - which defines the struture of an encoder for a modality
2. class `MultiModalEncoder` - which takes 3 encoders that inherit from `ModalityEncoder` and generates the global embedding
3. base class `TaskSpecificComponent` - which defines the structure of the task specific compoenent for a given task.
4. base class `MultiModalMultiTaskWrapper` - which combines the above classes into the full MM-MT model. This requires
   1. A `MultiModalEncoder` that encodes all the modalities. This encoder requires:
      1. A Time Series `ModalityEncodeer`
      2. A Text `ModalityEncoder`
      3. A Tabular `ModalityEncoder`
   2. A `TaskSpecificComponent` per task of interest. In this work we had six of tasks, leading to six `TaskSpecificComponenet`s.

Included in multi_modality_model_hy.py are `ModailtyEncoders` for time series(`LSTMModel`), text(`Text_CNN`), and tabular(`TabularEmbedding`) data.
Also included is our default `TaskSpecificCompoenent`, `FCTaskComponenet` which is just a FC linear layer, followed by Dropout, ReLU, and an output layer. The specific parameters per task are defined in the training script multimodal/experiments/multitasking/multitasking.py.





       

