import re
import os
import random
import shutil
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import pandas as pd
import glob
from sklearn.metrics import confusion_matrix
import sys
import IPython.display as ipd  # To play sound in the notebook

def get_label(filename):
    m = re.search('(.+?)_', filename)
    if m:
        return m.group(1)
    return None

def delete_dirs():
    shutil.rmtree("data/dev")
    shutil.rmtree("data/train")
    shutil.rmtree("data/test")

def create_dirs():
    os.mkdir("data/dev")
    os.mkdir("data/train")
    os.mkdir("data/test")

def get_speaker_file_dictionary():
    prepared_files = {}
    for root, dirs, files in os.walk("data/raw"):
        for filename in files:
            m = re.search('(.+?)_', filename)
            if m:
                found = m.group(1)
                if found not in prepared_files:
                    prepared_files[found] = []
                prepared_files[found].append(filename)
    return prepared_files

def make_data_for(list, mode):
    for file in list:
        shutil.copyfile(f"data/raw/{file}",f"data/{mode}/{file}")

def get_mode_counts(size):
    n_test = size // 10 * 1
    n_train = size // 10 * 8
    n_dev = size - (n_test + n_train)
    return n_dev, n_train, n_test

def process_file_list(files, n_dev, n_train, n_test):
    make_data_for(files[0:n_dev], "dev")
    make_data_for(files[n_dev: n_dev+n_train], "train")
    make_data_for(files[n_dev+n_train : ], "test")

def prepare():
    delete_dirs()
    create_dirs()
    prepared_files = get_speaker_file_dictionary()

    for key,value in prepared_files.items():
        n_dev, n_train, n_test = get_mode_counts(len(value))
        random.shuffle(value)
        process_file_list(value, n_dev, n_train, n_test )


def stat():
    print(f"Total dev files: {len(os.listdir('data/dev'))}")
    print(f"Total train files: {len(os.listdir('data/train'))}")
    print(f"Total test files: {len(os.listdir('data/test'))}")


def prepare_if_needed():
    if not os.path.exists("data/dev"):
        return prepare()
    stat()

def get_data(mode):
    df = pd.DataFrame(columns=['feature'])

    # loop feature extraction over the entire dataset
    counter = 0
    labels = []
    for index, path in enumerate(os.listdir(f"data/{mode}")):
        X, sample_rate = librosa.load(f"data/{mode}/{path}"
                                      , res_type='kaiser_fast'
                                      , duration=2.5
                                      , sr=44100
                                      , offset=0.5
                                      )
        sample_rate = np.array(sample_rate)

        # mean as the feature. Could do min and max etc as well.
        mfccs = np.mean(librosa.feature.mfcc(y=X,
                                             sr=sample_rate,
                                             n_mfcc=13),
                        axis=0)
        df.loc[counter] = [mfccs]
        labels.append(get_label(path))
        counter = counter + 1

    # Check a few records to make sure its processed successfully
    # print(len(df))
    a_data_frame = pd.DataFrame()
    a_data_frame = pd.concat([a_data_frame, pd.DataFrame(df['feature'].values.tolist())], axis=1)
    a_data_frame = a_data_frame.fillna(0)
    return a_data_frame.values, labels

