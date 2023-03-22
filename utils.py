import pickle

import pandas as pd
import numpy as np
import json


def load_dict_from_json(json_file):
    with open(json_file,"r") as f:
        data = json.load(f)
    return data


def load_partial_data(csv_path, n=-1): # if n=-1 it will load all data
    df = pd.read_csv(csv_path, delimiter="@")
    df = df.sample(frac=1)  # shuffling rows
    texts = []
    labels = []
    c = 0
    if n == -1:
        n = len(df)
    for idx in df.index:
        text = df["tweet"][idx]
        label = df["class"][idx]
        texts.append(text)
        labels.append(label)
        c += 1
        if c >= n:
            break
    return texts, labels


def split_for_train(x,y,val_split=0.3):
    total = len(x)
    nb_train = 1 - int(total*val_split)
    train_x = x[:nb_train]
    train_y = y[:nb_train]
    val_x = x[nb_train:]
    val_y = y[nb_train:]
    return (train_x,train_y),(val_x,val_y)


def load_batch_data(x_data, y_data, start, end,one_hot_y=True,label_map=None):
    b_x = x_data[start:end]
    if one_hot_y:
        b_y = encode_one_hot(y_data[start:end],label2ind=label_map)
    else:
        b_y = y_data[start:end]
    return b_x, b_y


def encode_one_hot(y,label2ind):
    n = len(y)
    nc = len(label2ind.keys())
    y_onehot = np.zeros([n,nc])
    for d in range(n):
        lbl = label2ind[y[d]]
        y_onehot[d][lbl] = 1
    return y_onehot


def load_pkl_data(pkl_file):
    with open(pkl_file,"rb") as f:
        data = pickle.load(f)
    x,y = data[0],data[1]
    return x,y