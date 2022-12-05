import numpy as np
import json
import os
import pandas as pd
import torch
import torch.nn as nn
from transformers import ElectraModel, ElectraTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils import clip_grad_norm_
import random
import time
from sklearn.utils.class_weight import compute_class_weight
from model import Cartesian, MainCategoryClassifier ,Cartesian2, AddOne_onehot, AddOne_sparse
from sklearn.metrics import f1_score
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import random

def prepare_ids(y_data_train, y_data_val, main_only=False, sub_only=False):
    entity_property_pair = set()
    polarity_id_to_name = set()
    for sample in y_data_train:
        for annotation in sample:
            if sub_only: property_pair = annotation[0].split('#')[-1]
            elif main_only: property_pair = annotation[0].split('#')[0]
            else: property_pair = annotation[0]
            polarity = annotation[2]
            entity_property_pair.add(property_pair)
            polarity_id_to_name.add(polarity)
    for sample in y_data_val:
        for annotation in sample:
            if sub_only: property_pair = annotation[0].split('#')[-1]
            elif main_only: property_pair = annotation[0].split('#')[0]
            else: property_pair = annotation[0]
            polarity = annotation[2]
            entity_property_pair.add(property_pair)
            polarity_id_to_name.add(polarity)
            
    label_id_to_name = list(entity_property_pair)
    label_name_to_id = {label_id_to_name[i]: i for i in range(len(label_id_to_name))}

    polarity_id_to_name = list(polarity_id_to_name) 
    polarity_name_to_id = {polarity_id_to_name[i]: i+1 for i in range(len(polarity_id_to_name))} 

    return label_id_to_name, label_name_to_id, polarity_id_to_name, polarity_name_to_id

def prep_data(y_data_train, num_labels, label_name_to_id, polarity_name_to_id, sub_only=False):
    y_train_sparse = []
    y_train_onehot = []
    y_train_cartesian = []
    for sample in y_data_train:
        y_sample_sparse = np.zeros((num_labels), dtype='int32')
        y_sample_onehot = np.array([[1,0,0,0] for i in range(num_labels)])
        y_sample_cartesian = np.zeros((3*num_labels), dtype='int32')
        for annotation in sample:
            if sub_only:
                property_pair = annotation[0].split('#')[-1]
            else:
                property_pair = annotation[0]

            aspect_term = annotation[1]
            polarity = annotation[2]
        
            id_property_pair = label_name_to_id[property_pair]
            id_polarity = polarity_name_to_id[polarity]

            cartesian_id = id_property_pair * 3 + id_polarity-1
            
            y_sample_sparse[id_property_pair] = id_polarity
            y_sample_onehot[id_property_pair][id_polarity] = 1
            y_sample_onehot[id_property_pair][0] = 0
            y_sample_cartesian[cartesian_id] = 1
        y_train_sparse.append(y_sample_sparse)
        y_train_onehot.append(y_sample_onehot)
        y_train_cartesian.append(y_sample_cartesian)
    y_train_sparse = np.array(y_train_sparse)
    y_train_onehot = np.array(y_train_onehot)
    y_train_cartesian = np.array(y_train_cartesian)
    return y_train_sparse, y_train_onehot, y_train_cartesian

def prep_data_main_only(y_data_train, num_labels, label_name_to_id):
    y_train = []
    for sample in y_data_train:
        y_sample = np.zeros((num_labels), dtype='int32')
        for annotation in sample:
            main_property = annotation[0].split('#')[0]
            id_main_property = label_name_to_id[main_property]
            y_sample[id_main_property] = 1
        y_train.append(y_sample)
    y_train = np.array(y_train)
    return y_train


def jsonlload(fname, encoding="utf-8"):
    json_list = []
    with open(fname, encoding=encoding) as f:
        for line in f.readlines():
            json_list.append(json.loads(line))
    return json_list

def preprocess_for_BERT(X_data, tokenizer):
    max_seq_len = 60
    X_temp = tokenizer(X_data,
                       padding=True,
                       truncation=True,
                    #    return_tensors='pt'
                       max_length=max_seq_len)
    
    X_train = [X_temp['input_ids'], X_temp['attention_mask']]
    return X_train

# type = ['main', 'sub']
def process_data(data_mode='sub', 
                 batch_size=16, 
                 return_labels=False, 
                 steal_val_ratio=0,
                 bert_type="monologg/koelectra-base-v3-discriminator"):
    print('Preprocessing data...')
    config_directory = 'config/config.yaml'
    
    train_data_path = 'data/nikluge-sa-2022-train.jsonl'
    val_data_path = 'data/nikluge-sa-2022-dev.jsonl'
    test_data_path = 'data/nikluge-sa-2022-test.jsonl'

    train_data = jsonlload(train_data_path)
    val_data = jsonlload(val_data_path)
    test_data = jsonlload(test_data_path)

    X_data_train = []
    y_data_train = []
    for sample in train_data:
        X_data_train.append(sample['sentence_form'])
        y_data_train.append(sample['annotation'])
    
    X_data_val = []
    y_data_val = []
    
    if steal_val_ratio != 0:
        random.shuffle(val_data)
    
    for sample in val_data:
        X_data_val.append(sample['sentence_form'])
        y_data_val.append(sample['annotation'])
    
    steal_id = int(steal_val_ratio * len(X_data_val))
        
    X_data_stolen = X_data_val[:steal_id]
    y_data_stolen = y_data_val[:steal_id]
    X_data_val = X_data_val[steal_id:]
    y_data_val = y_data_val[steal_id:]
    
    X_data_train = X_data_train + X_data_stolen
    y_data_train = y_data_train + y_data_stolen    
    
    X_data_test = []
    y_data_test = []
    for sample in test_data:
        X_data_test.append(sample['sentence_form'])
        y_data_test.append(sample['annotation'])
    """
    23 in total
    제품 전체: 6개
    본품: 7개
    패키지: 6개
    브랜드: 4개
    """
    tokenizer = AutoTokenizer.from_pretrained(bert_type)
    
    X_train = preprocess_for_BERT(X_data_train, tokenizer)
    X_val = preprocess_for_BERT(X_data_val, tokenizer)
    X_test = preprocess_for_BERT(X_data_test, tokenizer)
    
    # Preprocess Sub-cateogires + sentiments
    if data_mode == 'sub':
        label_id_to_name, label_name_to_id, polarity_id_to_name, polarity_name_to_id = prepare_ids(y_data_train, 
                                                                                                   y_data_val, 
                                                                                                   main_only=False,
                                                                                                   sub_only=True)
        num_labels = len(label_id_to_name)
        
        y_train_sparse, y_train_onehot, y_train_cartesian = prep_data(y_data_train, num_labels, label_name_to_id, polarity_name_to_id, sub_only=True)
        y_val_sparse, y_val_onehot, y_val_cartesian = prep_data(y_data_val, num_labels, label_name_to_id, polarity_name_to_id, sub_only=True)
        
        train_labels = torch.tensor(y_train_cartesian)
        val_labels = torch.tensor(y_val_cartesian)
        
    # Preprocess Main categories
    elif data_mode == 'main':
        label_id_to_name, label_name_to_id, polarity_id_to_name, polarity_name_to_id = prepare_ids(y_data_train, 
                                                                                                   y_data_val, 
                                                                                                   main_only=True,
                                                                                                   sub_only=False)
        num_labels = len(label_id_to_name)
    
        y_train_main_only = prep_data_main_only(y_data_train, num_labels, label_name_to_id)
        y_val_main_only = prep_data_main_only(y_data_val, num_labels, label_name_to_id)
    
        train_labels = torch.tensor(y_train_main_only)
        val_labels = torch.tensor(y_val_main_only)
    
    if return_labels:
        return train_labels, val_labels
        
    train_data = TensorDataset(torch.tensor(X_train[0]),
                               torch.tensor(X_train[1]),
                               train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, 
                                  sampler=train_sampler, 
                                  batch_size=batch_size)

    val_data = TensorDataset(torch.tensor(X_val[0]),
                             torch.tensor(X_val[1]),
                             val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data,
                                sampler=val_sampler,
                                batch_size=batch_size)
    test_data = TensorDataset(torch.tensor(X_test[0]),
                                torch.tensor(X_test[1]))
    X_test = tuple(torch.tensor(i) for i in X_test)

    return train_dataloader, val_dataloader, X_test
