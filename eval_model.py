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
from preprocess_data import process_data, jsonlload, prepare_ids
from train import set_seed, inintialize_model, evaluate, get_class_weights
import copy
import yaml
import pickle

def jsondump(j, fname):
    with open(fname, "w", encoding="UTF8") as f:
        json.dump(j, f, ensure_ascii=False)

if __name__ == '__main__':
    CUDA_LAUNCH_BLOCKING=1
    set_seed(43)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print('No GPU available.')
        exit()

    class_weights_sub, class_weights_main = get_class_weights()
    model_list = ['Cartesian', 'MainOnly', 'Cartesian2', 'Add-one_sparse', 'Add-one_onehot']
    
    config_number = 6
    config_dir = 'config/config copy ' + str(config_number) + '.yaml'
    with open(config_dir) as input_file:
            config = yaml.safe_load(input_file)
    train_mode = config['train_mode']
    exp_number = config['exp_number']
    batch_size = config['batch_size']
    model_type = model_list[0]
    sub_only=True
    train_dataloader, val_dataloader, X_test = process_data(data_mode='sub',
                                                            batch_size=batch_size)

    save_path = 'saved_model/' + train_mode + '_' + str(exp_number) + '.pt'

    loss_fn = nn.BCELoss()
    model_sub = inintialize_model(model_type, device, sub_only=sub_only)
    model_sub.load_state_dict(torch.load(save_path))
    
    torch.cuda.empty_cache()
    with torch.no_grad():
        preds_sub, true_sub = evaluate(model_sub, 
                                        val_dataloader, 
                                        show_progress=True, 
                                        return_predictions=True)
    with torch.no_grad():
        preds_test_sub = model_sub(X_test[0].to(device),X_test[1].to(device))
    preds_test_sub = preds_test_sub.argmax(-1)
    
    exp_number = 12
    config_dir = 'config/config copy ' + str(exp_number) + '.yaml'
    with open(config_dir) as input_file:
            config = yaml.safe_load(input_file)
    train_mode = config['train_mode']
    exp_number = config['exp_number']
    batch_size = config['batch_size']
    model_type = model_list[1]
    sub_only=False
    train_dataloader_main, val_dataloader_main, X_test = process_data(data_mode='main',
                                                            batch_size=batch_size)
    
    save_path = 'saved_model/' + train_mode + '_' + str(exp_number) + '.pt'
    
    model_main = inintialize_model(model_type, device, sub_only=sub_only)
    model_main.load_state_dict(torch.load(save_path))
    
    torch.cuda.empty_cache()
    with torch.no_grad():
        preds_main, true_main = evaluate(model_main, 
                                        val_dataloader_main, 
                                        show_progress=True, 
                                        return_predictions=True)
    with torch.no_grad():
        preds_test_main = model_main(X_test[0].to(device),X_test[1].to(device))
    preds_test_main = preds_test_main.argmax(-1)
    
    with open('test_pred/preds_test_main.pickle', 'wb') as output:
        pickle.dump(preds_test_main, output)
        
    with open('test_pred/preds_test_sub.pickle', 'wb') as output:
        pickle.dump(preds_test_sub, output)
        
    config_directory = 'config/config.yaml'
    
    train_data_path = 'data/nikluge-sa-2022-train.jsonl'
    val_data_path = 'data/nikluge-sa-2022-dev.jsonl'
    test_data_path = 'data/nikluge-sa-2022-test.jsonl'
    
    train_data = jsonlload(train_data_path)
    val_data = jsonlload(val_data_path)
    test_data = jsonlload(test_data_path)

    y_data_train = []
    for sample in train_data:
        y_data_train.append(sample['annotation'])
    
    y_data_val = []
    for sample in val_data:
        y_data_val.append(sample['annotation'])

    
    label_id_to_name, label_name_to_id, polarity_id_to_name, polarity_name_to_id = prepare_ids(y_data_train, 
                                                                                                y_data_val, 
                                                                                                main_only=False,
                                                                                                sub_only=True)
    # cartesian id to subcategory and sentiment id
    cart_id_to_sub_sent_id = [(int(i/3),i%3) for i in range(21)] # cartesian id: [0,21)
    
    preds_sub_sent_id = [cart_id_to_sub_sent_id[i] for i in preds_sub]
    preds_sub_sent_name = [(label_id_to_name[i], polarity_id_to_name[j]) for i,j in preds_sub_sent_id]
    
    label_id_to_name, label_name_to_id, polarity_id_to_name, polarity_name_to_id = prepare_ids(y_data_train, 
                                                                                                y_data_val, 
                                                                                                main_only=True,
                                                                                                sub_only=False)
    main_id_to_name = [label_id_to_name[i] for i in preds_main]
    
    final_pred = [(main_id_to_name[i]+'#'+preds_sub_sent_name[i][0],
                   preds_sub_sent_name[i][1]) for i in range(len(main_id_to_name))]
    pred_data = [[[final_pred[i][0],final_pred[i][1]]] for i in range(len(final_pred))]

    pred_data_json = copy.deepcopy(val_data)
    for i in range(len(val_data)):
        pred_data_json[i]['annotation'] = pred_data[i]
                
    jsondump(pred_data_json, './pred_val.json')

    # creating test data
    label_id_to_name, label_name_to_id, polarity_id_to_name, polarity_name_to_id = prepare_ids(y_data_train, 
                                                                                                y_data_val, 
                                                                                                main_only=False,
                                                                                                sub_only=True)
    # cartesian id to subcategory and sentiment id
    cart_id_to_sub_sent_id = [(int(i/3),i%3) for i in range(21)] # cartesian id: [0,21)
    preds_test_sub[10]
    preds_sub_sent_id = [cart_id_to_sub_sent_id[i] for i in preds_test_sub]
    preds_sub_sent_name = [(label_id_to_name[i], polarity_id_to_name[j]) for i,j in preds_sub_sent_id]
    
    label_id_to_name, label_name_to_id, polarity_id_to_name, polarity_name_to_id = prepare_ids(y_data_train, 
                                                                                                y_data_val, 
                                                                                                main_only=True,
                                                                                                sub_only=False)
    main_id_to_name = [label_id_to_name[i] for i in preds_test_main]
    
    final_pred = [(main_id_to_name[i]+'#'+preds_sub_sent_name[i][0],
                   preds_sub_sent_name[i][1]) for i in range(len(main_id_to_name))]
    pred_data = [[[final_pred[i][0],final_pred[i][1]]] for i in range(len(final_pred))]

    pred_data_json = copy.deepcopy(test_data)
    for i in range(len(test_data)):
        pred_data_json[i]['annotation'] = pred_data[i]

    jsondump(pred_data_json, './pred_test_data.json')
    