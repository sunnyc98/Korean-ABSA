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
import yaml
import copy
import yaml
import pickle

def jsondump(j, fname):
    with open(fname, "w", encoding="UTF8") as f:
        json.dump(j, f, ensure_ascii=False)

def scheduler(avg_train_loss, lr):
    if avg_train_loss > 0.1:
        return lr    
    else:
        print('Learning rate scheduled to {:.3f}'.format(lr * 0.9))
        return lr * 0.9

def inintialize_model(mode, device, freeze_bert=False, sub_only=False, bert_type="monologg/koelectra-base-v3-discriminator"):
    if mode == 'Cartesian':
        model = Cartesian(device=device, sub_only=sub_only, freeze_bert=freeze_bert, bert_type=bert_type)
    elif mode == 'MainOnly':
        model = MainCategoryClassifier(device=device, freeze_bert=freeze_bert, bert_type=bert_type)
    else:
        print('Wrong mode')
        return
    model.to(device)
    return model

def set_seed(seed_value=43):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, 
          train_dataloader, 
          val_dataloader=None,
          loss_fn=nn.BCELoss(),
          save_path=None, 
          epochs=30, 
          lr=1e-5, 
          evaluation=True, 
          patience=2, 
          sparse_data=False
          ):
    print('Training...')
    best_val_path = save_path + ".pt"
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    cur_patience = 0
    best_val_loss = 1e5
    best_val_f1 = 0
    for epoch in range(epochs):
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)
        
        t0_epoch, t0_batch = time.time(), time.time()
        
        total_loss, batch_loss, batch_counts = 0,0,0
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = next(iter(train_dataloader))
            batch_counts += 1
            b_input_ids, b_attention_mask, b_labels = tuple(b.to(device) for b in batch)
            
            model.zero_grad()
            
            logits = model(b_input_ids, b_attention_mask)
            loss = loss_fn(logits, b_labels.type(torch.float))

            batch_loss += loss.item()
            total_loss += loss.item()
            
            loss.backward()
            
            clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            if (step%20 ==0 and step != 0) or (step == len(train_dataloader)-1):
                time_elapsed = time.time() - t0_batch
                print(f"{epoch + 1:^7} | {str(step)+'/'+str(len(train_dataloader)):^7} | {batch_loss / batch_counts:^12.3f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
                
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()
            avg_train_loss = total_loss / len(train_dataloader)
        
        print("-"*70)
        if evaluation==True:
            val_loss, val_acc, val_f1 = evaluate(model, 
                                                 val_dataloader, 
                                                 loss_fn=loss_fn,
                                                 sparse_data=sparse_data)
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch + 1:^7} | {'-':^7} | {avg_train_loss:^12.3f} | {val_loss:^10.3f} | {val_acc:^9.2f} | {time_elapsed:^9.2f}")
            print('Val F1 score: {:.3f}'.format(val_f1))
            print("-"*70)
            lr = scheduler(avg_train_loss, lr)
            # Early stopping with patience
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                cur_patience = 0
            else:
                print('Val_loss did not improve.')
                cur_patience += 1
            if cur_patience >= patience:
                print('Early Stopping. Training complete!')
                return best_save_path
             
            # Saving best model (Val acc)
            if val_f1 > best_val_f1:
                print('Val f1 increased from {:.3f} to {:.3f}, saved new best model at {}'.format(best_val_f1,
                                                                                            val_f1,
                                                                                            save_path))
                best_val_f1 = val_f1
                best_save_path = save_path + f"_val{best_val_f1:.3f}" + ".pt"
                torch.save(model.state_dict(), best_save_path)
                # torch.save(model, save_path)
        
        print('\n')    
    print('Training complete!')
    return best_save_path

def evaluate(model, val_dataloader, 
             loss_fn=nn.BCELoss(),
             sparse_data=False, 
             show_progress=False,
             return_predictions=False,
             ):
    model.eval()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print('No GPU available.')
        exit()
    
    
    val_acc = []
    val_loss = []
    val_f1 = []

    predictions = []
    true_labels = []
    
    past_percentage = 0
    for index, batch in enumerate(val_dataloader):
        b_input_ids, b_attention_mask, b_labels = tuple(b.to(device) for b in batch)
        
        with torch.no_grad():
            logits = model(b_input_ids, b_attention_mask)
        loss = loss_fn(logits, b_labels.type(torch.float))
        val_loss.append(loss.item())
        
        if sparse_data:
            preds = torch.argmax(logits, dim=1)
        else:
            preds = logits.argmax(-1)
            b_labels = b_labels.argmax(-1)

        predictions.append(preds.cpu().numpy())
        true_labels.append(b_labels.cpu().numpy())

        f1 = f1_score(preds.cpu().numpy(), b_labels.cpu().numpy(), average='micro')
        val_f1.append(f1)
        
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_acc.append(accuracy)
        if show_progress:
            percentage = int(index / len(val_dataloader) * 100)
            if percentage%20 == 0 and percentage != past_percentage:
                print('Evaluation {}% complete'.format(percentage))
                past_percentage = percentage
        
    val_loss = np.mean(val_loss)
    val_acc = np.mean(val_acc)
    val_f1 = np.mean(val_f1)
    if return_predictions:
        print('acc: {:.3f}\nf1: {:.3f}'.format(val_acc, val_f1))
        return np.concatenate(predictions), np.concatenate(true_labels)
        
    return val_loss, val_acc, val_f1

def get_class_weights():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print('No GPU available.')
        exit()
    train_labels_sub, val_labels_sub = process_data(data_mode='sub',
                                                    return_labels=True)
    train_labels_main, val_labels_main = process_data(data_mode='main',
                                                    return_labels=True)
    # For class weights
    train_labels_sub_sparse = train_labels_sub.argmax(-1).numpy()
    val_labels_sub_sparse = val_labels_sub.argmax(-1).numpy()
    train_labels_main_sparse = train_labels_main.argmax(-1).numpy()
    val_labels_main_sparse = val_labels_main.argmax(-1).numpy()
    
    labels_sub_sparse = np.concatenate([train_labels_sub_sparse, val_labels_sub_sparse])
    labels_main_sparse = np.concatenate([train_labels_main_sparse, val_labels_main_sparse])

    
    class_weights_sub = np.array(compute_class_weight('balanced',
                                                      np.unique(labels_sub_sparse),
                                                      labels_sub_sparse))
    
    class_weights_sub = np.log(class_weights_sub).astype('int32')
    if np.min(class_weights_sub) < 0:
        class_weights_sub -= np.min(class_weights_sub)
    class_weights_sub += 1
    
    for i in range(21):
        if i not in np.unique(labels_sub_sparse):
            class_weights_sub = np.insert(class_weights_sub, i, 0)
    class_weights_sub = torch.tensor(class_weights_sub).to(device)
    
    class_weights_main = np.array(compute_class_weight('balanced',
                                                np.unique(labels_main_sparse),
                                                labels_main_sparse))
    
    class_weights_main = np.log(class_weights_main).astype('int32')
    if np.min(class_weights_main) < 0:
        class_weights_main -= np.min(class_weights_main)
    class_weights_main += 1
    
    for i in range(4):
        if i not in np.unique(labels_main_sparse):
            class_weights_main = np.insert(class_weights_main, i, 0)
    class_weights_main = torch.tensor(class_weights_main).to(device)
    
    return (class_weights_sub, class_weights_main)

def Experiment(config, base_model, class_weights=None):
    if class_weights != None:
        class_weights_sub, class_weights_main = class_weights
    
    project_name = config['project_name']
    train_mode = config['train_mode']
    exp_number = config['exp_number']
    lr = config['lr']
    epochs = config['epochs']
    batch_size = config['batch_size']
    dropout_ratio = config['dropout_ratio']
    run_name = train_mode + str(exp_number)
    
    model_list = ['Cartesian', 'MainOnly', 'Cartesian2', 'Add-one_sparse', 'Add-one_onehot']
    if train_mode == 'main':
        model_type = model_list[1]
        sub_only=False
        # class_weight = class_weights_main
        train_dataloader, val_dataloader, X_test = process_data(data_mode='main',
                                                                batch_size=batch_size,
                                                                steal_val_ratio=0.5,
                                                                bert_type=base_model)
    elif train_mode == 'sub':
        model_type = model_list[0]
        sub_only=True
        # class_weight = class_weights_sub
        train_dataloader, val_dataloader, X_test = process_data(data_mode='sub',
                                                                batch_size=batch_size,
                                                                steal_val_ratio=0.5,
                                                                bert_type=base_model)

    base_model_path = base_model.split("/")[-1]
    save_path = 'saved_model/new_classifier/' + base_model_path + "_" + train_mode + '_' + str(exp_number)
        
    # loss_fn = nn.BCELoss(weight=class_weight)
    loss_fn = nn.BCELoss()
    model = inintialize_model(model_type, device, sub_only=sub_only, bert_type=base_model)
    best_save_path = train(model, train_dataloader, val_dataloader, 
                        save_path = save_path,
                        lr=lr, 
                        epochs=epochs, 
                        evaluation=True,
                        loss_fn=loss_fn
                    )
    return best_save_path, val_dataloader, X_test

def get_config(num):
    config_dir = 'config/config copy ' + str(num) + '.yaml'
    with open(config_dir) as input_file:
        config = yaml.safe_load(input_file)
        
    return config

if __name__ == '__main__':
    CUDA_LAUNCH_BLOCKING=1
    set_seed(43)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print('No GPU available.')
        exit()
    
    # =========================================== Sub Category Classification ==============================================
    
    pretrained_models = [#"monologg/koelectra-base-v3-discriminator",  
                         'klue/roberta-base',  
                         'klue/roberta-large',  
                         'xlm-roberta-base',
                         'xlm-roberta-large',  
                         'bert-base-multilingual-uncased', 
                         'kykim/funnel-kor-base',
                         "kykim/electra-kor-base",
                         "beomi/KcELECTRA-base-v2022",
                        ]
    
    for model in pretrained_models:
        print("=" * 40)
        print("\n")
        print(f"Training {model}...")
        print("\n")
        config = get_config(12)
        best_save_path, val_dataloader, X_test = Experiment(config, model)

    
    assert(0)
    train_mode = config['train_mode']
    exp_number = config['exp_number']
    batch_size = config['batch_size']
    model_type = 'Cartesian'
    sub_only=True

    loss_fn = nn.BCELoss()
    model_sub = inintialize_model(model_type, device, sub_only=sub_only)
    model_sub.load_state_dict(torch.load(best_save_path))
    
    
    torch.cuda.empty_cache()
    with torch.no_grad():
        preds_sub, true_sub = evaluate(model_sub, 
                                        val_dataloader, 
                                        show_progress=True, 
                                        return_predictions=True)
    
    
    with torch.no_grad():
        preds_test_sub = model_sub(X_test[0].to(device),X_test[1].to(device))
    preds_test_sub = preds_test_sub.argmax(-1)
    
    # =========================================== Main Category Classification ==============================================
    
    config = get_config(12)
    Experiment(config)
    
    train_mode = config['train_mode']
    exp_number = config['exp_number']
    batch_size = config['batch_size']
    model_type = "MainOnly"
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
    
    # =========================================== Create Prediction File ==============================================

    assert(0)
    
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
    

