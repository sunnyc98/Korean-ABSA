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
import wandb
import copy

def scheduler(avg_train_loss, lr):
    if avg_train_loss > 0.1:
        return lr    
    else:
        print('Learning rate scheduled to {:.3f}'.format(lr * 0.9))
        return lr * 0.9

def inintialize_model(mode, device, freeze_bert=False, sub_only=False):
    if mode == 'Cartesian':
        model = Cartesian(device=device, sub_only=sub_only, freeze_bert=freeze_bert)
    elif mode == 'MainOnly':
        model = MainCategoryClassifier(device=device, freeze_bert=freeze_bert)
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
                return
             
            # Saving best model (Val acc)
            if val_f1 > best_val_f1:
                print('Val f1 increased from {:.3f} to {:.3f}, saved new best model at {}'.format(best_val_f1,
                                                                                            val_f1,
                                                                                            save_path))
                best_val_f1 = val_f1
                torch.save(model.state_dict(), save_path)
                # torch.save(model, save_path)
        
        print('\n')    
    print('Training complete!')
    return best_val_f1

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

def Experiment(config, class_weights):
    class_weights_sub, class_weights_main = class_weights
    
    project_name = config['project_name']
    train_mode = config['train_mode']
    exp_number = config['exp_number']
    lr = config['lr']
    epochs = config['epochs']
    batch_size = config['batch_size']
    dropout_ratio = config['dropout_ratio']
    run_name = train_mode + str(exp_number)
    
    run = wandb.init(project=project_name, 
                     config={'train_mode': train_mode,
                             'lr': lr,
                             'batch_size': batch_size,
                             'epochs': epochs,
                             'dropout_ratio': dropout_ratio},
                     name=run_name)
    wandb_config = wandb.config
    
    model_list = ['Cartesian', 'MainOnly', 'Cartesian2', 'Add-one_sparse', 'Add-one_onehot']
    if train_mode == 'main':
        model_type = model_list[1]
        sub_only=False
        # class_weight = class_weights_main
        train_dataloader, val_dataloader, X_test = process_data(data_mode='main',
                                                                batch_size=batch_size,
                                                                steal_val_ratio=0.5)
    elif train_mode == 'sub':
        model_type = model_list[0]
        sub_only=True
        # class_weight = class_weights_sub
        train_dataloader, val_dataloader, X_test = process_data(data_mode='sub',
                                                                batch_size=batch_size,
                                                                steal_val_ratio=0.5)

    save_path = 'saved_model/' + train_mode + '_' + str(exp_number) + '.pt'
    # save_path = 'saved_model/models/' + train_mode + '_' + str(exp_number) + '.pickle'
    # loss_fn = nn.BCELoss(weight=class_weight)
    loss_fn = nn.BCELoss()
    model = inintialize_model(model_type, device, sub_only=sub_only)
    wandb.watch(model, log_freq=100)
    best_val_f1 = train(model, train_dataloader, val_dataloader, 
                        save_path = save_path,
                        lr=lr, 
                        epochs=epochs, 
                        evaluation=True,
                        loss_fn=loss_fn
                        )
    wandb.log({'Val_F1_score':best_val_f1})

if __name__ == '__main__':
    CUDA_LAUNCH_BLOCKING=1
    set_seed(43)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print('No GPU available.')
        exit()

    class_weights = get_class_weights()
    
    config_dir = 'config/config copy ' + str(6) + '.yaml'
    with open(config_dir) as input_file:
        config = yaml.safe_load(input_file)
    Experiment(config, class_weights)

    config_dir = 'config/config copy ' + str(12) + '.yaml'
    with open(config_dir) as input_file:
        config = yaml.safe_load(input_file)
    Experiment(config, class_weights)

