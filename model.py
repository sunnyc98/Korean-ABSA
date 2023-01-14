
import torch
import torch.nn as nn
from transformers import ElectraModel, ElectraTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from transformers import AutoModel, AutoConfig

class SimpleClassifier(nn.Module):

    def __init__(self, args, num_label):
        super().__init__()
        self.dense = nn.Linear(args.classifier_hidden_size, args.classifier_hidden_size)
        self.dropout = nn.Dropout(args.classifier_dropout_prob)
        self.output = nn.Linear(args.classifier_hidden_size, num_label)

    def forward(self, features):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

# For Cartesian labels
class Cartesian(nn.Module):
    def __init__(self, device, 
                 sub_only=False, 
                 freeze_bert=False,
                 bert_type="monologg/koelectra-base-v3-discriminator"):
        super(Cartesian, self).__init__()
        if 'large' in bert_type:
            D_in = 1024
        elif 'base' in bert_type:
            D_in = 768
        else:
            print('Use base or large type model')
            exit()
            
        self.max_seq_len = 60
        self.sentiments = 3
        dropout_ratio = 0.1
        if sub_only: self.categories = 7
        else: self.categories = 23
        D_out = self.sentiments * self.categories
        
        self.bert = AutoModel.from_pretrained(bert_type)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout_ratio),
                                        nn.Linear(D_in, D_in),
                                        nn.Tanh(),
                                        nn.Dropout(p=dropout_ratio),
                                        nn.Linear(D_in, D_out),
                                        nn.Sigmoid()).to(device)
        self.dropout = nn.Dropout(p=dropout_ratio)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # self.pool = nn.AvgPool1d()
        
        
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, 
                           attention_mask=attention_mask)[0]  # (batch, max_seq_len, hidden_dim)
        
        # To Try: Global Avg Pooling
        # output = output.permute(0,2,1) # (batch, hidden_dim, max_seq_len)
        # output = self.pool(self.max_seq_len)(output) # (batch, hidden_dim, 1)
        # output = torch.squeeze(output) # (batch, hidden_dim)
        
        last_hidden_cls = output[:, 0, :] # (batch, hidden_dim)
        last_hidden_clst = self.dropout(last_hidden_cls)
        final_output = self.classifier(last_hidden_cls)
        
        return final_output # (batch, sentiments*categories)
    
class MainCategoryClassifier(nn.Module):
    def __init__(self, device, 
                 freeze_bert=False, 
                 bert_type="monologg/koelectra-base-v3-discriminator"):
        super(MainCategoryClassifier, self).__init__()
        if 'large' in bert_type:
            D_in = 1024
        elif 'base' in bert_type:
            D_in = 768
        else:
            print('Use base or large type model')
            exit()
        self.main_categories = 4
        dropout_ratio = 0.1
        D_out = self.main_categories
        self.bert = AutoModel.from_pretrained(bert_type)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout_ratio),
                                        nn.Linear(D_in, D_in),
                                        nn.Tanh(),
                                        nn.Dropout(p=dropout_ratio),
                                        nn.Linear(D_in, D_out),
                                        nn.Sigmoid()).to(device)
        self.dropout = nn.Dropout(p=dropout_ratio)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
                
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, 
                           attention_mask=attention_mask)[0]
        CLS_output_token = output[:,0,:] # (batch, hidden_dim)
        # CLS_output_token = self.dropout(CLS_output_token)
        final_output = self.classifier(CLS_output_token)
        
        return final_output
    
class Cartesian2(nn.Module):
    def __init__(self, device, sub_only=False, freeze_bert=False ):
        super(Cartesian2, self).__init__()
        D_in = 768
        self.max_seq_len = 60
        self.sentiments = 3
        if sub_only: self.categories = 7
        else: self.categories = 23
        D_out = self.sentiments * self.categories
        
        self.bert = AutoModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.classifier = nn.Sequential(nn.Linear(D_in, D_out),
                                         nn.Sigmoid()).to(device)
        self.pool = nn.AvgPool1d(self.max_seq_len)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, 
                           attention_mask=attention_mask)[0]  # (batch, max_seq_len, hidden_dim)
        
        # To Try: Global Avg Pooling
        output = output.permute(0,2,1) # (batch, hidden_dim, max_seq_len)
        pooled = self.pool(output) # (batch, hidden_dim, 1)
        pooled = torch.squeeze(pooled) # (batch, hidden_dim)
        
        final_output = self.classifier(pooled)
        
        return final_output # (batch, sentiments*categories)
    
# For one-hot encoded data and using custom cce loss
class AddOne_onehot(nn.Module):
    def __init__(self, device, freeze_bert=False):
        super(AddOne_onehot, self).__init__()
        D_in = 768
        self.sentiments = 4
        self.categories = 23
        
        self.bert = AutoModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.classifier = [nn.Sequential(nn.Linear(D_in, self.sentiments),
                                         nn.Softmax(dim=-1)).to(device)
                            ] * self.categories
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids=input_ids, 
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)  # (batch, max_seq_len, hidden_dim)

        last_hidden_cls = output[0][:, 0, :] # (batch, hidden_dim)

        logits = []
        for i in range(self.categories):
            logit = self.classifier[i](last_hidden_cls)
            logits.append(logit)
        
        final_output = torch.stack(logits, dim=1)
        return final_output # (batch, categories, sentiments)

# For sparse data and using nn.cce
class AddOne_sparse(nn.Module):
    def __init__(self, device, sub_only=False, freeze_bert=False):
        super(AddOne_sparse, self).__init__()
        D_in = 768
        self.sentiments = 4
        self.categories = 7
        
        self.bert = AutoModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.classifier = [nn.Sequential(nn.Linear(D_in, self.sentiments),
                                         nn.ReLU()).to(device)
                            ] * self.categories
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids=input_ids, 
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)[0]  # (batch, max_seq_len, hidden_dim)
        
        last_hidden_cls = output[:, 0, :] # (batch, hidden_dim)

        logits = [] 
        for i in range(self.categories): 
            logit = self.classifier[i](last_hidden_cls) #(batch, sentiments)
            logits.append(logit)
        final_output = torch.stack(logits, dim=-1)
        return final_output # (batch, sentiments, categories)
