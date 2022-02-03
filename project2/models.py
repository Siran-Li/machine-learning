import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BigBirdTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig, BigBirdForSequenceClassification, GPT2Tokenizer, GPT2ForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import nltk

"""This class is for TextCNN"""
class textCNN(nn.Module):
    def __init__(self, inplane=1, input_dim=768, num_conv=3, conv_size=[2,3,4], dropout_prob=0, dim_output=2):
        super(textCNN, self).__init__()
        
        D_words = input_dim # dimension of word embedding
        self.convs = nn.ModuleList([nn.Conv2d(inplane,num_conv,(K,input_dim)) for K in conv_size]) ## list of convolutional layers
        self.dropout = nn.Dropout(dropout_prob) 
        self.fc = nn.Linear(len(conv_size)*num_conv, dim_output) 
        
    def forward(self,x):
        #x.size = (batch_size, sequence_length, word_embedding)
        
        x = x.unsqueeze(1) #(N,C,W,D) (C=1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] # len(conv_size)*(N,num_conv,W)
        x = [F.max_pool1d(line,line.size(2)).squeeze(2) for line in x]  # len(conv_size)*(N,num_conv)
        
        x = torch.cat(x,1) #(N,num_conv*len(conv_size))
        x = self.dropout(x)
        logit = self.fc(x)
        return logit

"""This class is for BiLSTM"""
class LSTM_attention(nn.Module):
    def __init__(self, input_dim=768, hidden_size=256, num_layers=1, dim_output=2, bi_directional=True):
        super(LSTM_attention, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bi_directional, bias=True)
        self.fc = nn.Linear((int(bi_directional)+1) * hidden_size, dim_output)

    def attention_layer(self,lstm_output, final_state):
        # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
        # final_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        batch_size = len(lstm_output)
        hidden = torch.cat((final_state[0], final_state[1]), dim=1).unsqueeze(2)
        # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights,1)

        # context: [batch_size, n_hidden * num_directions(=2)]
        context = torch.bmm(lstm_output.transpose(1,2),soft_attn_weights.unsqueeze(2)).squeeze(2)

        return context, soft_attn_weights

    def forward(self, inputs):
        output, (final_hidden_state, final_cell_state) = self.lstm(inputs.permute(1, 0, 2))
        atten_output, attention = self.attention_layer(output.permute(1, 0, 2), final_hidden_state)
        output = self.fc(atten_output)
        
        return output

"""This class is for transformers"""
class Transformer:
    def __init__(self, model_name, num_labels=2, **kwargs):
        super(Transformer, self).__init__()
    
        if model_name == 'BERT':
            self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = num_labels, output_attentions = False, output_hidden_states = True)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        elif model_name == 'GPT2':
            self.model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels = num_labels, output_attentions = False, output_hidden_states = True)
            self.model.config.pad_token_id = self.model.config.eos_token_id
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', do_lower_case=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token 
        elif model_name == 'BIGBIRD':
            self.model = BigBirdForSequenceClassification.from_pretrained('google/bigbird-roberta-base', num_labels = num_labels, output_attentions = False, output_hidden_states = True)
            self.tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base', do_lower_case=True)

    def preprocess_data(self, X_train=None, X_test=None, y_train=None, y_test=None):
        """This function convert the text to compatible format of transformers"""
        if X_train == None:
          train_dataset = None
          modes = ['test']
        else:
          modes = ['train', 'test']

        for mode in modes:

            sample_ids = []
            attention_masks = []

            samples = X_train if mode == 'train' else X_test
            labels = y_train if mode == 'train' else y_test
            length = len(samples)

            for i, sent in enumerate(samples):
                encoded_dict = self.tokenizer.encode_plus(sent, add_special_tokens = True, max_length = 100, truncation = True, \
                                                  padding = 'max_length', return_attention_mask = True, return_tensors = 'pt')

                # Add the encoded sample and mask 
                sample_ids.append(encoded_dict['input_ids'])
                attention_masks.append(encoded_dict['attention_mask'])
                print('\r----- Processing {}/{} {} samples'.format(i+1, length, mode), flush=True, end='')

            # Convert to pytorch tensors.
            sample_ids = torch.cat(sample_ids, dim=0)
            attention_masks = torch.cat(attention_masks, dim=0)
            labels = torch.tensor(labels)

            if mode == 'train': train_dataset = TensorDataset(sample_ids, attention_masks, labels)
            else: test_dataset = TensorDataset(sample_ids, attention_masks, labels)
        print('\n')

        if X_train == None:
          return test_dataset
        else:
          return train_dataset, test_dataset 

"""This class is for transformers + downstream models"""
class transformer_classifier:
    def __init__(self, model, classifier, device):
        self.model = model.to(device)
        self.classifier = classifier.to(device)
  
    def __call__(self, x_id, token_type_ids, attention_mask, labels):
        with torch.no_grad():  
            word_embedding = self.model(x_id, token_type_ids=None, attention_mask=attention_mask, labels=labels)['hidden_states'][-1]
            logits = self.classifier(word_embedding)
    
        return logits.max(1)[1]


  
  
        