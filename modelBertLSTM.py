from nltk.util import pr

import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn.functional as F
from pytorch_pretrained_bert import BertForMaskedLM
class BertLSTMPunc(nn.Module):
    def __init__(self, segment_size,hidden_size,num_layers ,output_size, p):
        super(BertLSTMPunc, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained('chinese_rbt3_pytorch')
        self.segment_size = segment_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bert_vocab_size = 21128
        self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        self.lstm = nn.LSTM(self.bert_vocab_size,self.hidden_size,self.num_layers,bidirectional = True,batch_first = True)
    
        self.fc = nn.Linear(hidden_size*2, output_size)
        self.p = p
        self.Dropout = nn.Dropout(p = self.p)
    def forward(self, x,segment):
        x = self.bert(x,segment) 
        output,(h_n) = self.lstm(x)
        output = self.fc(output) #output : (batch_size , seq_len , output_size)
        return output
    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)########
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(package['segment_size'], package['hidden_size'], package['num_layers'], package['output_size'],
                    package['p'])
        model.load_state_dict(package['state_dict'])
        return model
        
    def serialize(self,model, optimizer,scheduler, epoch,train_loss,val_loss):
        package = {
            # hyper-parameter
            'segment_size': model.segment_size,
            'hidden_size': model.hidden_size,
            'num_layers':model.num_layers,
            'output_size': model.output_size,
            'p':model.p,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss':val_loss
        }
        return package