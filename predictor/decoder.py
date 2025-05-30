import torch
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
device = torch.device('cuda:0')

'Decoder structure'
class Decoder(nn.Module):
    
    def __init__(self,p,pre_len):
        super(Decoder, self).__init__()
        self.pre_len=pre_len
        self.lstmcell = nn.LSTMCell(input_size=128,hidden_size=128)
        self.Batchnorm1 = nn.BatchNorm2d(1)
        self.Linear1 = nn.Linear(128,50)
        self.Batchnorm2 = nn.BatchNorm2d(1)
        self.Drop = nn.Dropout(p=p)
        self.Relu = nn.ReLU()
        self.Batchnorm3 = nn.BatchNorm2d(1)
        self.Linear2 = nn.Linear(50,2)


    def forward(self, h,c):
        h,c=h.squeeze(0),c.squeeze(0)
        H = torch.empty((h.shape[0], self.pre_len, 128))
        for i in range(self.pre_len):
            h_t,c_t=self.lstmcell(h,(h,c)) #prediction
            H[:,i,:]=h_t
            h,c=h_t,c_t

        H = H.transpose(1, 2).unsqueeze(1)  # [batch,c,w,h]
        H = self.Batchnorm1(H).squeeze(1).transpose(1, 2)  # [batch,h,w],BatchNorm and then linear transformation are beneficial to update the weights in one direction
        Linear1_out = self.Linear1(H).transpose(1, 2).unsqueeze(1)
        Linear1_out = self.Batchnorm2(Linear1_out).squeeze(1).transpose(1, 2)  # [batch,h,w],Batchnorm is activated to prevent the gradient from disappearing.
        relu_out = self.Relu(Linear1_out).transpose(1, 2).unsqueeze(1) 
        relu_out = self.Batchnorm3(relu_out).squeeze(1).transpose(1, 2) 
        drop_out = self.Drop(relu_out)
        decoder_out = self.Linear2(drop_out)

        return decoder_out


