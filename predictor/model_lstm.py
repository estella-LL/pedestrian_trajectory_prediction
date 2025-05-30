import torch.nn as nn
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from predictor.encoder import Encoder
from predictor.decoder import Decoder


'LSTM structure'
class LSTM(nn.Module):
    """Building an LSTM neural network"""
    # def __init__(self,p=0.75,pre_len=5):
    def __init__(self, p=0.75, pre_len=50):
        super(LSTM, self).__init__()
        self.Encoder=Encoder()
        self.Decoder=Decoder(p,pre_len)
        self.p=p
    def forward(self, x):
        _,(h,c)=self.Encoder(x)
        de_out=self.Decoder(h,c)
        # print(de_out.shape)
        return de_out

x = torch.rand(1,10,2)
model=LSTM()
y = model(x)

y = y.detach().numpy().squeeze()
