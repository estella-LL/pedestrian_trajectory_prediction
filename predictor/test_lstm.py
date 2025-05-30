####model-lstm
###[x,y,xv,yv,xa,ya]->[x,y]
import torch
import pandas as pd
from numpy import *
from model_lstm import LSTM
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# device = torch.device('cuda:0')
device = torch.device(device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
from data_pre import Data_pre

import numpy as np
import matplotlib.pyplot as plt

look_back=50 #Learn the past 2s (50 frames of data)
pre_len=50 #Predict the next 2s (50 frames of data)
batch=99
dropout=0 #Drop rate
loss=[] #Record the losses of multiple batches for subsequent MSE calculations
# 'Defining the loss function'
model=LSTM(dropout,pre_len) 
model.load_state_dict(torch.load('lstm_pre.pth'))
model.to(device)
loss_func=torch.nn.MSELoss()

'Get training data'
Data_pre=Data_pre('test',look_back,pre_len)
X,Y=Data_pre.Get_data()
X,Y=X[:,:,0:6].to(device),Y.to(device)
print(X.shape)

for t in range(int(len(X)/batch)):
    y_pre=model(X[t*batch:(t+1)*batch,:,:].float())
    y_label=Y[t*batch:(t+1)*batch,:,:].float()

    '''Attributes are scaled to a specified minimum and maximum value (usually 1-0). This can be achieved with the preprocessing.MinMaxScaler class.'''
    '''Commonly used minimum and maximum normalization methods (x-min(x))/(max(x)-min(x))'''
    y_pre[:,:,0]=y_pre[:,:,0]*(1884-13.5)+13.5
    y_pre[:,:,1]=y_pre[:,:,1]*(1058.5-4.5)+4.5
    y_label[:,:,0]=y_label[:,:,0]*(1884-13.5)+13.5
    y_label[:, :, 1] = y_label[:, :, 1] * (1058.5 - 4.5) + 4.5

    loss.append(loss_func(y_pre,y_label).item())

print('test RMSE:',torch.sqrt(torch.mean(torch.tensor(loss))))
print(y_pre[:,:,0])
print(y_pre[:,:,1])
print(y_label[:, :, 0])
print(y_label[:, :, 1])

##50: 0.0251
##100:0.0164
##150:0.0127
##200:0.0098(2.9904)























