import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Data_pre():
    """Build an LSTM neural network"""
    def __init__(self,type,look_back,pre_len):
        super(Data_pre, self).__init__()
        self.type=type
        self.look_back=look_back
        self.pre_len=pre_len

    def Get_data(self):
        if self.type=='train':
            ###Read the data and normalize all features except id.
            file = pd.read_csv(r'data\results.csv')
            data = file.drop(['frame'], axis=1) 
            data = data.loc[:,['id','x','y']]
            minMax = MinMaxScaler()
            data.loc[:, ['x']] = minMax.fit_transform(data.loc[:, ['x']])
            data.loc[:, ['y']] = minMax.fit_transform(data.loc[:, ['y']])
            x, y = self.creat_dataset(data,1,381,self.look_back,self.pre_len)  
            X = torch.from_numpy(x[:, :, 1:]) 
            Y = torch.from_numpy(y[:, :, 1:])
            return X,Y
        else:
            file = pd.read_csv(r'data\results.csv')
            data = file.drop(['frame'], axis=1) 
            data = data.loc[:, ['id', 'x', 'y']]
            minMax = MinMaxScaler()
            data.loc[:, ['x']] = minMax.fit_transform(data.loc[:, ['x']])
            data.loc[:, ['y']] = minMax.fit_transform(data.loc[:, ['y']])
            x, y = self.creat_dataset(data,382,580,self.look_back,self.pre_len)
            X = torch.from_numpy(x[:, :, 1:]) 
            Y = torch.from_numpy(y[:, :, 1:3]) 
            return X, Y

    def creat_dataset(self,data,start, end, look_back, pre_len):
        data_x = []
        data_y = []
        for id in range(start, end + 1):  # Select 70% for training
            data_id = data[data['id'] == id].values
            for i in range(len(data_id) - look_back - pre_len + 1):
                data_x.append(data_id[i:i + look_back, :])
                data_y.append(data_id[i + look_back:i + look_back + pre_len, :])
        return np.asarray(data_x), np.asarray(data_y)  # Convert to ndarray data
