####model-lstm
###[x,y,xv,yv,xa,ya]->[x,y]
import torch
import os
import time
from model_lstm import LSTM
from sklearn.preprocessing import MinMaxScaler
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from data_pre import Data_pre

def Showtime(T1,T2,t,i,epoch):
    ##When batch_size is 110, 1006 iters are required to complete one epoch
    T=(T2-T1) #The time (in seconds) used by the last iteration
    T = ((epoch - i - 1) * 720 + (719 - t)) * T  # time left

lr=0.001
look_back=50 #Learn the past 2s (50 frames of data)
pre_len=50 #Predict the next 2s (50 frames of data)
batch=110
dropout=0.2 #Drop rate
'Define optimizer and loss function'
model = LSTM(dropout,pre_len)
#Digital switch card number
model.to(device)
optimizer=torch.optim.Adam(model.parameters(), lr)
loss_func=torch.nn.MSELoss()

'Get training data'
Data_pre=Data_pre('train',look_back,pre_len)
X,Y=Data_pre.Get_data()
X,Y=X[:,:,0:2].to(device),Y.to(device)
print(X.shape)

'Conduct training'
for i in range(50):
    for t in range(int(len(X)/batch)):

        T1 = time.process_time()  # Start timing

        y_pre= model(X[t*batch:(t+1)*batch,:,:].float())
        y_label=Y[t*batch:(t+1)*batch,:,:].float()
        loss = loss_func(y_pre,y_label)
        optimizer.zero_grad()
        loss.backward()

        if (i % 10 == 0) & (t == 0):  # Output the gradient every 10 rounds
            for name, param in model.named_parameters():
                print('name:{},params.grad{}'.format(name, param.grad))
        optimizer.step()
        print('Epoch:',i+1,'\t','iters:','%.5f'%((t/(len(X)/batch))*100),'%','\t','loss:','%.5f'%loss.item())

        T2 = time.process_time()  # End timing
        Showtime(T1, T2, t, i, 50)

torch.save(model.state_dict(),'lstm_pre.pth')


