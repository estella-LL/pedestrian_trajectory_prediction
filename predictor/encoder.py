import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

'Encoder structure'
class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=2, #2 input features
                            hidden_size=128, #The hidden state h is expanded to 64 dimensions
                            num_layers=1, #1-layer LSTM
                            batch_first=True, # The input structure is(batch_size, seq_len, feature_size). Default: False
                            )
    def forward(self, x):
        encoder_out,(encoder_h,encoder_c) = self.lstm(x, None) #Initially h and c default to 0
        return encoder_out,(encoder_h,encoder_c)