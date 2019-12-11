import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(__file__))
from mLstm import mLSTM

embed_size = 128 # ascii representation
hidden_size = 2048

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.rnn = mLSTM(embed_size, hidden_size, embed_size, embed_size)
        if torch.cuda.is_available():
            self.rnn = self.rnn.cuda()
            self.rnn.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "../../data/models/lstm_11.pth")))
        else:
            self.rnn.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "../../data/models/lstm_11.pth")), map_location="cpu")
            self.rnn.cpu()
        
        self.fc1 = nn.Linear(2*hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/4))
        self.fc3 = nn.Linear(int(hidden_size/4), int(hidden_size/2))
        self.fc4 = nn.Linear(int(hidden_size/2), 5)
        self.dropout = nn.Dropout(0.5) 
        
    def forward(self, text_tensor):
        with torch.no_grad():
            hidden, cell = self.rnn.init_hidden()
            for p in range(len(text_tensor)):
                _, hidden, cell = self.rnn(text_tensor[p], hidden, cell)
        
        x = torch.cat((hidden, cell), 0).view(1, -1)
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        x = F.relu(self.dropout(self.fc3(x)))
        x = F.relu(self.fc4(x))
        return x