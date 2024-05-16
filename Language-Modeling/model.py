import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_length = 62
        self.hidden_length = 512
        self.output_length = 62
        self.n_layers = 4
        self.dropout = 0.1
        
        self.embedding = nn.Embedding(self.input_length, self.hidden_length)
        self.rnn = nn.RNN(input_size=self.hidden_length, 
                          hidden_size=self.hidden_length, 
                          num_layers=self.n_layers, 
                          dropout=self.dropout, 
                          batch_first=True)
        self.fc = nn.Linear(self.hidden_length, self.output_length)
        
    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        initial_hidden = torch.zeros(self.n_layers, batch_size, self.hidden_length)
        return initial_hidden

class CharLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_length = 62
        self.hidden_length = 512
        self.output_length = 62
        self.n_layers = 4
        self.dropout = 0.5
        
        self.embedding = nn.Embedding(self.input_length, self.hidden_length)
        self.lstm = nn.LSTM(input_size=self.hidden_length, 
                          hidden_size=self.hidden_length, 
                          num_layers=self.n_layers, 
                          dropout=self.dropout, 
                          batch_first=True)
        self.fc = nn.Linear(self.hidden_length, self.output_length)
        
    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, (hidden, hidden))
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        initial_hidden = torch.zeros(self.n_layers, batch_size, self.hidden_length)
        return initial_hidden
