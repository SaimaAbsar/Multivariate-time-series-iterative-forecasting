import torch
import torch.nn as nn
import torch.optim as optim

# Define LSTM model
class LSTM(nn.Module):
    def __init__(self, in_size, hidden_size, n_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(in_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, in_size)   # Predicting all features
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out