import torch
from torch.utils.data import Dataset

# Create Dataset classes
class TimeSeries(Dataset):
    def __init__(self, sequences, labels, device):
        self.seq = torch.tensor(sequences).to(device)
        self.labels = torch.tensor(labels).to(device)

    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, idx):
        return self.seq[idx], self.labels[idx]