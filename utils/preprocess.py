# This notebook contains functions to preprocess the input time series to create overlapping windows, normalize them, and convert them to batches with torch Dataloader for training.

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
import torch
from torch.utils.data import Dataset, DataLoader
import timeseries

class PREPROCESS_DATA():
  def __init__(self, data, in_window=20):
    self.data = data
    self.in_window = in_window

  # Create overlapping sequences for training
  def create_seq(self, data, seq_len): #, targ, out_size):
    sequences = []
    labels = []
    for i in range(len(data) - seq_len): #- out_size):
        s = data[i:i+seq_len]
        label = data[i+seq_len]  # All feature values at the next time step
        sequences.append(s)
        labels.append(label)
    return np.array(sequences), np.array(labels)

  # Normalize the data
  def normalize(self, df_selec):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_selec.values)
    return scaled_data, scaler
  
  def split(self, scaled_data, test_ratio):
    # Setting 80 percent data for training
    split_idx = int(len(scaled_data) * (1-test_ratio))
    print(split_idx)

    #Splitting the dataset
    train_data = scaled_data[:split_idx]
    test_data = scaled_data[split_idx - in_window:] # Include last sequence from training for consistency
    print(train_data.shape, test_data.shape)

    return train_data, test_data, split_idx
  
  
  # Convert scaled data to Dataframe
  def dataframe(self, train_data, test_data, batch_size = 128):
    #scaled_df = pd.DataFrame(scaled_train, columns=features+[target])

    sequences_tr, labels_tr = self.create_seq(train_data, self.in_window)
    print(sequences_tr.shape)
    sequences_test, labels_test = self.create_seq(test_data, self.in_window)
    print(labels_test.shape)

    # Dataloader
    train_dataset = timeseries.TimeSeries(sequences_tr, labels_tr)  
    test_dataset = timeseries.TimeSeries(sequences_test, labels_test)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return sequences_tr, labels_tr, sequences_test, labels_test, train_dataloader, test_dataloader