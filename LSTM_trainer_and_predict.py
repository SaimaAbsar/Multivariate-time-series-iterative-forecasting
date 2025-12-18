# July 2024
# Author: Saima Absar

# LSTM model for predicting pressure-drop using causal-features
# This project was done as part of the internship at CPChem
# Training and future predictions

import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math
import matplotlib.dates as mdates
from pyspark.sql import functions as F
from pyspark.sql.functions import col
import datetime
from torchinfo import summary
import mlflow
from mlflow.tracking import MlflowClient

torch.manual_seed(1230)
np.random.seed(1230)

# Use GPU
device = torch.device("cuda")
# MLflow
mlflow.login()
mlflow.set_experiment("/mlflow-decoke-lstm")

# Create overlapping sequences for training
def create_seq(data, seq_len): 
    sequences = []
    labels = []
    for i in range(len(data) - seq_len): 
        s = data[i:i+seq_len]
        label = data[i+seq_len]  # All feature values at the next time step
        sequences.append(s)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Create Dataset classes
class TimeSeries(Dataset):
    def __init__(self, sequences, labels, device):
        self.seq = torch.tensor(sequences).to(device)
        self.labels = torch.tensor(labels).to(device)

    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, idx):
        return self.seq[idx], self.labels[idx]

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
      

# Funciton to split data, train the model, and test
def train_and_test(df, pass_, coil, in_window, train, test_ratio, epochs, hidden_size, n_layers, batch_size, lr):
    df_coil = df[df['coil'] == coil]
    
    # The data is not sorted chronologically
    # Thus sorted wrt time for each coil
    df_sorted = df_coil.sort_values(by='Timestamp').reset_index(drop=True)

    # Extract the timestamps for plotting
    timestamps = pd.to_datetime(df_sorted['Timestamp'], format="%Y-%m-%d %H:%M:%S")

    # Drop all columns other than the feature values
    df_sorted = df_sorted.drop(['coil', 'Timestamp', 'pass'], axis=1)
    #print(df_coil.head())

    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_sorted.values)
    #print(scaled_data[:10])

    # Split
    # Setting 80 percent data for training
    split_idx = int(len(scaled_data) * (1-test_ratio))
    #print(split_idx)
    #Splitting the dataset
    train_data = scaled_data[:split_idx]
    test_data = scaled_data[split_idx - in_window:]

    # convert to dataloder
    train_seq, labels_tr = create_seq(train_data, in_window)
    test_seq, labels_test = create_seq(test_data, in_window)
    # Dataloader
    train_dataset = TimeSeries(train_seq, labels_tr, device)  
    test_dataset = TimeSeries(test_seq, labels_test, device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with mlflow.start_run() as run:
        # Hyperparameters
        in_size = train_seq.shape[2]
        # Define model and loss
        model = LSTM(in_size, hidden_size, n_layers).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.9)

        # parameters for logging
        params = {
            "pass": pass_,
            "coil": coil,
            "in_window": in_window,
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "n_layers": n_layers,
            "loss_function": criterion,
            "metric_function": nn.MSELoss(),
            "optimizer": optimizer,
        }
        # Log training parameters.
        mlflow.log_params(params)
        # Log model summary.
        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")
        
        # If training is enabled, train the model
        # Else load the saved model and only predict
        if train:
            # Train
            model.train()
            Loss = []
            for e in range(epochs):
                L = []
                for seq_tensor, label_tensor in train_dataloader:
                    seq_tensor, label_tensor = seq_tensor.to(torch.float32), label_tensor.to(torch.float32)
                    outs = model(seq_tensor).squeeze()
                    optimizer.zero_grad()
                    loss = criterion(outs, label_tensor)
                    loss.backward()
                    optimizer.step()
                    L.append(loss.item())
                if e%10 == 0:
                    print(f'Epoch [{e+1}/{epochs}], Loss: {np.mean(L):.6f}')
                    mlflow.log_metric("loss", f"{loss.item():2f}", step=e+1)
                scheduler.step()
                Loss.append(np.mean(L))
            # Save the trained model
            torch.save(model.state_dict(), f'models/model_{pass_}_{coil}.pt')    
            # Plotting the training loss
            #plt.plot(Loss)
            #plt.ylabel("Training Loss")
            #plt.show()

            # Save the trained model to MLflow.
            mlflow.pytorch.log_model(model, f'model_{pass_}_{coil}')

        else:
            model.load_state_dict(torch.load(f'./models/model_{pass_}_{coil}.pt'))

        # Validation on test data 
        model.eval()
        predictions = []
        L_test = []
        with torch.no_grad():
            total_test_loss = 0.0

            for batch_X_test, batch_y_test in test_dataloader:
                batch_X_test, batch_y_test = batch_X_test.to(torch.float32), batch_y_test.to(torch.float32)
                outputs = model(batch_X_test)#.squeeze()
                predictions.append(outputs.cpu().numpy())
                test_loss = criterion(outputs, batch_y_test)
                total_test_loss += test_loss.item()
                L_test.append(test_loss.item())

            # Calculate average test loss and accuracy
            average_test_loss = total_test_loss / len(test_dataloader)
        print('Error of prediction on test data: ', average_test_loss)
        mlflow.log_metric("eval_loss", f"{average_test_loss:2f}", step=epochs)
        # Plotting the test loss
        #plt.plot(L_test)
        #plt.ylabel("Test MSE (Batch-wise)")
        #plt.show()

    predictions = np.concatenate(predictions, axis = 0)

    # Inverse transform the predictions and test labels to original scale
    predictions_org = scaler.inverse_transform(predictions)
    test_labels_org = scaler.inverse_transform(labels_test) 

    # Extract the last sequence from the test data
    last_sequence = test_data[-in_window:]

    return model, scaler, last_sequence, split_idx, timestamps, predictions_org, test_labels_org


# Make future predictions based on trained model
def predict_future(model, scaler, last_sequence, future_steps, in_window):
  model.eval()
  predictions = []
  last_sequence = last_sequence.copy() #take the last sequence from the data
  #print(last_sequence.shape)

  with torch.no_grad():
    for _ in range(future_steps):
      last_sequence_tensor = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)
      #print(last_sequence_tensor.shape)
      prediction = model(last_sequence_tensor).squeeze().cpu().numpy()
      #print(prediction.shape)
      predictions.append(prediction)

      # Update the last sequence by appending the predicted values and shifting the sequence
      last_sequence = np.vstack((last_sequence[1:], prediction))
  predictions = np.array(predictions)
  predictions_org = scaler.inverse_transform(predictions)

  return predictions_org


def main(path, in_window, features, target, future_steps, train, outfile):

    df = pd.read_csv(path)
    #features = ['Timestamp', 'coil', 'pass']+features
    df_with_time = df[['Timestamp', 'coil', 'pass'] + features + [target]]
    df_with_time = df_with_time.dropna()
    print(df_with_time.shape)

    # open output file to save predictions
    with open(outfile, "w") as f:
        f.write("Pass,Coil,Timestamp,Prediction\n")

    # Predict for each pass and each coil
    for pass_ in df['pass'].unique():
        print(pass_)
        # Select specific pass
        df_pass = df_with_time[(df_with_time['pass'] == pass_)]

        for coil in df_pass['coil'].unique():
            print(coil)

            model, scaler, last_sequence, split_idx, timestamps, predictions_org, test_labels_org = train_and_test(df_pass, pass_, coil, in_window, train, test_ratio=0.2, epochs=200, hidden_size=32, n_layers=3, batch_size=128, lr=1e-3)
            
            k = 10 # the number of time-steps to plot
            # Extract datetime of last window
            last_timestamps = timestamps.iloc[-in_window*k:].reset_index(drop=True)
            #print("last: ", last_timestamps)

            future_predictions = predict_future(model, scaler, last_sequence, future_steps, in_window)

            # Extract last seq timestamp
            last_sequence_time = last_timestamps.iloc[-1] 
            #print(last_sequence_time)

            # Generate future timestamp
            future_timestamps = pd.date_range(start=last_sequence_time, periods=future_steps+1, freq=pd.infer_freq(last_timestamps)).tolist()[1:]
            #print("future: ", future_timestamps)
            #print('last time: ', len(last_timestamps))
            #print('labels: ', len(test_labels_org[-in_window:, -1]))

            # Append predictions to the CSV file
            with open(outfile, "a") as f:
                for times, pred in zip(future_timestamps, future_predictions[:,-1]):
                    f.write(f'{pass_},{coil},{times},{pred}\n')

            # Plot
            plt.figure(figsize=(12, 6))
            plt.scatter(last_timestamps, test_labels_org[-in_window*k:, -1], label='Actual (Test Data)')
            plt.scatter(last_timestamps, predictions_org[-in_window*k:, -1], label='Predicted (Test Data)')
            plt.scatter(future_timestamps, future_predictions[:,-1], label='Unseen future predictions', marker='x')

            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
            plt.gcf().autofmt_xdate(rotation=45)

            plt.xlabel('Date')
            plt.ylabel('Adjusted Pressure_drop')
            plt.legend()
            plt.title(f'LSTM Predictions for {pass_}: {coil}')

            plt.grid(True)
            plt.show()

    # Final output message
    print(f'Future predictions for each coil saved to {outfile}.')
  
# main function to read data from user and train the model
if __name__=='__main__':
    # update your potential causal features and targets
    features = ""
    target = ''

    # Need to provide the input data path and
    parser = argparse.ArgumentParser("input info")
    parser.add_argument("--input_path", type=str, help="path to the input data")
    parser.add_argument("--in_window", type=int, help="input window size", default=10)
    parser.add_argument("--target", type=str, help="target feature to predict", default='pressure_drop_adjusted')
    parser.add_argument("--future_steps", type=int, help="number of steps to predict", default=50)
    parser.add_argument("--train", type=int, help="whether to train or only predict based on saved model", default=1)
    parser.add_argument("--in_features", type=str, help="causal features", default=features) 
    parser.add_argument("--out_file", type=str, help="output file name", default='lstm_predictions.csv') 

    args = parser.parse_args()
    path = args.input_path
    in_window = args.in_window
    future_steps = args.future_steps
    train = args.train
    features = args.in_features.split()
    target = args.target
    outfile = args.out_file

    # path = '/dbfs/mnt/edp-sandbox/DECOKE/SAIMA/DataH104_only2024.csv'

    # main(path, in_window=5, features=features.split(), target=target, future_steps=50, train=0, outfile='lstm_predictions.csv')

    main(path, in_window, features, target, future_steps, train, outfile)

    # Fetch the associated conda environment
    env = mlflow.pytorch.get_default_conda_env()
    print(f"conda env: {env}")

    #mlflow.pytorch.get_default_pip_requirements()

    