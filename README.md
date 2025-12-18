# Multivariate-time-series-iterative-forecasting
Multivariate time series forecasting with LSTM using iterative predictions

**Implementation**:

**Usage (Prediction model)**:
>   The LSTM model, trainer, and prediction codes are available in LSTM_trainer_and_predict.py. It can be run by using the notebook run_LSTM_predictions. This model saves the trained models for each coil, and can use those trained models for prediction. The ‘train’ variable decides whether to train or use the saved models for prediction.

> -   Example run: 

      %run -i "LSTM_trainer_and_predict.py" --input_path '---' --in_window 24 --future_steps 50 --train 1 --target 'pressure_drop' --out_file 'lstm_predictions.csv'--in_features "___"

> -   This notebook reads the user input and use them to run the "LSTM_trainer_and_predict.py" The user inputs include:\
  --input_path: path to the input data"\
  --in_window: input window size"\
  --in_features: causal features\
  --target: target feature to predict\
  --future_steps: number of steps to predict\
  --train: whether to train or only predict based on saved model\
  --out_file: output .csv file to save the prediction results with 4 columns = ['Timestamp', 'Pass', 'Coil', Predictions]

> -   Recommended hyperparameters:\
>     in_window=24, test_ratio=0.2, epochs=200, hidden_size=32, n_layers=3, batch_size=128, lr=1e-3,
      in_features = "venturi_outlet_pressure coil_venturi_ratio cracking_days feed_C2H6"

