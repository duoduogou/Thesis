import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import logging

# Set default encoding to utf-8
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Configure logging output to file and console
def setup_logging(log_filename):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_filename, encoding='utf-8'), logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    return logger

# Create output directory
def create_output_directory(script_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Get current timestamp
    output_dir = f"{script_name}_{timestamp}"  # Generate directory name
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it does not exist
    return output_dir

# Create dataset for LSTM model
def create_dataset(data, time_step=1):
    """
    Generate input and output datasets with a time step of `time_step`.
    """
    X, Y = [], []
    for i in range(len(data) - time_step):
        a = data[i:(i + time_step)]
        X.append(a)
        Y.append(data[i + time_step])  # Keep shapes consistent
    return np.array(X), np.array(Y)

# Define LSTM model for time series prediction
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_layer_size=32, output_size=2, num_layers=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True)  # LSTM layer
        self.linear = nn.Linear(hidden_layer_size, output_size)  # Linear layer
        self.dropout = nn.Dropout(0.2)  # Dropout for regularization

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)  # LSTM output
        lstm_out = self.dropout(lstm_out)  # Apply dropout
        predictions = self.linear(lstm_out[:, -1])  # Prediction
        return predictions

# Check and log device (GPU or CPU)
def check_device():
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("GPU is not available, using CPU.")

# Train and predict using historical data
def train_and_predict(data, time_step, company_name):
    logger.info(f"Processing {company_name} for weight calculation")

    # Prepare dataset
    X, Y = create_dataset(data, time_step)

    # Split into training and validation sets (70:30 ratio)
    train_size = int(len(X) * 0.7)
    X_train, X_val = X[:train_size], X[train_size:]
    Y_train, Y_val = Y[:train_size], Y[train_size:]

    # Convert to PyTorch tensors and move to GPU if available
    X_train = torch.from_numpy(X_train).float().to(device)
    X_val = torch.from_numpy(X_val).float().to(device)
    Y_train = torch.from_numpy(Y_train).float().to(device)
    Y_val = torch.from_numpy(Y_val).float().to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Batch size 32
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize LSTM model
    model = LSTMModel(input_size=2, hidden_layer_size=32, output_size=2, num_layers=1).to(device)
    loss_function = nn.MSELoss()  # Mean squared error loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

    # Train the model
    epochs = 50  # Number of training epochs
    patience = 20  # Early stopping patience
    min_delta = 1e-4  # Minimum improvement
    best_val_loss = float('inf')
    no_improve_count = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_function(y_pred, Y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                y_pred = model(X_batch)
                loss = loss_function(y_pred, Y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        logger.info(f'{company_name} - Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Check early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                logger.info(f'Early stopping triggered after {epoch+1} epochs.')
                break

    # Make predictions
    model.eval()
    with torch.no_grad():
        final_pred = model(torch.from_numpy(data).float().to(device).unsqueeze(0))

    return final_pred.cpu().numpy()

# Normalize data to range [0, 1]
def normalize_data(data):
    """
    Normalize data to the range [0, 1].
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(data.reshape(-1, 1))
    return normalized_data.flatten(), scaler

# Calculate weights
def calculate_weights(company_list, close_prices, volumes, predicted_prices):
    price_changes = [pred[-1, 0] for pred in predicted_prices]  # Price changes
    volume_changes = [pred[-1, 1] for pred in predicted_prices]  # Volume changes

    # Calculate weights: weighted average of price and volume changes
    weights = np.maximum(0, np.array(price_changes) * np.array(volume_changes))

    # Ensure all weights are positive and normalize
    total_weight = np.sum(weights)
    weights = weights / total_weight if total_weight != 0 else np.zeros_like(weights)

    # Ensure each weight is at least 0.03 and at most 0.8
    min_weight = 0.03
    max_weight = 0.8
    weights = np.clip(weights, min_weight, max_weight)

    # Renormalize
    weights = weights / np.sum(weights)

    return weights

# Main function
def main():
    # Get script name to create output directory
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    # Create output directory
    output_dir = create_output_directory(script_name)
    log_filename = os.path.join(output_dir, 'training.log')
    global logger
    logger = setup_logging(log_filename)

    # Check device availability
    check_device()

    # Set data path and date range
    directory_path = 'C:/Users/xucha/OneDrive/桌面/Quasi-QE/实验一/数据/company_data/已处理数据/'
    start_date = '2020-02-01'
    end_date = '2022-11-30'

    # Define company list and stock symbols
    company_list = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA']

    close_prices = []
    volumes = []
    predicted_prices = []

    for company in company_list:
        file_path = os.path.join(directory_path, f'{company}_processed.csv')
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} does not exist. Skipping {company}.")
            continue

        data = pd.read_csv(file_path)

        # Parse dates and restrict to range
        data['date'] = pd.to_datetime(data['date'])
        data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

        close_price = data['close'].values
        volume = data['volume'].values

        # Normalize data
        close_price, _ = normalize_data(close_price)
        volume, _ = normalize_data(volume)

        # Combine price and volume data
        combined_data = np.column_stack((close_price, volume))

        time_step = 78  # Time step

        # Train and predict
        test_pred = train_and_predict(combined_data, time_step, company)
        predicted_prices.append(test_pred)

        close_prices.append(close_price)
        volumes.append(volume)

    # Calculate weights
    weights = calculate_weights(company_list, close_prices, volumes, predicted_prices)

    # Print weights
    weight_df = pd.DataFrame({'Company': company_list, 'Weight': weights})
    print("Technology Stock Weights:")
    print(weight_df)

# Check device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run main function
if __name__ == '__main__':
    main()
