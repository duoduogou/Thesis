import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import sys

# Set default encoding to utf-8
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
    """
    Create a directory for saving output results. The directory name consists of the script name and current timestamp.
    """
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
        Y.append(data[i + time_step])
    return np.array(X), np.array(Y)

# Define LSTM model for time series prediction
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1, num_layers=1):
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

# Train and evaluate the model using historical data
def train_and_evaluate(data, feature_name, time_step, company_name, hidden_layer_size, num_layers, lr, version, output_dir):
    logger.info(f"Processing {company_name} {feature_name} with LSTM version {version}")

    # Prepare dataset
    X, Y = create_dataset(data, time_step)

    # Split into training and testing sets (70:30 ratio)
    train_size = int(len(X) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]

    # Convert to PyTorch tensors and move to GPU if available
    X_train = torch.from_numpy(X_train).float().unsqueeze(-1).to(device)
    X_test = torch.from_numpy(X_test).float().unsqueeze(-1).to(device)
    Y_train = torch.from_numpy(Y_train).float().unsqueeze(-1).to(device)
    Y_test = torch.from_numpy(Y_test).float().unsqueeze(-1).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    # Reduce batch size to save memory
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Halve the batch size
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize LSTM model
    model = LSTMModel(input_size=1, hidden_layer_size=hidden_layer_size, num_layers=num_layers).to(device)
    loss_function = nn.MSELoss()  # Mean squared error loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam optimizer

    # Use torch.cuda.amp for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Train the model
    epochs = 50  # Number of training epochs
    patience = 20  # Set patience to 20 for early stopping
    min_delta = 1e-5  # Minimum improvement threshold
    best_val_loss = float('inf')
    no_improve_count = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # Use autocast for mixed precision
                y_pred = model(X_batch)
                loss = loss_function(y_pred, Y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                y_pred = model(X_batch)
                loss = loss_function(y_pred, Y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)

        # Log training and validation loss
        logger.info(f'{company_name} - {feature_name} - Version {version} - Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.5g}, Validation Loss: {val_loss:.5g}')

        # Check for early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                logger.info(f'Early stopping triggered for version {version} after {epoch+1} epochs.')
                break

    # Clear unused memory
    torch.cuda.empty_cache()

    # Calculate test set prediction error
    model.eval()
    test_predict = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            y_pred = model(X_batch)
            test_predict.extend(y_pred.cpu().numpy())

    test_predict = np.array(test_predict).flatten()  # Convert to 1D array

    # Calculate evaluation metrics
    train_rmse = np.sqrt(mean_squared_error(Y_train.cpu().numpy().flatten(), model(X_train).detach().cpu().numpy().flatten()))
    test_rmse = np.sqrt(mean_squared_error(Y_test.cpu().numpy().flatten(), test_predict))
    test_mae = mean_absolute_error(Y_test.cpu().numpy().flatten(), test_predict)
    test_r2 = r2_score(Y_test.cpu().numpy().flatten(), test_predict)

    # Log evaluation metrics
    logger.info(f"{company_name} - {feature_name} - Version {version} - Train RMSE: {train_rmse:.5g}")
    logger.info(f"{company_name} - {feature_name} - Version {version} - Test RMSE: {test_rmse:.5g}")
    logger.info(f"{company_name} - {feature_name} - Version {version} - Test MAE: {test_mae:.5g}")
    logger.info(f"{company_name} - {feature_name} - Version {version} - Test R2: {test_r2:.5g}")

    # Save model parameters
    torch.save(model.state_dict(), os.path.join(output_dir, f"{feature_name}_model.pth"))

    # Save loss values
    np.savetxt(os.path.join(output_dir, f"{feature_name}_train_losses.txt"), train_losses)
    np.savetxt(os.path.join(output_dir, f"{feature_name}_val_losses.txt"), val_losses)

    return train_losses, val_losses, train_rmse, test_rmse, test_mae, test_r2

# Main function
def main():
    # Get script name to create output directory
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    # Create output directory
    version = 1
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

    # Study stock data for one company only
    company = 'AAPL'  # Apple Inc.

    file_path = os.path.join(directory_path, f'{company}_processed.csv')
    if not os.path.exists(file_path):
        logger.error(f"File {file_path} does not exist. Exiting.")
        return

    data = pd.read_csv(file_path)

    # Parse dates and restrict range
    data['date'] = pd.to_datetime(data['date'])
    data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

    # Extract close prices and volumes
    close_price = data['close'].values
    volume = data['volume'].values

    # Standardize data
    scaler = StandardScaler()
    scaled_close_price = scaler.fit_transform(close_price.reshape(-1, 1)).flatten()
    scaled_volume = scaler.fit_transform(volume.reshape(-1, 1)).flatten()

    # Store experimental results
    results = []

    # Base version
    for feature_name, scaled_data in [('Close Price', scaled_close_price), ('Volume', scaled_volume)]:
        train_losses, val_losses, train_rmse, test_rmse, test_mae, test_r2 = train_and_evaluate(
            scaled_data, feature_name, time_step=78, company_name=company, hidden_layer_size=32, num_layers=1, lr=0.001, version=version, output_dir=output_dir
        )
        results.append({
            'version': version,
            'feature_name': feature_name,
            'hidden_layer_size': 32,
            'num_layers': 1,
            'learning_rate': 0.001,
            'time_step': 78,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'train_losses': train_losses,
            'val_losses': val_losses
        })

    # Output results
    results_file = os.path.join(output_dir, 'results.txt')
    with open(results_file, 'w') as f:
        for result in results:
            result_str = (
                f"Version {result['version']} - {result['feature_name']}:\n"
                f"  Hidden Layer Size: {result['hidden_layer_size']}\n"
                f"  Num Layers: {result['num_layers']}\n"
                f"  Learning Rate: {result['learning_rate']}\n"
                f"  Time Step: {result['time_step']}\n"
                f"  Train RMSE: {result['train_rmse']:.5g}\n"
                f"  Test RMSE: {result['test_rmse']:.5g}\n"
                f"  Test MAE: {result['test_mae']:.5g}\n"
                f"  Test R2: {result['test_r2']:.5g}\n"
                "----------------------------------------\n"
            )
            print(result_str)
            f.write(result_str)

# Enable cuDNN acceleration
torch.backends.cudnn.benchmark = True

# Check device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run main function
if __name__ == '__main__':
    main()
