import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Get current timestamp
    output_dir = f"{script_name}_{timestamp}"  # Generate directory name
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it does not exist
    return output_dir

# Calculate rolling volatility
def calculate_rolling_volatility(prices, window=48):
    """
    Calculate rolling window volatility.
    window: Rolling window size, default is 48 (48 five-minute intervals).
    """
    adjusted_prices = np.where(prices <= 0, np.nan, prices)
    log_returns = np.log(adjusted_prices[1:] / adjusted_prices[:-1])  # Calculate log returns
    log_returns = np.nan_to_num(log_returns, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN with 0
    volatility = pd.Series(log_returns).rolling(window=window, min_periods=1).std().bfill().values  # Calculate rolling standard deviation
    volatility = np.insert(volatility, 0, volatility[0])  # Insert initial value to match length
    return volatility

# Calculate rolling average volume
def calculate_rolling_liquidity(volumes, window=48):
    """
    Calculate rolling average volume.
    window: Rolling window size, default is 48.
    """
    liquidity = pd.Series(volumes).rolling(window=window, min_periods=1).mean().bfill().values  # Calculate rolling average
    liquidity = np.insert(liquidity, 0, liquidity[0])  # Insert initial value to match length
    return liquidity

# Calculate Quasi-QE Index
def calculate_quasi_qe_index(fed_rate, closing_prices, volumes, market_caps):
    """
    Calculate weighted Quasi-QE Index based on federal funds rate, closing prices, and volumes.
    """
    # Normalize data
    fed_rate_normalized = normalize_data(fed_rate)
    print(f"Fed Rate Normalized Min/Max: {fed_rate_normalized.min()}/{fed_rate_normalized.max()}")

    # Calculate weighted volatility and liquidity for each company
    weighted_volatility = np.zeros(closing_prices.shape[1])
    weighted_liquidity = np.zeros(volumes.shape[1])

    for i in range(closing_prices.shape[0]):
        # Calculate volatility and liquidity
        volatility = calculate_rolling_volatility(closing_prices[i], window=48)
        liquidity = calculate_rolling_liquidity(volumes[i], window=48)

        # Ensure calculated lengths match original data
        volatility = volatility[:closing_prices.shape[1]]
        liquidity = liquidity[:volumes.shape[1]]

        # Normalize
        volatility_normalized = normalize_data(volatility)
        liquidity_normalized = normalize_data(liquidity)

        # Apply weights
        weighted_volatility += volatility_normalized * market_caps[i]
        weighted_liquidity += liquidity_normalized * market_caps[i]

    # Normalize weighted volatility and liquidity
    weighted_volatility_normalized = normalize_data(weighted_volatility)
    weighted_liquidity_normalized = normalize_data(weighted_liquidity)

    # Print normalized data for debugging
    print(f"Weighted Volatility Normalized Min/Max: {weighted_volatility_normalized.min()}/{weighted_volatility_normalized.max()}")
    print(f"Weighted Liquidity Normalized Min/Max: {weighted_liquidity_normalized.min()}/{weighted_liquidity_normalized.max()}")

    # Define state space
    n_levels = 10  # Ten levels
    states = [f"Level {i}" for i in range(n_levels)]
    n_states = len(states)

    # Use KBinsDiscretizer to bin weighted volatility and liquidity
    discretizer = KBinsDiscretizer(n_bins=n_states, encode='ordinal', strategy='uniform', subsample=None)
    volatility_states = discretizer.fit_transform(weighted_volatility_normalized.reshape(-1, 1)).flatten()
    liquidity_states = discretizer.fit_transform(weighted_liquidity_normalized.reshape(-1, 1)).flatten()

    # Print binning results
    print(f"Volatility States: {volatility_states[:5]} ...")
    print(f"Liquidity States: {liquidity_states[:5]} ...")

    # Build Markov chain transition matrix
    transition_matrix = np.zeros((n_states, n_states))

    for i in range(len(volatility_states) - 1):
        current_state = int(volatility_states[i])
        next_state = int(volatility_states[i + 1])
        transition_matrix[current_state, next_state] += 1

    # Normalize transition matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.nan_to_num(transition_matrix)  # Replace NaN with zero

    # Print transition matrix
    print(f"Transition Matrix: \n{transition_matrix}")

    # Calculate Quasi-QE Index under each state
    quasi_qe_index = []

    # Calculate rolling window weighted volatility
    for i in range(len(fed_rate_normalized)):
        current_state = int(volatility_states[i])
        current_fed_rate = fed_rate_normalized[i]
        current_liquidity = weighted_liquidity_normalized[i]

        # Weight volatility by current state's transition probability
        state_weighted_volatility = np.sum(transition_matrix[current_state] * weighted_volatility_normalized[i:i + n_states])

        # Calculate index value
        index_value = 1 - (0.3 * current_fed_rate + 0.5 * state_weighted_volatility + 0.2 * (1 - current_liquidity))

        # Clip index value between 0 and 1
        index_value = np.clip(index_value, 0, 1)

        quasi_qe_index.append(index_value)

    # Check and return 1D array
    return np.array(quasi_qe_index).flatten()

# Normalize data to range 0 to 1
def normalize_data(data):
    """
    Scale data to range 0 to 1 using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(data.reshape(-1, 1)).flatten()

# Main function
def main():
    # Get script name to create output directory
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    # Create output directory
    output_dir = create_output_directory(script_name)
    log_filename = os.path.join(output_dir, 'training.log')
    global logger
    logger = setup_logging(log_filename)

    # Set data path and date range
    directory_path = 'C:/Users/xucha/OneDrive/桌面/Quasi-QE/实验一/数据/company_data/已处理数据/'
    fed_rate_path = 'C:/Users/xucha/OneDrive/桌面/Quasi-QE/实验一/数据/Federal_Funds_Rate_Data/federal_funds_rate_2020_2022_complete.csv'
    start_date = '2020-02-01'
    end_date = '2022-11-30'

    # Read federal funds rate data
    fed_rate_data = pd.read_csv(fed_rate_path)

    # Use correct column names
    fed_rate_data['DATE'] = pd.to_datetime(fed_rate_data['DATE'])
    fed_rate_data = fed_rate_data[(fed_rate_data['DATE'] >= start_date) & (fed_rate_data['DATE'] <= end_date)]
    fed_rate_values = fed_rate_data['FEDFUNDS'].values

    # Store all company data for merging
    all_closing_prices = []
    all_volumes = []

    # Predefined weights
    market_cap_weights = np.array([0.104980, 0.316339, 0.241482, 0.137142, 0.066073, 0.104360, 0.029626])

    # Select seven companies for calculation
    selected_companies = ['AAPL_processed.csv', 'MSFT_processed.csv', 'AMZN_processed.csv',
                          'GOOGL_processed.csv', 'META_processed.csv', 'NVDA_processed.csv',
                          'TSLA_processed.csv']

    for filename in os.listdir(directory_path):
        if filename in selected_companies:
            file_path = os.path.join(directory_path, filename)
            data = pd.read_csv(file_path)

            # Parse dates and restrict range
            data['date'] = pd.to_datetime(data['date'])
            data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

            close_price = data['close'].values
            volume = data['volume'].values

            # Store data for index calculation
            all_closing_prices.append(close_price)  # Use real data for index calculation
            all_volumes.append(volume)  # Use real data for index calculation

    # Convert to numpy arrays
    combined_closing_prices = np.array(all_closing_prices)
    combined_volumes = np.array(all_volumes)

    # Calculate weighted Quasi-QE Index
    quasi_qe_index = calculate_quasi_qe_index(fed_rate_values, combined_closing_prices, combined_volumes, market_cap_weights)

    # Print debugging information
    print(f"Quasi-QE Index Length: {len(quasi_qe_index)}, Shape: {quasi_qe_index.shape}")

    # Save Quasi-QE Index data
    quasi_qe_df = pd.DataFrame({
        'DATE': fed_rate_data['DATE'].values.flatten(),
        'QUASI_QE_INDEX': quasi_qe_index.flatten()
    })
    quasi_qe_df.to_csv(os.path.join(output_dir, 'quasi_qe_index.csv'), index=False)

    logger.info(f"Saved Quasi-QE index data to {output_dir}")

# Run main function
if __name__ == '__main__':
    main()
