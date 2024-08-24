import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from tqdm import tqdm
import sys

# Set default encoding to utf-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Set up logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'strategy_log_{timestamp}.log'

# Create output directory
output_dir = f'algorithm_run_{timestamp}'
os.makedirs(output_dir, exist_ok=True)

# Configure logging
logging.basicConfig(filename=os.path.join(output_dir, log_filename),
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# File paths (update these paths to your local paths)
base_path = r'C:\Users\xucha\OneDrive\桌面\Quasi-QE\实验一\数据'
quasi_qe_index_path = os.path.join(base_path, 'quasi_index', 'quasi_qe_index.csv')
tech_index_path = os.path.join(base_path, '科技股指数', 'tech_index.csv')

company_files = {
    'AAPL': os.path.join(base_path, 'company_data', 'AAPL.csv'),
    'MSFT': os.path.join(base_path, 'company_data', 'MSFT.csv'),
    'AMZN': os.path.join(base_path, 'company_data', 'AMZN.csv'),
    'GOOGL': os.path.join(base_path, 'company_data', 'GOOGL.csv'),
    'META': os.path.join(base_path, 'company_data', 'META.csv'),
    'NVDA': os.path.join(base_path, 'company_data', 'NVDA.csv'),
    'TSLA': os.path.join(base_path, 'company_data', 'TSLA.csv')
}

# Define company weights
company_weights = {
    'AAPL': 0.104980,
    'MSFT': 0.316339,
    'AMZN': 0.241482,
    'GOOGL': 0.137142,
    'META': 0.066073,
    'NVDA': 0.104360,
    'TSLA': 0.029626
}

# Initial funds (1,000,000 USD)
initial_funds = 1000000  # 1,000,000 USD

# Read quasi-QE index data
quasi_qe_index = pd.read_csv(quasi_qe_index_path, parse_dates=['DATE'], index_col='DATE')
# Ensure timezone is set
if quasi_qe_index.index.tz is None:
    quasi_qe_index.index = quasi_qe_index.index.tz_localize('UTC')
else:
    quasi_qe_index.index = quasi_qe_index.index.tz_convert('UTC')

# Read tech index data and parse dates
tech_index = pd.read_csv(tech_index_path, parse_dates=['Date'], index_col='Date')
# Ensure timezone is set
if tech_index.index.tz is None:
    tech_index.index = tech_index.index.tz_localize('UTC')
else:
    tech_index.index = tech_index.index.tz_convert('UTC')

# Resample quasi-QE index data to a five-minute frequency
quasi_qe_index_5min = quasi_qe_index.resample('5min').ffill()

# Initialize an empty DataFrame to store merged stock data
all_stock_data = pd.DataFrame()

# Read and merge stock data for each company
print("Loading company data...")
for company, file_path in tqdm(company_files.items(), desc="Loading data", unit="file"):
    # Read date column and parse dates
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')

    # Ensure timezone is set
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC')
    else:
        data.index = data.index.tz_convert('UTC')

    # Calculate daily returns
    data['return'] = data['close'].pct_change()

    # Detect splits: if any day's price change exceeds 49%
    data['is_split'] = data['return'].abs() > 0.49

    # Calculate split ratios and adjust prices
    split_factors = []
    for idx in range(1, len(data)):
        if data['is_split'].iloc[idx]:
            # Calculate split ratio, rounding to the nearest integer
            split_factor = data['close'].iloc[idx - 1] / data['close'].iloc[idx]
            split_factor = round(split_factor)
            split_factors.append((idx, split_factor))

            # Adjust prices day by day up to the split date
            data.loc[data.index[:idx], 'close'] /= split_factor

    # Output detected split events
    for idx, split_factor in split_factors:
        date = data.index[idx]
        print(f"Detected split for {company} on {date} with split ratio approximately {split_factor}:1")

    # Keep adjusted close price and split flag, renaming with the company code
    data = data[['close', 'is_split']].rename(columns={'close': company, 'is_split': f'{company}_is_split'})

    # Merge data
    if all_stock_data.empty:
        all_stock_data = data
    else:
        all_stock_data = all_stock_data.join(data, how='outer')

# Merge all data
all_data = all_stock_data.join(quasi_qe_index_5min).join(tech_index)

# Check for missing values and fill
if all_data.isnull().any().any():
    print("Warning: Missing values detected. Using forward fill to fill missing values.")
    all_data.ffill(inplace=True)

# Filter data for the backtest period (2020-03-03 to 2022-03-16)
backtest_data = all_data.loc['2020-03-03':'2022-03-16'].copy()

# Calculate momentum indicators
momentum_window = 10
backtest_data['Momentum'] = backtest_data['Tech_Index'].diff(momentum_window)

# Calculate volatility indicators (e.g., standard deviation)
volatility_window = 10
backtest_data['Volatility'] = backtest_data['QUASI_QE_INDEX'].rolling(window=volatility_window).std()

# Check for NaN values in Momentum
print("Checking for NaN values in Momentum:")
print(backtest_data[['Momentum']].isna().sum())

# Output statistics for key indicators
print("\nQUASI_QE_INDEX Statistics:")
print(backtest_data['QUASI_QE_INDEX'].describe())

print("\nMomentum Statistics:")
print(backtest_data['Momentum'].describe())

print("\nVolatility Statistics:")
print(backtest_data['Volatility'].describe())

# Define function to calculate maximum drawdown
def calculate_max_drawdown(values):
    max_drawdown = 0
    peak = values[0]
    for value in values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown

# Execute strategy using the best parameter combination
best_buy_threshold = 0.915  # Loosen condition
best_sell_threshold = 0.930  # Loosen condition

# Generate backtest trading signals
backtest_data['Signal'] = 0

# Adjust buy condition using momentum and QUASI_QE_INDEX to confirm signals
buy_signals = (
    (backtest_data['QUASI_QE_INDEX'] < best_buy_threshold) &
    (backtest_data['Momentum'] > 0) &
    (backtest_data['Volatility'] > backtest_data['Volatility'].mean())
)

sell_signals = (
    (backtest_data['QUASI_QE_INDEX'] > best_sell_threshold) &
    (backtest_data['Momentum'] < 0) &
    (backtest_data['Volatility'] > backtest_data['Volatility'].mean())
)

# Initialize transaction log
transaction_log = []

# Initialize portfolio cash and holdings
cash = initial_funds
shares = {company: 0 for company in company_files.keys()}
portfolio_values = []

# Flag to track position status
in_position = False

# Add signal timing constraints
last_buy_signal_time = None
last_sell_signal_time = None
signal_time_gap = pd.Timedelta(minutes=15)  # Set signal time interval

# Execute trading logic and calculate daily portfolio value
for index, row in backtest_data.iterrows():
    daily_log = {
        'Date': index,
        'Action': 'Hold',
        'Portfolio Value': cash,
    }

    # If buy signal and not currently holding
    if buy_signals.loc[index] and not in_position:
        current_time = index

        if last_buy_signal_time is None or (current_time - last_buy_signal_time) > signal_time_gap:
            daily_log['Action'] = 'Buy'
            backtest_data.at[index, 'Signal'] = 1  # Mark buy signal
            in_position = True  # Update position status
            last_buy_signal_time = current_time  # Update buy signal time

            print(f"Buying at {index}")

            # Execute buy operation
            for company in company_files.keys():
                # Calculate funds allocated to each stock based on weights
                allocation = cash * company_weights[company]
                price = row[company]
                if price > 0:
                    shares_to_buy = allocation // price
                    shares[company] += shares_to_buy
                    cash -= shares_to_buy * price
                    daily_log[f'{company} Buy'] = shares_to_buy
                    daily_log[f'{company} Price'] = price

    # If sell signal and currently holding
    elif sell_signals.loc[index] and in_position:
        current_time = index

        if last_sell_signal_time is None or (current_time - last_sell_signal_time) > signal_time_gap:
            daily_log['Action'] = 'Sell'
            backtest_data.at[index, 'Signal'] = -1  # Mark sell signal
            in_position = False  # Update position status
            last_sell_signal_time = current_time  # Update sell signal time

            print(f"Selling at {index}")

            # Execute sell operation
            for company in company_files.keys():
                if shares[company] > 0:
                    price = row[company]
                    cash += shares[company] * price
                    daily_log[f'{company} Sell'] = shares[company]
                    daily_log[f'{company} Price'] = price
                    shares[company] = 0

    # Update current portfolio value
    current_value = cash
    for company in company_files.keys():
        current_value += shares[company] * row[company]
        daily_log[f'{company} Holdings'] = shares[company]

    daily_log['Portfolio Value'] = current_value
    portfolio_values.append(current_value)
    transaction_log.append(daily_log)

# Calculate final portfolio value
final_portfolio_value = cash
for company in company_files.keys():
    final_portfolio_value += shares[company] * backtest_data[company].iloc[-1]

# Calculate annualized return based on actual date range
# Get start and end dates
start_date = backtest_data.index[0]
end_date = backtest_data.index[-1]

# Calculate total days
total_days = (end_date - start_date).days

# Convert days to years
years = total_days / 365.25  # Use 365.25 to account for leap years

# Calculate annualized return
annual_return = (final_portfolio_value / initial_funds) ** (1 / years) - 1

# Calculate maximum drawdown
max_drawdown = calculate_max_drawdown(portfolio_values)

# Output results
print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
print(f"Annualized Return: {annual_return:.2%}")
print(f"Maximum Drawdown: {max_drawdown:.2%}")

# Save transaction log to CSV
transaction_df = pd.DataFrame(transaction_log)
transaction_df.to_csv(os.path.join(output_dir, 'transaction_log.csv'), index=False)

# Save strategy parameters
with open(os.path.join(output_dir, 'strategy_params.txt'), 'w') as f:
    f.write(f"Buy Threshold: {best_buy_threshold}\n")
    f.write(f"Sell Threshold: {best_sell_threshold}\n")
    f.write(f"Initial Funds: {initial_funds}\n")
    f.write(f"Final Portfolio Value: {final_portfolio_value}\n")
    f.write(f"Annualized Return: {annual_return}\n")
    f.write(f"Maximum Drawdown: {max_drawdown}\n")

logging.info("Strategy execution complete. Results saved to %s", output_dir)
