"""
Hourly Stock/Crypto Price Prediction using Current Market Data
Downloads recent hourly data from Yahoo Finance and predicts using LSTM

Assumptions made:
- Default ticker is BTC-USD (crypto has continuous hourly data).
- We download the last 90 days of hourly data (about 2,160 points). Yahoo may limit hourly history for long ranges.
- Lookback window is 24 (use previous 24 hours to predict next hour).
- We predict the next 24 hours.

If you want a different ticker or period, change the TICKER and `days_back` variables below.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Try to import TensorFlow; if it's not available, print a helpful message and exit
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
except Exception as e:
    venv_python = '/Users/ivanliu/Desktop/isehw/.venv/bin/python'
    msg = f"""
Required dependency missing: TensorFlow (import error: {e}).

Run this script with the project's virtualenv which already has the required packages:

  source /Users/ivanliu/Desktop/isehw/.venv/bin/activate
  python predstock_hourly.py

or without activation:
  {venv_python} /Users/ivanliu/Desktop/isehw/predstock_hourly.py

Exiting.
"""
    print(msg)
    raise SystemExit(1)

import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ---------------- USER CONFIG ----------------
TICKER = 'NNE'    # default to BTC hourly; change to any ticker (e.g., 'AAPL', 'ETH-USD')
days_back = 90        # how many days of hourly data to download (assumption)
lookback = 24         # use last 24 hours to predict next hour
predict_hours = 24    # predict next 24 hours
# ---------------------------------------------

print("=" * 80)
print(f"{TICKER} Hourly Price Prediction - Current Market Data")
print("=" * 80)

# STEP 1: Download hourly data
end_date = datetime.now()
start_date = end_date - timedelta(days=days_back)
print(f"\n[STEP 1] Downloading {TICKER} hourly data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

try:
    data_df = yf.download(TICKER, start=start_date, end=end_date, interval='1h', progress=False)
    if len(data_df) == 0:
        print(f"✗ Error: No hourly data found for ticker '{TICKER}' in the requested period")
        print("Try a different ticker or reduce `days_back` for hourly data.")
        raise SystemExit(1)
    # Try to get a nicer name
    try:
        ticker_info = yf.Ticker(TICKER)
        full_name = ticker_info.info.get('longName', TICKER)
    except Exception:
        full_name = TICKER
    print(f"✓ Successfully downloaded {len(data_df)} hourly rows of {full_name} data")
    print(f"\nData columns: {list(data_df.columns)}")
except Exception as e:
    print(f"✗ Error downloading hourly data: {e}")
    print("Make sure you have internet connection and yfinance installed!")
    raise SystemExit(1)

# Use the Close price
if 'Close' not in data_df.columns:
    print("✗ 'Close' column not found in downloaded data; available columns:", list(data_df.columns))
    raise SystemExit(1)

data = data_df[['Close']].values
print(f"\nTotal hourly points: {len(data)}")
print(f"Price range: ${data.min():.2f} - ${data.max():.2f}")
print(f"Most recent price: ${data[-1][0]:.2f}")

# STEP 2: Normalize
print("\n[STEP 2] Normalizing data to [0,1]...")
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)
print(f"✓ Data normalized. Min: {data_normalized.min():.4f}, Max: {data_normalized.max():.4f}")

# STEP 3: Split
print("\n[STEP 3] Splitting data into training/testing sets (80/20)")
train_size = int(len(data_normalized) * 0.8)
test_size = len(data_normalized) - train_size
train_data = data_normalized[:train_size]
test_data = data_normalized[train_size:]
print(f"✓ Training: {len(train_data)} samples; Testing: {len(test_data)} samples")

# STEP 4: Create sequences
print(f"\n[STEP 4] Creating sequences with lookback = {lookback} (hours)")

def create_sequences(data, lookback=24):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

x_train, y_train = create_sequences(train_data, lookback)
# include context for test
test_with_context = np.concatenate((train_data[-lookback:], test_data), axis=0)
x_test, y_test = create_sequences(test_with_context, lookback)

# reshape
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
print(f"✓ x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")

# STEP 5: Build & Train
print("\n[STEP 5] Building and training LSTM model (LSTM(4) -> Dense(1))")
model = Sequential()
model.add(LSTM(4, input_shape=(x_train.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
print(model.summary())

print("\nTraining (this may take a few minutes)...")
history = model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=1, validation_split=0.1)
print("✓ Training completed")

# STEP 6: Predictions
print("\n[STEP 6] Making predictions...")
train_pred = model.predict(x_train, verbose=0)
test_pred = model.predict(x_test, verbose=0)

train_pred = scaler.inverse_transform(train_pred)
y_train_act = scaler.inverse_transform(y_train.reshape(-1, 1))

test_pred = scaler.inverse_transform(test_pred)
y_test_act = scaler.inverse_transform(y_test.reshape(-1, 1))
print("✓ Predictions completed")

# Metrics
train_rmse = np.sqrt(mean_squared_error(y_train_act, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test_act, test_pred))
print(f"\nTraining RMSE: ${train_rmse:.2f}")
print(f"Testing RMSE: ${test_rmse:.2f}")

# STEP 7: Future hourly predictions
print(f"\n[STEP 7] Predicting next {predict_hours} hours...")
last_seq = data_normalized[-lookback:].reshape(1, lookback, 1)
future_preds = []
for i in range(predict_hours):
    nxt = model.predict(last_seq, verbose=0)
    future_preds.append(nxt[0, 0])
    last_seq = np.append(last_seq[0, 1:, :], [[nxt[0, 0]]], axis=0).reshape(1, lookback, 1)

future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

print(f"\nPredicted prices for next {predict_hours} hours:")
for i, p in enumerate(future_preds, 1):
    print(f"Hour +{i}: ${p[0]:,.2f}")

# STEP 8: Plots (hourly)
print("\n[STEP 8] Creating plots...")
plt.figure(figsize=(16, 6))
plt.plot(data_df.index, data_df['Close'], color='blue', linewidth=1)
plt.xlabel('Datetime')
plt.ylabel('Price (USD)')
plt.title(f'{full_name} ({TICKER}) - Hourly Price History')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{TICKER}_hourly_history.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {TICKER}_hourly_history.png")

# Prediction plot: last 1/3 of training + testing
train_plot_start = len(y_train_act) - len(y_train_act)//3
train_indices = range(train_plot_start, len(y_train_act))
test_indices = range(len(y_train_act), len(y_train_act) + len(y_test_act))

plt.figure(figsize=(16, 8))
plt.plot(train_indices, y_train_act[train_plot_start:], label='True (Train)', color='blue')
plt.plot(train_indices, train_pred[train_plot_start:], label='Predicted (Train)', color='cyan', linestyle='--')
plt.plot(test_indices, y_test_act, label='True (Test)', color='green')
plt.plot(test_indices, test_pred, label='Predicted (Test)', color='red', linestyle='--')
plt.axvline(x=len(y_train_act), color='black', linestyle=':', label='Train/Test Split')
plt.xlabel('Time step (hours)')
plt.ylabel('Price (USD)')
plt.title(f'{TICKER} - Hourly Prediction (Last 1/3 Train + Test)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{TICKER}_hourly_predictions.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {TICKER}_hourly_predictions.png")

# Future predictions plot
future_index = pd.date_range(start=data_df.index[-1] + timedelta(hours=1), periods=predict_hours, freq='H')
plt.figure(figsize=(12, 5))
plt.plot(data_df.index[-48:], data[-48:], label='Last 48 hours')
plt.plot(future_index, future_preds, label='Predicted next hours', marker='o')
plt.axvline(x=data_df.index[-1], color='black', linestyle=':', label='Now')
plt.xlabel('Datetime')
plt.ylabel('Price (USD)')
plt.title(f'{TICKER} - Next {predict_hours} Hour Forecast')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{TICKER}_hourly_future.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {TICKER}_hourly_future.png")

# Training history
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title(f'{TICKER} - Training History')
plt.tight_layout()
plt.savefig(f'{TICKER}_hourly_training.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {TICKER}_hourly_training.png")

print("\nAll done! Check the generated PNG files for visualizations.")
