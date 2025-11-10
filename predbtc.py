"""
Bitcoin Price Prediction using Current Market Data
Downloads recent Bitcoin data from Yahoo Finance and predicts using LSTM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Bitcoin Price Prediction - Current Market Data")
print("=" * 80)

# ============================================================================
# STEP 1: Download Current Bitcoin Data from Yahoo Finance
# ============================================================================
print("\n[STEP 1] Downloading current Bitcoin data from Yahoo Finance...")

# Define time period - get last 2 years of data for better training
end_date = datetime.now()
start_date = end_date - timedelta(days=730)  # 2 years of data

print(f"Downloading BTC-USD data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# Download Bitcoin data (BTC-USD is the ticker for Bitcoin in USD)
try:
    bitcoin_data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
    print(f"✓ Successfully downloaded {len(bitcoin_data)} days of Bitcoin data")
    print(f"\nData columns: {list(bitcoin_data.columns)}")
    print(f"\nFirst few rows:")
    print(bitcoin_data.head())
    print(f"\nLast few rows (most recent):")
    print(bitcoin_data.tail())
except Exception as e:
    print(f"✗ Error downloading data: {e}")
    print("Make sure you have internet connection and yfinance installed!")
    exit(1)

# Use the 'Close' price for prediction
data = bitcoin_data[['Close']].values
print(f"\nTotal data points: {len(data)}")
print(f"Price range: ${data.min():.2f} - ${data.max():.2f}")
print(f"Current Bitcoin price: ${data[-1][0]:.2f}")

# ============================================================================
# STEP 2: Visualize Recent Bitcoin Price History
# ============================================================================
print("\n[STEP 2] Visualizing Bitcoin price history...")

plt.figure(figsize=(16, 6))
plt.plot(bitcoin_data.index, bitcoin_data['Close'], linewidth=2, color='blue')
plt.xlabel('Date', fontsize=14, fontweight='bold')
plt.ylabel('Bitcoin Price (USD)', fontsize=14, fontweight='bold')
plt.title('Bitcoin Price History (BTC-USD)', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bitcoin_history.png', dpi=300, bbox_inches='tight')
print("✓ Historical price chart saved as 'bitcoin_history.png'")

# ============================================================================
# STEP 3: Prepare Data for LSTM (Same as Homework)
# ============================================================================
print("\n[STEP 3] Preparing data for LSTM...")

# Normalize data to [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)
print(f"✓ Data normalized to range [{data_normalized.min():.4f}, {data_normalized.max():.4f}]")

# Split: 80% training, 20% testing
train_size = int(len(data_normalized) * 0.8)
test_size = len(data_normalized) - train_size

train_data = data_normalized[:train_size]
test_data = data_normalized[train_size:]

print(f"✓ Training set: {len(train_data)} days ({len(train_data)/len(data_normalized)*100:.1f}%)")
print(f"✓ Testing set: {len(test_data)} days ({len(test_data)/len(data_normalized)*100:.1f}%)")

# Create sequences with lookback window = 10
lookback = 10

def create_sequences(data, lookback=10):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Training sequences
x_train, y_train = create_sequences(train_data, lookback)
print(f"✓ Created {len(x_train)} training sequences")

# Testing sequences
test_data_with_context = np.concatenate((train_data[-lookback:], test_data), axis=0)
x_test, y_test = create_sequences(test_data_with_context, lookback)
print(f"✓ Created {len(x_test)} testing sequences")

# Reshape for LSTM: [samples, time steps, features]
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# ============================================================================
# STEP 4: Build and Train LSTM Model (Same Architecture as Homework)
# ============================================================================
print("\n[STEP 4] Building LSTM model...")
print("Architecture: LSTM(4) → Dense(1)")

model = Sequential()
model.add(LSTM(4, input_shape=(x_train.shape[1], 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

print("\nModel Summary:")
model.summary()

print("\n[STEP 5] Training the model (this may take 2-3 minutes)...")
history = model.fit(
    x_train, y_train, 
    batch_size=32,  # Smaller batch size for smaller dataset
    epochs=100, 
    verbose=1,
    validation_split=0.1
)

print("✓ Training completed!")

# ============================================================================
# STEP 6: Make Predictions
# ============================================================================
print("\n[STEP 6] Making predictions...")

train_predictions = model.predict(x_train, verbose=0)
test_predictions = model.predict(x_test, verbose=0)

# Inverse transform to get actual prices
train_predictions = scaler.inverse_transform(train_predictions)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))

test_predictions = scaler.inverse_transform(test_predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

print("✓ Predictions completed")

# ============================================================================
# STEP 7: Calculate Performance Metrics
# ============================================================================
print("\n" + "=" * 80)
print("PERFORMANCE METRICS")
print("=" * 80)

train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))

print(f"\nTraining RMSE: ${train_rmse:.2f}")
print(f"Testing RMSE: ${test_rmse:.2f}")
print(f"RMSE Ratio (Test/Train): {test_rmse/train_rmse:.3f}")

train_mean = y_train_actual.mean()
test_mean = y_test_actual.mean()
print(f"\nTraining RMSE as % of mean: {(train_rmse/train_mean)*100:.2f}%")
print(f"Testing RMSE as % of mean: {(test_rmse/test_mean)*100:.2f}%")

# Performance interpretation
print("\n--- Performance Assessment ---")
if (train_rmse/train_mean)*100 < 5 and (test_rmse/test_mean)*100 < 5:
    print("✓ EXCELLENT performance (RMSE < 5% of mean)")
elif (train_rmse/train_mean)*100 < 10 and (test_rmse/test_mean)*100 < 10:
    print("✓ GOOD performance (RMSE < 10% of mean)")
else:
    print("⚠ Fair performance (RMSE ≥ 10% of mean)")

print("\n--- Overfitting Analysis ---")
ratio = test_rmse / train_rmse
if ratio < 1.2:
    print("✓ No significant overfitting (Test ≈ Train)")
elif ratio < 1.5:
    print("⚠ Mild overfitting detected")
else:
    print("✗ Significant overfitting detected")

# ============================================================================
# STEP 8: Future Price Prediction
# ============================================================================
print("\n" + "=" * 80)
print("FUTURE PRICE PREDICTION")
print("=" * 80)

# Predict next 7 days
print("\nPredicting next 7 days of Bitcoin prices...")

# Get last 10 days of normalized data
last_sequence = data_normalized[-lookback:].reshape(1, lookback, 1)
future_predictions = []

for i in range(7):
    # Predict next day
    next_pred = model.predict(last_sequence, verbose=0)
    future_predictions.append(next_pred[0, 0])
    
    # Update sequence for next prediction
    last_sequence = np.append(last_sequence[0, 1:, :], [[next_pred[0, 0]]], axis=0).reshape(1, lookback, 1)

# Inverse transform predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

print("\nPredicted Bitcoin Prices for Next 7 Days:")
print("-" * 50)
for i, price in enumerate(future_predictions, 1):
    print(f"Day +{i}: ${price[0]:,.2f}")

# ============================================================================
# STEP 9: Visualization
# ============================================================================
print("\n[STEP 9] Creating visualization...")

# Prepare data for plotting
train_dates = bitcoin_data.index[lookback:train_size]
test_dates = bitcoin_data.index[train_size:]

# Last 1/3 of training for plot
train_plot_start = len(train_dates) - len(train_dates) // 3

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

# Plot 1: Training and Testing Performance
ax1.plot(train_dates[train_plot_start:], y_train_actual[train_plot_start:], 
         label='True Price (Training)', color='blue', linewidth=2, alpha=0.7)
ax1.plot(train_dates[train_plot_start:], train_predictions[train_plot_start:], 
         label='Predicted Price (Training)', color='cyan', linewidth=2, linestyle='--', alpha=0.7)
ax1.plot(test_dates, y_test_actual, 
         label='True Price (Testing)', color='green', linewidth=2, alpha=0.7)
ax1.plot(test_dates, test_predictions, 
         label='Predicted Price (Testing)', color='red', linewidth=2, linestyle='--', alpha=0.7)
ax1.axvline(x=test_dates[0], color='black', linestyle=':', linewidth=2, label='Train/Test Split')

ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
ax1.set_ylabel('Bitcoin Price (USD)', fontsize=12, fontweight='bold')
ax1.set_title('Bitcoin Price Prediction - Current Market (Last 1/3 Training + Testing)', 
              fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Future Predictions
future_dates = pd.date_range(start=bitcoin_data.index[-1] + timedelta(days=1), periods=7, freq='D')

ax2.plot(bitcoin_data.index[-30:], data[-30:], 
         label='Historical Prices (Last 30 Days)', color='blue', linewidth=2, alpha=0.7)
ax2.plot(future_dates, future_predictions, 
         label='Predicted Prices (Next 7 Days)', color='red', linewidth=2, linestyle='--', marker='o', markersize=8)
ax2.axvline(x=bitcoin_data.index[-1], color='black', linestyle=':', linewidth=2, label='Today')

ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.set_ylabel('Bitcoin Price (USD)', fontsize=12, fontweight='bold')
ax2.set_title('Bitcoin Price Forecast - Next 7 Days', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bitcoin_current_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Prediction chart saved as 'bitcoin_current_predictions.png'")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
Data Source: Yahoo Finance (BTC-USD)
Time Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
Total Days: {len(data)}

Model Configuration:
  - Architecture: LSTM(4) → Dense(1)
  - Lookback Window: 10 days
  - Train/Test Split: 80/20
  - Epochs: 100
  - Batch Size: 32

Performance:
  - Training RMSE: ${train_rmse:.2f} ({(train_rmse/train_mean)*100:.2f}% of mean)
  - Testing RMSE: ${test_rmse:.2f} ({(test_rmse/test_mean)*100:.2f}% of mean)
  - Test/Train Ratio: {test_rmse/train_rmse:.3f}

Current Bitcoin Price: ${data[-1][0]:,.2f}
Predicted Price Tomorrow: ${future_predictions[0][0]:,.2f}

Files Generated:
  ✓ bitcoin_history.png - Historical price chart
  ✓ bitcoin_current_predictions.png - Predictions and forecast

""")

print("=" * 80)
print("Analysis complete!")
print("=" * 80)

# Optional: Show training loss history
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
plt.title('Model Training History', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("✓ Training history saved as 'training_history.png'")

print("\n🚀 Done! Check the generated images for visualizations.")