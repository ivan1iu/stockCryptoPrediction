

# stockCryptoPrediction

A Python project for **short-term stock and cryptocurrency price prediction** using **LSTM neural networks** on historical time-series data.

This repository is focused on **experimentation and learning**, not live trading.

---

## Overview

The goal of this project is to:
- Explore time-series forecasting with deep learning
- Train LSTM models on historical price data
- Predict short-term price movements for stocks and crypto
- Visualize predictions against actual prices

---

## Key Features

- LSTM-based price prediction models
- Supports both stocks and cryptocurrencies
- Hourly time-series forecasting
- Training, validation, and prediction plots
- CSV exports for analysis

---

## Repository Structure (Simplified)

```text
stockCryptoPrediction/
├── predstock.py
├── predstock_hourly.py
├── predbtc.py
├── run_predstock.sh
├── csv data files
└── output plots


Running the prediction scripts

This project uses a local virtualenv located at `.venv`.

Preferred run (no system changes):

1. Activate the venv:

   source .venv/bin/activate

2. Install requirements (first time only):

   pip install -r requirements.txt

3. Run the script:

   python predstock.py

Or run without activating the venv:

   .venv/bin/python predstock.py

Helper script:

   ./run_predstock.sh

If you see "Required dependency missing: TensorFlow" when running with your system `python`, use the venv python shown above instead. If you want TensorFlow installed into your system Python, run `python -m pip install tensorflow` for that interpreter (not recommended if `.venv` already works).
