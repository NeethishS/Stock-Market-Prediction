# Stock Market Prediction App

A simple web application for stock market price prediction using machine learning.

## Features

- 📊 Interactive stock price charts
- 🔮 Future price predictions using Linear Regression
- 📈 Real-time stock data from Yahoo Finance
- 📋 Key metrics and statistics
- 🎯 User-friendly Streamlit interface

## Installation

1. Clone this repository:
```bash
git clone https://github.com/NeethishS/Stock-Market-Prediction.git
cd Stock-Market-Prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and go to `http://localhost:8501`

3. Enter a stock symbol (e.g., AAPL, GOOGL, TSLA) and click "Analyze Stock"

## How it Works

The app uses a Linear Regression model with the following features:
- Historical price data
- Moving averages (5-day and 20-day)
- Trading volume
- Time-based features

## Disclaimer

This is a simple prediction model for educational purposes only. Do not use for actual trading decisions!

## Technologies Used

- Python
- Streamlit (Web UI)
- yfinance (Stock data)
- scikit-learn (Machine Learning)
- Plotly (Interactive charts)
- Pandas & NumPy (Data processing)