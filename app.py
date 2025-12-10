import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Stock Market Prediction",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Stock Market Prediction App")
st.markdown("---")

# Sidebar for user inputs
st.sidebar.header("Stock Selection")
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, GOOGL, TSLA)", value="AAPL")
period = st.sidebar.selectbox("Select Time Period", ["1y", "2y", "5y", "max"])
prediction_days = st.sidebar.slider("Days to Predict", 1, 30, 7)

if st.sidebar.button("Analyze Stock"):
    try:
        # Fetch stock data
        with st.spinner(f"Fetching data for {symbol}..."):
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
        
        if data.empty:
            st.error("No data found for this symbol. Please check the symbol and try again.")
        else:
            # Display basic info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${data['Close'][-1]:.2f}")
            with col2:
                change = data['Close'][-1] - data['Close'][-2]
                st.metric("Daily Change", f"${change:.2f}", f"{change:.2f}")
            with col3:
                st.metric("Volume", f"{data['Volume'][-1]:,}")
            with col4:
                st.metric("52W High", f"${data['High'].max():.2f}")
            
            # Plot historical data
            st.subheader("📊 Historical Price Data")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#1f77b4', width=2)
            ))
            fig.update_layout(
                title=f"{symbol} Stock Price History",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Prepare data for prediction
            st.subheader("🔮 Price Prediction")
            
            # Create features for prediction
            data['Days'] = range(len(data))
            data['MA_5'] = data['Close'].rolling(window=5).mean()
            data['MA_20'] = data['Close'].rolling(window=20).mean()
            data['Price_Change'] = data['Close'].pct_change()
            
            # Remove NaN values
            clean_data = data.dropna()
            
            # Features and target
            features = ['Days', 'MA_5', 'MA_20', 'Volume']
            X = clean_data[features]
            y = clean_data['Close']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions for future days
            last_day = len(data)
            future_days = []
            future_predictions = []
            
            for i in range(1, prediction_days + 1):
                future_day = last_day + i
                last_ma5 = data['Close'][-5:].mean()
                last_ma20 = data['Close'][-20:].mean()
                last_volume = data['Volume'][-1]
                
                future_features = [[future_day, last_ma5, last_ma20, last_volume]]
                prediction = model.predict(future_features)[0]
                
                future_days.append(data.index[-1] + timedelta(days=i))
                future_predictions.append(prediction)
            
            # Plot predictions
            fig_pred = go.Figure()
            
            # Historical data
            fig_pred.add_trace(go.Scatter(
                x=data.index[-30:],
                y=data['Close'][-30:],
                mode='lines',
                name='Historical Price',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Predictions
            fig_pred.add_trace(go.Scatter(
                x=future_days,
                y=future_predictions,
                mode='lines+markers',
                name='Predicted Price',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                marker=dict(size=6)
            ))
            
            fig_pred.update_layout(
                title=f"{symbol} Price Prediction for Next {prediction_days} Days",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Display prediction table
            st.subheader("📋 Prediction Summary")
            pred_df = pd.DataFrame({
                'Date': future_days,
                'Predicted Price': [f"${p:.2f}" for p in future_predictions]
            })
            st.dataframe(pred_df, use_container_width=True)
            
            # Model performance
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Accuracy", f"{train_score:.3f}")
            with col2:
                st.metric("Testing Accuracy", f"{test_score:.3f}")
            
            # Disclaimer
            st.warning("⚠️ This is a simple prediction model for educational purposes. Do not use for actual trading decisions!")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Instructions
else:
    st.info("👈 Enter a stock symbol in the sidebar and click 'Analyze Stock' to get started!")
    
    st.markdown("""
    ### How to use this app:
    1. Enter a stock symbol (e.g., AAPL for Apple, GOOGL for Google)
    2. Select the time period for historical data
    3. Choose how many days ahead you want to predict
    4. Click 'Analyze Stock' to see the results
    
    ### Features:
    - 📊 Historical price charts
    - 🔮 Future price predictions
    - 📈 Key metrics and statistics
    - 📋 Detailed prediction table
    """)