import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(page_title="AlphaPulse AI", layout="wide", page_icon="📈")
st.title("⚡ AlphaPulse: Real-time Financial Sentiment & Technical Analytics")
st.markdown("---")

# --- Sidebar Configuration ---
st.sidebar.header("User Control Panel")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()
forecast_days = st.sidebar.slider("AI Forecast Period (Days)", 1, 15, 5)

# --- Data Fetching Engine ---
@st.cache_data
def get_data(symbol):
    return yf.download(symbol, start="2023-01-01")

data = get_data(ticker)

if data.empty:
    st.error("Ticker not found. Please verify the symbol (e.g., TSLA, NVDA).")
else:
    # --- Column Layout for Dashboard ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Technical Trend Analysis")
        # Moving Averages using NumPy/Pandas
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data['Close'], label="Market Price", color='#1f77b4', linewidth=2)
        ax.plot(data['MA20'], label="Short-term (20D)", color='#ff7f0e', linestyle='--')
        ax.plot(data['MA50'], label="Long-term (50D)", color='#2ca02c')
        ax.set_facecolor('#f8f9fa')
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("🤖 NLP Sentiment Intelligence")
        try:
            raw_news = yf.Ticker(ticker).news
            # Data Cleaning: Filter out entries without real headlines
            valid_news = [n for n in raw_news if n.get('title') and len(n.get('title')) > 10]
            
            if not valid_news:
                st.info("No active news pulses detected.")
            else:
                scores = []
                for item in valid_news[:5]:
                    headline = item['title']
                    analysis = TextBlob(headline)
                    score = analysis.sentiment.polarity
                    scores.append(score)
                    
                    # Sentiment Visualization
                    icon = "🟢" if score > 0.05 else "🔴" if score < -0.05 else "⚪"
                    st.write(f"{icon} {headline[:75]}...")

                avg_mood = np.mean(scores)
                st.divider()
                if avg_mood > 0.05:
                    st.success(f"Pulse Signal: BULLISH ({avg_mood:.2f})")
                elif avg_mood < -0.05:
                    st.error(f"Pulse Signal: BEARISH ({avg_mood:.2f})")
                else:
                    st.warning(f"Pulse Signal: NEUTRAL ({avg_mood:.2f})")
        except:
            st.write("Sentiment engine offline.")

    st.markdown("---")

    # --- Machine Learning: AlphaPulse Forecast ---
    st.subheader("🔮 AlphaPulse AI Price Projection")
    
    # ML Preprocessing
    df_ml = data.reset_index()
    df_ml['Day_Index'] = df_ml.index
    X = df_ml[['Day_Index']].values
    y = df_ml['Close'].values
    
    # Scikit-learn Linear Regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Future Prediction logic
    last_idx = df_ml['Day_Index'].iloc[-1]
    future_range = np.array([last_idx + i for i in range(1, forecast_days + 1)]).reshape(-1, 1)
    future_preds = model.predict(future_range)
    
    # Forecasting Visual
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(df_ml['Day_Index'][-60:], y[-60:], label="Historical Data", color='gray')
    ax2.plot(future_range, future_preds, 'ro--', label="AlphaPulse Forecast")
    ax2.set_title("Future Trend Projection")
    ax2.legend()
    st.pyplot(fig2)
    
    # Final Result with Array-to-Scalar Fix
    target_price = future_preds[-1].item()
    st.info(f"Predicted target for {ticker} in {forecast_days} days: **${target_price:.2f}**")