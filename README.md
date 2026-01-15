# alphapulse-finance-ai

"AlphaPulse is an AI-driven market intelligence dashboard that bridges the gap between raw financial data and actionable insights. By combining Natural Language Processing (NLP) for sentiment analysis with Machine Learning for price forecasting, it provides a comprehensive 360-degree view of market trends."

## Key Features
### Financial Sentiment Intelligence (NLP):
Leverages TextBlob to perform real-time sentiment analysis on global financial news headlines, quantifying market mood into Bullish, Bearish, or Neutral signals.
### Predictive Trend Modeling (ML):
Utilizes a Linear Regression model via Scikit-learn to analyze historical price actions and project future price targets based on numerical trends.
### Technical Analytics Engine: 
Automatically calculates and visualizes 20-day and 50-day Moving Averages (MA) using Pandas and NumPy to identify key trend crossovers and market momentum.
### Interactive UI: 
A responsive, professional-grade dashboard built with Streamlit, featuring dynamic sidebar navigation and real-time data input for any global ticker symbol.

## Tech Stack & Tools
Frontend: Streamlit for a responsive and high-performance web interface.

Machine Learning: Scikit-learn for predictive modeling and trend forecasting.

NLP: TextBlob for natural language processing and financial lexicon analysis.

Data Handling: Pandas and NumPy for high-speed data manipulation and technical indicator calculations.

Data Source: yfinance for real-time market data integration.

Development Environment: VS Code on Microsoft Windows.

## How to Run Locally
Clone this repo:

git clone https://github.com/YOUR_USERNAME/alphapulse-finance-ai.git

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py
