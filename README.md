# MarketLens

## Current functions:
- Recieves input ticker
- Reponds with buy/sell/hold analysis
- Provides quantifyable reasoning
- Customtkinter for GUI

## WIP:
- Backtesting
- Prediction data over graph
- Integration of more data structures
- User sentiment analysis with ML


## Logic and Info

Data Fetching and Preprocessing:
- Historical stock data is gathered from yfinance
- Fills/drops NaN values for data integrity

Indicators Calculated:
- SMA (20-day)
- EMA (20-day)
- RSI (14-day)
- MACD (difference of 12-day and 26-day EMAs)
- Bollinger Bands (20-day, ±2 std)
- ADX (14-day)

Prediction Logic:
- Each indicator assigns a buy/hold/sell signal based on threshold values
- Signals are weighted based on importance and combined into a final score
- An additional ML model provides more insight for the decision algorithm

Recommendation Logic:
- Buy if positive
- Sell if negative
- Hold if near zero

Customization:
- Adjust weights for each indicator in prediction_model.py

