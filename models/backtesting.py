class Backtesting:
    def __init__(self, data):
        """
        Initializes the Backtesting class.

        :param data: DataFrame with historical stock data.
        """
        self.data = data

    def backtest_strategy(self, strategy_function):
        """
        Backtests a strategy on historical data by generating buy/sell/hold signals.

        :param strategy_function: Function that generates recommendations based on row data.
        :return: DataFrame with signals added.
        """
        signals = []
        
        for index, row in self.data.iterrows():
            recommendation, _ = strategy_function(row)  # Pass each row to strategy function
            if recommendation == "Buy":
                signal = 1
            elif recommendation == "Sell":
                signal = -1
            else:
                signal = 0  # Hold signal
            signals.append(signal)
        
        # Adding 'Signal' column to data
        self.data['Signal'] = signals
        # Fill any NaN values in Signal column with 0 (default to 'Hold')
        self.data['Signal'].fillna(0, inplace=True)
        return self.data

    def calculate_performance_metrics(self, backtest_results):
        """
        Calculates performance metrics for the strategy.

        :param backtest_results: DataFrame with 'Close' prices and 'Signal' column.
        :return: Dictionary with total return, annual volatility, and Sharpe ratio.
        """
        # Calculate daily returns and fill any NaN values with 0
        backtest_results['Return'] = backtest_results['Close'].pct_change().fillna(0)
        
        # Calculate strategy returns; shift Signal and fill any NaNs with 0
        backtest_results['Strategy Return'] = backtest_results['Signal'].shift(1).fillna(0) * backtest_results['Return']

        # Check for any remaining NaNs
        backtest_results['Strategy Return'].fillna(0, inplace=True)
        
        # Calculate cumulative return
        cumulative_return = (1 + backtest_results['Strategy Return']).cumprod() - 1
        total_return = cumulative_return.iloc[-1]

        # Calculate annualized metrics
        annual_volatility = backtest_results['Strategy Return'].std() * (252**0.5)
        sharpe_ratio = (backtest_results['Strategy Return'].mean() / backtest_results['Strategy Return'].std()) * (252**0.5)
        
        return {
            'Total Return': total_return,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratio
        }
