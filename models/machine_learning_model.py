from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

class MachineLearningModel:
    def __init__(self, data, prediction_horizon=1):
        """
        Initializes the MachineLearningModel.

        :param data: DataFrame with historical stock data and calculated indicators.
        :param prediction_horizon: Number of days ahead for prediction target (default is 1).
        """
        self.data = data
        self.model = LogisticRegression()
        self.prediction_horizon = prediction_horizon

    def preprocess_data(self):
        X = self.data.drop(columns=['Close'])
        y = (self.data['Close'].shift(-self.prediction_horizon) > self.data['Close']).astype(int)
        
        # Fill NaNs with forward fill; fallback to mean fill if necessary
        X = X.ffill().fillna(X.mean())
        
        # Remove rows where target is NaN due to shifting
        X = X.iloc[:-self.prediction_horizon]
        y = y.iloc[:-self.prediction_horizon]
        
        return X, y


    def train_model(self):
        # Prepare data
        X, y = self.preprocess_data()

        # Split data into training and validation sets (80/20 split)
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Train the model on the training set
        self.model.fit(X_train, y_train)
        
        # Validate and log accuracy on the validation set
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Validation Accuracy: {accuracy:.2f}")

    def predict(self, X):
        # Ensure X is a DataFrame and has matching columns to the training data
        if isinstance(X, pd.Series):
            X = X.to_frame().T  # Convert Series to DataFrame if necessary
        X_df = pd.DataFrame(X, columns=self.data.columns.drop('Close'))
        
        # Fill any remaining NaNs with forward fill; fallback to mean fill if necessary
        X_df = X_df.ffill().fillna(X_df.mean())

        # Make prediction and return the result
        return self.model.predict(X_df)


