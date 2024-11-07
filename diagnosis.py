from data.stock_data import StockData
from models.prediction_model import PredictionModel
from models.machine_learning_model import MachineLearningModel


ticker = "AAPL"
stock_data = StockData(ticker)
stock_data.fetch_data()  
preprocessed_data = stock_data.preprocess_data()  
print("Preprocessed Data Sample:")
print(preprocessed_data.head())  


prediction_model = PredictionModel(preprocessed_data)
final_recommendation, reasons = prediction_model.combined_strategy()
print("Final Recommendation:", final_recommendation)
print("Reasons:", reasons)


ml_model = MachineLearningModel(preprocessed_data)
ml_model.train_model() 

X = preprocessed_data.drop(columns=['Close']).iloc[-1].values.reshape(1, -1)
ml_prediction = ml_model.predict(X)
print("Machine Learning Prediction:", "Buy" if ml_prediction[0] == 1 else "Sell")
