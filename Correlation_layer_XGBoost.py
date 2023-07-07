import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
from datetime import datetime
import xgboost
from bayes_opt import BayesianOptimization
from math import sqrt

# Read DataSet
df = pd.read_csv("Assinging_VAD_scores_BERT\DataSet\emobank.csv")
# Extract VAD from dataset
VAD = df[["V","A","D"]]

# Split data for train and test
X_train, X_test = train_test_split(np.array(VAD), test_size=0.05, random_state=1225)

# Set input and label
y_train = X_train
y_test = X_test
dtrain = xgboost.DMatrix(X_train, label=y_train)
dtest = xgboost.DMatrix(X_test, label=y_test)

# Define the objective function for Bayesian Optimization
class XGB_evaluate: 
    # Define the parameter bounds for Bayesian Optimization
    param_bounds = {
        'max_depth': (2, 11),
        'learning_rate': (0.01, 0.2),
        'num_boost_round': (100, 1000),
        'subsample': (0.4, 0.9),
        'colsample_bytree': (0.4, 0.9),
    }
    
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def xgb_evaluate(self, max_depth, learning_rate, num_boost_round, subsample, colsample_bytree):
        params ={
            "max_depth": int(max_depth),
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "tree_method": "gpu_hist",
            "gpu_id": 0
        }
        model = xgboost.train(params, dtrain, num_boost_round=int(num_boost_round), evals=[(dtest, 'eval')], early_stopping_rounds=10)
        y_pred = model.predict(self.X_train)
        mse = mean_squared_error(self.y_train, y_pred)
        return -mse  # Minimize the negative MSE

    # Run Bayesian Optimization
    def __call__(self):
        xgb_bo = BayesianOptimization(self.xgb_evaluate, self.param_bounds)
        xgb_bo.maximize(init_points=100, n_iter=500) #Bayesian Optimization's Hyper parameters
    
        return xgb_bo.max['params']

# Function for validation
def Verify(y_test, y_predict):
    for i in range(495, 505):
        print("{:5.3f} | {:5.3f}".format(round(y_test[i], 4), round(y_predict[i], 4)))

    mse = mean_squared_error(y_test, y_predict)
    print("MSE: ", mse)
    print("RMSE", sqrt(mse))
    print("\n===========================\n")

# Get the best hyperparameters
best_params = XGB_evaluate(X_train, y_train)()

# Train the final XGBoost model with the best hyperparameters
XGB_layer = model.train(params, dtrain, num_boost_round=int(best_params["num_boost_round"]), evals=[(dtest, 'eval')], early_stopping_rounds=10)

# Predict
y_predict = XGB_layer.predict(X_test)

# Verify
print("Validate model")
Verify(y_test, y_predict)