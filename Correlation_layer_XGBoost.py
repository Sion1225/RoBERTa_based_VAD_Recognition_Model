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
X_train, X_temp = train_test_split(np.array(VAD), test_size=0.1, random_state=1225)
X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=1225)

# Set input and label
y_train = X_train
y_test = X_test
y_val = X_val
dtrain = xgboost.DMatrix(X_train, label=y_train)
dval = xgboost.DMatrix(X_val, label=y_val)
dXtest = xgboost.DMatrix(X_test)

# Define the objective function for Bayesian Optimization
class XGB_evaluate: 
    # Define the parameter bounds for Bayesian Optimization
    param_bounds = {
        'max_depth': (5, 15),
        'learning_rate': (0.01, 0.2),
        'num_boost_round': (100, 1000),
        'subsample': (0.4, 0.9),
        'colsample_bytree': (0.4, 0.9),
        'min_child_weight': (1, 10),
        'reg_lambda': (0, 1),
        'reg_alpha': (0, 1)
    }
    
    def __init__(self, dtrain, dval, y_val):
        self.dtrain = dtrain
        self.dval = dval
        self.y_val = y_val
    
    def xgb_evaluate(self, max_depth, learning_rate, num_boost_round, subsample, colsample_bytree, min_child_weight, reg_lambda, reg_alpha):
        params ={
            "max_depth": int(max_depth),
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "tree_method": "gpu_hist",
            "gpu_id": 0
        }
        model = xgboost.train(params, self.dtrain, num_boost_round=int(num_boost_round), evals=[(self.dval, 'eval')], early_stopping_rounds=10)
        y_pred = model.predict(self.dval)
        mse = mean_squared_error(y_val, y_pred)
        return -mse  # Minimize the negative MSE

    # Run Bayesian Optimization
    def __call__(self):
        self.xgb_bo = BayesianOptimization(self.xgb_evaluate, self.param_bounds)
        self.xgb_bo.maximize(init_points=200, n_iter=1000) #Bayesian Optimization's Hyper parameters
    
        return self.xgb_bo.max['params']

# Function for validation
def Verify(y_test, y_predict):
    for i in range(495, 505):
        print("{:5.3f} | {:5.3f}".format(np.round(y_test[i][0], 4), np.round(y_predict[i][0], 4)))
        print("{:5.3f} | {:5.3f}".format(np.round(y_test[i][1], 4), np.round(y_predict[i][1], 4)))
        print("{:5.3f} | {:5.3f}".format(np.round(y_test[i][2], 4), np.round(y_predict[i][2], 4)))

    mse = mean_squared_error(y_test, y_predict)
    print("MSE: ", mse)
    print("RMSE", sqrt(mse))
    print("\n===========================\n")

# Run Bayesian optimization & Get the best hyperparameters
xgb_eval = XGB_evaluate(dtrain, dval, y_val)
best_params = xgb_eval()

# Write all tried hyperparameters and their results to a log file
with open("Assinging_VAD_scores_BERT\\Learning_log\\XGBoost\\3.txt", "a") as f:
    for i, res in enumerate(xgb_eval.xgb_bo.res):
        f.write(f"iteration {i}:\n")
        f.write(f"\thyperparameters: {res['params']}\n")
        f.write(f"\tobjective: {-res['target']}\n")

# Int
best_params["max_depth"] = int(best_params["max_depth"])
best_params["num_boost_round"] = int(best_params["num_boost_round"])

# Train the final XGBoost model with the best hyperparameters
XGB_layer = xgboost.train(best_params, dtrain, num_boost_round=int(best_params["num_boost_round"]), evals=[(dval, 'eval')], early_stopping_rounds=10)

# Predict
y_predict = XGB_layer.predict(dXtest)

# Verify
print("=============================")
print(best_params)

print("Validate model")
Verify(y_test, y_predict)

# Save model
XGB_layer.save_model("Assinging_VAD_scores_BERT\\Model\\XGBoost\\1st_MSE_7e-6.json")

"""
1st
iteration 634:
	hyperparameters: {'colsample_bytree': 0.4, 'learning_rate': 0.2, 'max_depth': 11, 'min_child_weight': 1.0, 'num_boost_round': 869, 'reg_alpha': 0.0, 'reg_lambda': 1.0, 'subsample': 0.9}
	objective: 6.908561952897087e-06
"""

"""
2nd
{'colsample_bytree': 0.4, 'learning_rate': 0.2, 'max_depth': 11, 'min_child_weight': 1.0, 'num_boost_round': 877, 'reg_alpha': 0.0, 'reg_lambda': 1.0, 'subsample': 0.9}
6.908561952897087e-06
"""