# Part of fine best hyper-parameter for FFNN_VAD_Model

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from FFNN_VAD_model import FFNN_VAD_model

# Read DataSet
df = pd.read_csv("Assinging_VAD_scores_BERT\DataSet\emobank.csv")
# Extract VAD from dataset
VAD = df[["V","A","D"]]

# Define the objective function for Bayesian Optimization
class Correlation_Layer:
    # Define the parameter bounds for Bayesian Optimization
    param_bounds = { # <<<<<<<<<
        "units": (128, 512),
        "kernel_l2_lambda": (0.0001, 0.005),
        "activity_l2_lambda": (0.0001, 0.005),
        "dropout_late": (0, 0.3),
        "batch_size": (8, 200),
        "epochs": (6, 15)
    }
    
    def __init__(self, X_data, y_data, X_test, y_test):
        self.X_data = X_data
        self.y_data = y_data
        self.X_test = X_test
        self.y_test = y_test
        
    def fitNevaluate(self, units, kernel_l2_lambda, activity_l2_lambda, dropout_late, batch_size, epochs):
        # Load model
        model = FFNN_VAD_model(int(units), kernel_l2_lambda, activity_l2_lambda, dropout_late)
        # Model Compile
        model.compile(optimizer="Adam", loss="mse", metrics=["mse"])
        # Fit model
        model.fit(self.X_data, self.y_data, batch_size = int(batch_size), epochs = int(epochs))
        # Predict
        y_pred = model.predict(self.X_test)
        # MSE
        mse = mean_squared_error(self.y_test, y_pred)
        
        return -mse  # Minimize the negative MSE
    
    # Run Bayesian Optimization
    def __call__(self):
        xgb_bo = BayesianOptimization(self.fitNevaluate, self.param_bounds)
        xgb_bo.maximize(init_points=65, n_iter=260) #Bayesian Optimization's Hyper parameters <<<<<<<
        
        return xgb_bo.max['params']
    

# Split data for train and test
X_train, X_test = train_test_split(np.array(VAD), test_size=0.05, random_state=1225)

# Set input and label
y_train = X_train
y_test = X_test
print(f"Input and Label's sample: {X_train[3]}")
    
# Get the best hyperparameters
best_H_params = Correlation_Layer(X_train, y_train, X_test, y_test)()
    
# Train the final model with best hyperparameters
model = FFNN_VAD_model(
    units=int(best_H_params["units"]), 
    kernel_l2_lambda=best_H_params["kernel_l2_lambda"], 
    activity_l2_lambda=best_H_params["activity_l2_lambda"], 
    dropout_late=best_H_params["dropout_late"]
    )
model.compile(optimizer="Adam", loss="mse", metrics=["mse"])

model.fit(X_train, y_train, batch_size = int(best_H_params["batch_size"]), epochs = int(best_H_params["epochs"]))

# Predict
pred = model.predict(X_test)

# Evaluate
out_of_range_count = tf.reduce_sum(tf.cast((pred > 5) | (pred < 0), tf.int32))
print(y_test[:25])
print(pred[:25])

model_MSE = mean_squared_error(y_test, pred)
print(f"MSE: {model_MSE}")
print(f"Count of out of range (0<= pred <=5): {out_of_range_count}")
print(f"Best Hyper-parameter: {best_H_params}")

# Best Hyper parameters
# Xavier & output layer: gelu ; Wrong setting
# ver.1: MSE: 5.154155498358941e-05, 'activity_l2_lambda': 0.001, 'batch_size': 14, 'dropout_late': 0, 'epochs': 11, 'kernel_l2_lambda': 0.001, 'units': 335
# ver.2: MSE: 3.318632144118595e-05, 'activity_l2_lambda': 0.0029252438038930863, 'batch_size': 33, 'dropout_late': 0, 'epochs': 12, 'kernel_l2_lambda': 0.0005, 'units': 342

# He_normal & output layer: linear
# ver.3: MSE: 0.00013909743179621112, 'activity_l2_lambda': 0.0005, 'batch_size': 109, 'dropout_late': 0, 'epochs': 9, 'kernel_l2_lambda': 0.0005, 'units': 385
# ver.4: MSE: 0.00048316358499479045, 'activity_l2_lambda': 0.0006709484309491943, 'batch_size': 18, 'dropout_late': 0.014174288321380634, 'epochs': 12, 'kernel_l2_lambda': 0.0002161249853798384, 'units': 429
# ver.5: MSE: 0.00010226922018003949, 'activity_l2_lambda': 0.002229853672868904, 'batch_size': 43, 'dropout_late': 0.13422752486715445, 'epochs': 14, 'kernel_l2_lambda': 0.0015969551956577952, 'units': 475

# He_uniform & output layer: linear
# ver.6: MSE: 0.004850356600907168, 'activity_l2_lambda': 0.0023742977480897165, 'batch_size': 8, 'dropout_late': 0.10319616603117594, 'epochs': 8, 'kernel_l2_lambda': 0.0007538979997234782, 'units': 208
# ver.7: MSE: 4.063779009953945e-05, 'activity_l2_lambda': 0.0001, 'batch_size': 88, 'dropout_late': 0, 'epochs': 15, 'kernel_l2_lambda': 0.0001, 'units': 466