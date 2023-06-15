import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split

# Read DataSet
df = pd.read_csv("Assinging_VAD_scores_BERT\DataSet\emobank.csv")
# Extract VAD from dataset
VAD = df[["V","A","D"]]


# Build model
def FFNN_VAD_model(units, kernel_l2_lambda, activity_l2_lambda, dropout_late):
    inputs = tf.keras.layers.Input(shape=(3,))
    hidden = tf.keras.layers.Dense(
        units=units,
        kernel_regularizer=tf.keras.regularizers.L2(kernel_l2_lambda), 
        activity_regularizer=tf.keras.regularizers.L2(activity_l2_lambda),
        activation="gelu"
    )(inputs)
    hidden = tf.keras.layers.Dropout(dropout_late)(hidden)
    outputs = tf.keras.layers.Dense(3,activation="gelu")(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


# Define the objective function for Bayesian Optimization
class Correlation_Layer:
    # Define the parameter bounds for Bayesian Optimization
    param_bounds = { # <<<<<<<<<
        "units": (128, 512),
        "kernel_l2_lambda": (0.0005, 0.01),
        "activity_l2_lambda": (0.0005, 0.01),
        "dropout_late": (0, 0.3),
        "batch_size": (8, 128),
        "epochs": (4, 13)
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
        xgb_bo.maximize(init_points=60, n_iter=240) #Bayesian Optimization's Hyper parameters <<<<<<<
        
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
model_MSE = mean_squared_error(y_test, pred)
print(model_MSE)
print(f"Best Hyper-parameter: {best_H_params}")

# Best Hyper parameters
# ver.1: MSE: 0.00044273559175212867, 'activity_l2_lambda': 0.023043456153875184, 'batch_size': 18, 'epochs': 11, 'kernel_l2_lambda': 0.001, 'units': 452
# ver.2; Added Dropout: MSE: 5.154155498358941e-05, 'activity_l2_lambda': 0.001, 'batch_size': 14, 'dropout_late': 0, 'epochs': 11, 'kernel_l2_lambda': 0.001, 'units': 335
# ver.3; Added Dropout: MSE: 3.318632144118595e-05, 'activity_l2_lambda': 0.0029252438038930863, 'batch_size': 33, 'dropout_late': 0, 'epochs': 12, 'kernel_l2_lambda': 0.0005, 'units': 342
# Seems like going worng way. need to make better test data.