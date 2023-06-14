import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization

# Read DataSet
df = pd.read_csv("Assinging_VAD_scores_BERT\DataSet\emobank.csv")
# Extract VAD from dataset
VAD = df[["V","A","D"]]

# Define the objective function for Bayesian Optimization
class Correlation_Layer:
    # Define the parameter bounds for Bayesian Optimization
    param_bounds = {
        "units": (8, 512),
        "kernel_l2_lambda": (0.005, 0.1),
        "activity_l2_lambda": (0.005, 0.1)
    }
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def fitNevaluate(self, units, kernel_l2_lambda, activity_l2_lambda):
        # Build model
        hidden = tf.keras.layers.Dense(
            units=units,
            kernel_regularizer=tf.keras.regularizers.L2(kernel_l2_lambda), 
            activity_regularizer=tf.keras.regularizers.L2(activity_l2_lambda),
            activation="gelu"
            )(self.X_data)
        outputs = tf.keras.layers.Dense(3,activation="gelu")(hidden)
        model = tf.keras.Model(inputs=self.X_data, outputs=outputs)
        # Fit model
        model.fit(self.X_data, self.y_data)
        # Predict
        y_pred = model.predict(self.X_data)
        # MSE
        mse = mean_squared_error(self.y_data, y_pred)
        
        return -mse  # Minimize the negative MSE
    
    # Run Bayesian Optimization
    def __call__(self):
        xgb_bo = BayesianOptimization(self.fitNevaluate, self.param_bounds)
        xgb_bo.maximize(init_points=50, n_iter=200) #Bayesian Optimization's Hyper parameters
        
        return xgb_bo.max['params']
    
    
    # Set input and label
    X_train = y_train = np.array(VAD)
    print(f"Input and Label's sample: {X_train[3]}")
    
    # Get the best hyperparameters
    best_H_params = Correlation_Layer(X_train, y_train)()
    
    # Train the final model with best hyperparameters