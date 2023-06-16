# Part of pre-train with best hyper-parameters FFNN_VAD_Model

# Best Hyper parameters
# ver.1: MSE: 0.00044273559175212867, 'activity_l2_lambda': 0.023043456153875184, 'batch_size': 18, 'epochs': 11, 'kernel_l2_lambda': 0.001, 'units': 452
# ver.2; Added Dropout: MSE: 5.154155498358941e-05, 'activity_l2_lambda': 0.001, 'batch_size': 14, 'dropout_late': 0, 'epochs': 11, 'kernel_l2_lambda': 0.001, 'units': 335 || Test MSE: 0.00009378134
# ver.3; Added Dropout: MSE: 3.318632144118595e-05, 'activity_l2_lambda': 0.0029252438038930863, 'batch_size': 33, 'dropout_late': 0, 'epochs': 12, 'kernel_l2_lambda': 0.0005, 'units': 342 || Test MSE: 0.000065812168

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
from datetime import datetime

from FFNN_VAD_model import FFNN_VAD_model

# Setting for Tensorboard
dir_name = "Assinging_VAD_scores_BERT\Learning_log"

def make_tensorboard_dir(dir_name):
    root_logdir = os.path.join(os.curdir, dir_name)
    sub_dir_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(root_logdir, sub_dir_name)

# Read DataSet
df = pd.read_csv("Assinging_VAD_scores_BERT\DataSet\emobank.csv")
# Extract VAD from dataset
VAD = df[["V","A","D"]]

# Split data for train and test
X_train, X_test = train_test_split(np.array(VAD), test_size=0.05, random_state=1225)

# Set input and label
y_train = X_train
y_test = X_test
print(f"Input and Label's sample: {X_train[3]}")

# Set Hyper-parameter
Corr_model = FFNN_VAD_model(
    units=342,
    kernel_l2_lambda= 0.0005,
    activity_l2_lambda=0.0029252438038930863,
    dropout_late=0
)
Corr_model.compile(optimizer="Adam", loss="mse", metrics=["mse"])

# Define Callback function
TB_log_dir = make_tensorboard_dir(dir_name)
TensorB = tf.keras.callbacks.TensorBoard(log_dir=TB_log_dir)

# Model train
Corr_model.fit(X_train, y_train, batch_size = 33, epochs = 12, callbacks=[TensorB])

# Save Model
Corr_model.save("Assinging_VAD_scores_BERT\Model\FFNN_VAD_Model_ver3_" + datetime.now().strftime("%Y%m%d-%H%M%S"))

# Load Model
#Corr_model = tf.keras.models.load_model("Assinging_VAD_scores_BERT\Model\FFNN_VAD_Model_ver3")

# Predict
pred = Corr_model.predict(X_test)

# Evaluate
print(y_test[:25])
print(pred[:25])

model_MSE = mean_squared_error(y_test, pred)
print(model_MSE)