# Copyright 2023 by Siwon Seo

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
"""VAD Assinging RoBERTa Model."""

import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, TFRobertaModel
import os
from math import ceil
from datetime import datetime

from kerastuner import BayesianOptimization
from Encode_datas import convert_datas_to_features
from RoBERTa_Learning_scheduler import Linear_schedule_with_warmup
#from FFNN_VAD_model import FFNN_VAD_model

# Load RoBERTa's Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Set RoBERTa Model's Hyper-parameter
class H_parameter:
    def __init__(self, max_seq_len: int = None, num_epochs: int = None, num_batch_size: int = None):
        self.max_seq_len = 512 if max_seq_len is None else max_seq_len # RoBERTa's sequence length is 512
        self.num_epochs = 10 if num_epochs is None else num_epochs
        self.num_batch_size = 32 if num_batch_size is None else num_batch_size

# Set Hyper parameters
model_H_param = H_parameter(num_epochs=20, num_batch_size=16) # <<<<<<<<<<<<<<<<<<<<<< Set Hyper parameters

# Read and Split data
df = pd.read_csv("Assinging_VAD_scores_BERT\DataSet\emobank.csv", keep_default_na=False)
#print(df.isnull().sum())
VAD = df[["V","A","D"]]
V, A, D = df[["V"]], df[["A"]], df[["D"]]
texts = df["text"]

# Encode Datas
input_ids, input_masks = convert_datas_to_features(texts, max_seq_len=model_H_param.max_seq_len, tokenizer=tokenizer)
y_datas = np.array(VAD) # <<<<<< V, A, D

# Split Datas for Train and Test
X_id_train, X_id_test, X_mask_train, X_mask_test, y_train, y_test = train_test_split(input_ids, input_masks, y_datas, test_size=0.1, random_state=1225)

# Assemble ids and masks
X_train = (X_id_train, X_mask_train)
X_test = (X_id_test, X_mask_test)

# Convert the Numpy data to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Shuffle and batch the datasets
BUFFER_SIZE = len(X_train[0])
BATCH_SIZE = model_H_param.num_batch_size
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# load pre-trained model and define the model for fine-tuning
class TF_RoBERTa_VAD_Classification(tf.keras.Model):
    def __init__(self, model_name, units: int, kernel_l2_lambda: float, activity_l2_lambda: float, dropout_rate: float):
        super(TF_RoBERTa_VAD_Classification, self).__init__()

        self.model_name = model_name
        self.roberta = TFRobertaModel.from_pretrained(model_name, from_pt=True)

        self.predict_V_1 = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02), activation="linear", name="predict_V_1") # Initializer function test
        self.predict_A_1 = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02), activation="linear", name="predict_A_1")
        self.predict_D_1 = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02), activation="linear", name="predict_D_1")

        self.units = units
        self.kernel_l2_lambda = kernel_l2_lambda
        self.activity_l2_lambda = activity_l2_lambda
        self.dropout_rate = dropout_rate

        self.hidden1 = tf.keras.layers.Dense(
            units=self.units,
            kernel_regularizer=tf.keras.regularizers.L2(self.kernel_l2_lambda),
            activity_regularizer=tf.keras.regularizers.L2(self.activity_l2_lambda),
            activation="gelu",
            kernel_initializer="he_normal"  # he_normal or he_uniform
        )
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.output_layer = tf.keras.layers.Dense(3, activation="linear")
    
    def call(self, inputs):
        input_ids, attention_mask = inputs

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs[1]

        self.V_1 = self.predict_V_1(cls_token)
        self.A_1 = self.predict_A_1(cls_token)
        self.D_1 = self.predict_D_1(cls_token)

        VAD_1 = tf.concat([self.V_1, self.A_1, self.D_1], 1) # 0: up-down 1: side

        hidden = self.hidden1(VAD_1)
        hidden = self.dropout(hidden)
        ouputs = self.output_layer(hidden)
        return ouputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "model_name": self.model_name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        model = cls(config["model_name"])
        return model
    

# Set Callback function
dir_name = "Assinging_VAD_scores_BERT\Learning_log\Basic"
file_name = "VAD_Assinging_Basic_RoBERTa_model_ver1.3_" + datetime.now().strftime("%Y%m%d-%H%M%S") # <<<<< Edit

def make_tensorboard_dir(dir_name):
    root_logdir = os.path.join(os.curdir, dir_name)
    return os.path.join(root_logdir, file_name)

# Define callbacks
TB_log_dir = make_tensorboard_dir(dir_name)
TensorB = tf.keras.callbacks.TensorBoard(log_dir=TB_log_dir)
ES = tf.keras.callbacks.EarlyStopping(monitor="val_mse", mode="min", patience=4, restore_best_weights=True, verbose=1)

# Define the build_model function for Keras Tuner
def build_model(hp): # Hyper parameter bounds
    units = hp.Int('units', min_value=700, max_value=1000, step=10)
    dropout_rate = hp.Float('dropout_rate', min_value=0.05, max_value=0.3, step=0.01)
    kernel_l2_lambda = hp.Float('kernel_l2_lambda', min_value=0.0001, max_value=0.0025, step=0.0001)
    activity_l2_lambda = hp.Float('activity_l2_lambda', min_value=0.0001, max_value=0.0025, step=0.0001)


    model = TF_RoBERTa_VAD_Classification("roberta-base", units, kernel_l2_lambda, activity_l2_lambda, dropout_rate)
    
    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=hp.Float('learning_rate', min_value=5e-6, max_value=1e-4, step=1e-6),
        weight_decay=hp.Float('weight_decay', min_value=0.0, max_value=0.001, step=0.0001)
        )
    
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss, metrics = ['mse'])
    
    return model

class HyperparametersLogger(tf.keras.callbacks.Callback):
    def __init__(self, tuner, log_file):
        super().__init__()
        self.tuner = tuner
        self.log_file = log_file

    def on_trial_end(self, trial, logs=None):
        with open(self.log_file, 'a') as f:
            f.write(f'Trial {trial.trial_id} ended with hyperparameters: {trial.hyperparameters.values}\n')
        print(trial.hyperparameters.values)

log_file = 'Assinging_VAD_scores_BERT\Model\Basic\hyperparameters_log.txt'

# Instantiate the tuner
tuner = BayesianOptimization(
    build_model,
    objective='val_mse',
    max_trials=50,
    executions_per_trial=2,
    directory='Assinging_VAD_scores_BERT\Model\Basic\Bayesian',
    project_name='VAD_Assinging_Basic_RoBERTa_model_1.3') # <<<<<<<<<<<<<<< edit

# Perform the hyperparameter search
tuner.search(train_dataset,
             validation_data=test_dataset,
             epochs = model_H_param.num_epochs,
             callbacks=[TensorB, ES, HyperparametersLogger(tuner, log_file)])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

# Build the model with the optimal hyperparameters
model = tuner.hypermodel.build(best_hps)

# Train the model
model.fit(train_dataset, validation_data=test_dataset, callbacks=[TensorB, ES])

# Save Model
model_path = os.path.join(os.curdir, "Assinging_VAD_scores_BERT\Model\Basic", file_name)
model.save(model_path)