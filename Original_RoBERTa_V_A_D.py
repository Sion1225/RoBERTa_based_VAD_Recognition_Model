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
model_H_param = H_parameter(num_epochs=25, num_batch_size=16) # <<<<<<<<<<<<<<<<<<<<<< Set Hyper parameters

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
        
        self.units = units
        self.kernel_l2_lambda = kernel_l2_lambda
        self.activity_l2_lambda = activity_l2_lambda
        self.dropout_rate = dropout_rate

        ''' ver.1
        self.hidden1 = tf.keras.layers.Dense(
            units=self.units,
            kernel_regularizer=tf.keras.regularizers.L2(self.kernel_l2_lambda),
            activity_regularizer=tf.keras.regularizers.L2(self.activity_l2_lambda),
            activation="gelu",
            kernel_initializer="he_normal",  # he_normal or he_uniform
            name="Shared_layer"
        )
        '''
        # ver.2
        self.D_V1 = tf.keras.layers.Dense(
            units=self.units,
            kernel_regularizer=tf.keras.regularizers.L2(self.kernel_l2_lambda),
            activity_regularizer=tf.keras.regularizers.L2(self.activity_l2_lambda),
            activation="gelu",
            kernel_initializer="he_normal",  # he_normal or he_uniform
            name="Dense_V1"
        )
        self.D_A1 = tf.keras.layers.Dense(
            units=self.units,
            kernel_regularizer=tf.keras.regularizers.L2(self.kernel_l2_lambda),
            activity_regularizer=tf.keras.regularizers.L2(self.activity_l2_lambda),
            activation="gelu",
            kernel_initializer="he_normal",  # he_normal or he_uniform
            name="Dense_A1"
        )
        self.D_D1 = tf.keras.layers.Dense(
            units=self.units,
            kernel_regularizer=tf.keras.regularizers.L2(self.kernel_l2_lambda),
            activity_regularizer=tf.keras.regularizers.L2(self.activity_l2_lambda),
            activation="gelu",
            kernel_initializer="he_normal",  # he_normal or he_uniform
            name="Dense_D1"
        )
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.output_layer = tf.keras.layers.Dense(3, activation="linear")
    
    def call(self, inputs):
        input_ids, attention_mask = inputs

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs[1]

        ''' ver.1
        VAD_1 = tf.concat([self.V_1, self.A_1, self.D_1], 1) # 0: up-down 1: side # ver.1

        hidden = self.hidden1(VAD_1) #ver.1

        hidden = self.dropout(hidden)
        ouputs = self.output_layer(hidden)
        '''

        # ver.2
        hidden_V = self.D_V1(cls_token)
        hidden_A = self.D_A1(cls_token)
        hidden_D = self.D_D1(cls_token)

        hidden_V = self.dropout(hidden_V)
        hidden_A = self.dropout(hidden_A)
        hidden_D = self.dropout(hidden_D)

        hidden = tf.concat([hidden_V, hidden_A, hidden_D], 1)
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
file_name = "VAD_Assinging_Basic_RoBERTa_model_ver2.compare_" + datetime.now().strftime("%Y%m%d-%H%M%S") # <<<<< Edit

def make_tensorboard_dir(dir_name):
    root_logdir = os.path.join(os.curdir, dir_name)
    return os.path.join(root_logdir, file_name)

# Define callbacks
TB_log_dir = make_tensorboard_dir(dir_name)
TensorB = tf.keras.callbacks.TensorBoard(log_dir=TB_log_dir)
ES = tf.keras.callbacks.EarlyStopping(monitor="val_mse", mode="min", patience=4, restore_best_weights=True, verbose=1)

'''
# Define the build_model function for Keras Tuner
def build_model(hp): # Hyper parameter bounds
    units = hp.Int('units', min_value=700, max_value=1200, step=10)
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.3, step=0.01)
    kernel_l2_lambda = hp.Float('kernel_l2_lambda', min_value=0.0001, max_value=0.0025, step=0.0001)
    activity_l2_lambda = hp.Float('activity_l2_lambda', min_value=0.0001, max_value=0.0025, step=0.0001)


    model = TF_RoBERTa_VAD_Classification("roberta-base", units, kernel_l2_lambda, activity_l2_lambda, dropout_rate)
    
    optimizer = tf.keras.optimizers.experimental.AdamW(
        learning_rate=hp.Float('learning_rate', min_value=1e-6, max_value=7e-5, step=1e-6),
        weight_decay=hp.Float('weight_decay', min_value=0.0, max_value=0.0005, step=0.0001)
        )
    
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss, metrics = ['mse'])
    
    return model

# Instantiate the tuner
tuner = BayesianOptimization(
    build_model,
    objective='val_mse',
    max_trials=50,
    executions_per_trial=2,
    #directory='Assinging_VAD_scores_BERT\Model\Basic\Bayesian',
    directory="D:\\Experiment Datas\\Assinging_VAD_scores_BERT\\Model\\Basic\\Bayesian",
    project_name='VAD_Assinging_Basic_RoBERTa_model_1.3') # <<<<<<<<<<<<<<< edit


# Perform the hyperparameter search
tuner.search(train_dataset,
             validation_data=test_dataset,
             epochs = model_H_param.num_epochs,
             callbacks=[TensorB, ES])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

# Build the model with the optimal hyperparameters
model = tuner.hypermodel.build(best_hps)
'''

a = {'units': 700, 'dropout_rate': 0.16, 'kernel_l2_lambda': 0.0004, 'activity_l2_lambda': 0.0008, 'learning_rate': 5.5e-05, 'weight_decay': 0.0002}
b = {'units': 750, 'dropout_rate': 0.13, 'kernel_l2_lambda': 0.0004, 'activity_l2_lambda': 0.0002, 'learning_rate': 3.3e-05, 'weight_decay': 0.0003}
dic = (a, b)


for i in range(len(dic)):

    with open(dir_name+"\\val_datas.txt","a") as f:
        f.write("Ver1.3\n")
        f.write(f"Hyper-parameters: {dic[i]}\n")

    for _ in range(5):
        model = TF_RoBERTa_VAD_Classification("roberta-base", units=dic[i]["units"], kernel_l2_lambda=dic[i]["kernel_l2_lambda"], activity_l2_lambda=dic[i]["activity_l2_lambda"], dropout_rate=dic[i]["dropout_rate"])
    
        optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=dic[i]["learning_rate"], weight_decay=dic[i]["weight_decay"])
    
        loss = tf.keras.losses.MeanSquaredError()
        model.compile(optimizer=optimizer, loss=loss, metrics = ['mse'])

        # Train the model
        model.fit(train_dataset, validation_data=test_dataset, epochs = model_H_param.num_epochs, callbacks=[TensorB, ES])

        # Test model
        loss, mse = model.evaluate(test_dataset)

        print(f"mse: {mse}, loss: {loss}")

        # Note log
        with open(dir_name+"\\val_datas.txt","a") as f:
            f.write(f"mse: {mse}, loss: {loss}\n")



'''
# Save Model
model_path = os.path.join(os.curdir, "Assinging_VAD_scores_BERT\Model\Basic", file_name)
model.save(model_path)
'''