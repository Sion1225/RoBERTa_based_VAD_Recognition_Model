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

from Encode_datas import convert_datas_to_features
from RoBERTa_Learning_scheduler import Linear_schedule_with_warmup
#from FFNN_VAD_model import FFNN_VAD_model

# Load RoBERTa's Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Set RoBERTa Model's Hyper-parameter
class H_parameter:
    def __init__(self, max_seq_len: int = None, num_epochs: int = None, num_batch_size: int = None):
        self.max_seq_len = 512 if max_seq_len is None else max_seq_len # RoBERTa's sequence length is 512
        self.num_epochs = 4 if num_epochs is None else num_epochs
        self.num_batch_size = 32 if num_batch_size is None else num_batch_size

# Set Hyper parameters
model_H_param = H_parameter(num_epochs=15, num_batch_size=16) # <<<<<<<<<<<<<<<<<<<<<< Set Hyper parameters

# Read and Split data
df = pd.read_csv("Assinging_VAD_scores_BERT\DataSet\emobank.csv", keep_default_na=False)
#print(df.isnull().sum())
VAD = df[["V","A","D"]]
V, A, D = df["V"], df["A"], df["D"]
texts = df["text"]

# Encode Datas
input_ids, input_masks = convert_datas_to_features(texts, max_seq_len=model_H_param.max_seq_len, tokenizer=tokenizer)
y_datas = np.array(VAD)

# Split Datas for Train and Test
X_id_train, X_id_test, X_mask_train, X_mask_test, y_train, y_test = train_test_split(input_ids, input_masks, y_datas, test_size=0.1, random_state=1225)

# Assemble ids and masks
X_train = (X_id_train, X_mask_train)
X_test = (X_id_test, X_mask_test)


# load pre-trained model and define the model for fine-tuning
class TF_RoBERTa_VAD_Classification(tf.keras.Model):
    def __init__(self, model_name):
        super(TF_RoBERTa_VAD_Classification, self).__init__()

        self.model_name = model_name
        self.roberta = TFRobertaModel.from_pretrained(model_name, from_pt=True)

        self.predict_V_1 = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02), activation="linear", name="predict_V_1") # Initializer function test
        self.predict_A_1 = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02), activation="linear", name="predict_A_1")
        self.predict_D_1 = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02), activation="linear", name="predict_D_1")

        # Learn Correlation Layers
        self.Corr_layer_path = "Assinging_VAD_scores_BERT\Model\FFNN_VAD_Model_ver1_MSE_00048_20230625-231002" # <<<<< Change the model
        self.Corr_layer = tf.keras.models.load_model(self.Corr_layer_path) 

    
    def call(self, inputs):
        input_ids, attention_mask = inputs

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs[1]

        self.V_1 = self.predict_V_1(cls_token)
        self.A_1 = self.predict_A_1(cls_token)
        self.D_1 = self.predict_D_1(cls_token)

        VAD_1 = tf.concat([self.V_1, self.A_1, self.D_1], 1) # 0: up-down 1: side
        final_outputs = self.Corr_layer(VAD_1)

        return final_outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "model_name": self.model_name,
            "Corr_layer_config": self.Corr_layer_path  # suppose Corr_layer_path is the variable that holds the path to Corr_layer
        })
        return config

    @classmethod
    def from_config(cls, config):
        model = cls(config["model_name"])
        model.Corr_layer = tf.keras.models.load_model(config["Corr_layer_config"])
        return model
    

# Set Callback function
dir_name = "Assinging_VAD_scores_BERT\Learning_log"
file_name = "VAD_Assinging_RoBERTa_model_ver1.2_test_" + datetime.now().strftime("%Y%m%d-%H%M%S") # <<<<< Edit

def make_tensorboard_dir(dir_name):
    root_logdir = os.path.join(os.curdir, dir_name)
    return os.path.join(root_logdir, file_name)

# Define callbacks
TB_log_dir = make_tensorboard_dir(dir_name)
TensorB = tf.keras.callbacks.TensorBoard(log_dir=TB_log_dir)
ES = tf.keras.callbacks.EarlyStopping(monitor="accuracy", mode="max", patience=4, restore_best_weights=True, verbose=1)

# load defined model and compile
model = TF_RoBERTa_VAD_Classification("roberta-base")
num_training = ceil((len(X_train[0])/model_H_param.num_batch_size))
warmup_ratio = 0.048 # <<< Hyper-parameter; RoBERTa's: 4.8% (0.048)
lr_schedule = Linear_schedule_with_warmup(max_lr=6e-4, num_warmup=round(num_training*warmup_ratio), num_traning=num_training) # RoBERTa's max_lr: 6e-4, num_training: Number of all backpropagation <<<<<< Hyper parameter
optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-7, weight_decay=0.01) # In the RoBERTa; beta_2=0.98, epsilon=1e-6, weight_decay=0.01 <<<<<< Hyper parameter
loss = tf.keras.losses.MeanSquaredError()
model.compile(optimizer=optimizer, loss=loss, metrics = ['accuracy'])
model.fit(X_train, y_train, epochs=model_H_param.num_epochs, batch_size=model_H_param.num_batch_size, validation_data=(X_test, y_test), callbacks=[TensorB, ES])

# Save Model
model_path = os.path.join(os.curdir, "Assinging_VAD_scores_BERT\Model", file_name)
model.save(model_path)

# Test Model
for i, (id, mask) in enumerate(zip(X_id_test, X_mask_test)):
    if i >= 10:
        break

    pad_start = np.where(mask == 0)[0]
    if len(pad_start) > 0:
        pad_start = pad_start[0]
    else:
        pad_start = len(id)

    id_without_pad = id[:pad_start]

    print(f"Sentence: {tokenizer.decode(id_without_pad)}")
    pred = model.predict((np.array([id]), np.array([mask])))
    print(f"Predicted Value: {pred[0][0], pred[0][1], pred[0][2]}")
    print(pred.shape)