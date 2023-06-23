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
"""Test Model"""

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
model_H_param = H_parameter(num_epochs=4, num_batch_size=32) # <<<<<<<<<<<<<<<<<<<<<< Set Hyper parameters

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

        self.roberta = TFRobertaModel.from_pretrained(model_name, from_pt=True)

        self.predict_V_1 = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02), activation="linear", name="predict_V_1") # Initializer function test
        self.predict_A_1 = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02), activation="linear", name="predict_A_1")
        self.predict_D_1 = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02), activation="linear", name="predict_D_1")

        # Learn Correlation Layers
        self.Corr_layer = tf.keras.models.load_model("Assinging_VAD_scores_BERT\Model\FFNN_VAD_Model_ver1_MSE_00048_20230620-222055") # <<<<< Change the model

    
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
    

# Load trained model
custom_objects = {'TF_RoBERTa_VAD_Classification': TF_RoBERTa_VAD_Classification}
model = tf.keras.models.load_model("Assinging_VAD_scores_BERT\Model\VAD_Assinging_RoBERTa_model_ver1.1_20230623-074029", custom_objects=custom_objects)

# Test Model
for i, 