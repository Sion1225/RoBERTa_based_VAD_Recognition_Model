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
from sklearn.model_selection import KFold

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
df = pd.read_csv("DataSet\emobank.csv", keep_default_na=False)
#print(df.isnull().sum())
VAD = df[["V","A","D"]]
V, A, D = df[["V"]], df[["A"]], df[["D"]]
texts = df["text"]

# Encode Datas
input_ids, input_masks = convert_datas_to_features(texts, max_seq_len=model_H_param.max_seq_len, tokenizer=tokenizer)
y_datas = np.array(VAD) # <<<<<< V, A, D


# For original
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


# define Architecture
class TF_RoBERTa_VAD_Classification(tf.keras.Model):
    def __init__(self, model_name, units: int, kernel_l2_lambda: float, activity_l2_lambda: float, dropout_rate: float):
        super(TF_RoBERTa_VAD_Classification, self).__init__()

        self.model_name = model_name
        self.roberta = TFRobertaModel.from_pretrained(model_name, from_pt=True)
        
        self.units = units
        self.kernel_l2_lambda = kernel_l2_lambda
        self.activity_l2_lambda = activity_l2_lambda
        self.dropout_rate = dropout_rate

        
        # ver.1
        self.hidden1 = tf.keras.layers.Dense(
            units=self.units,
            kernel_regularizer=tf.keras.regularizers.L2(self.kernel_l2_lambda),
            activity_regularizer=tf.keras.regularizers.L2(self.activity_l2_lambda),
            activation="gelu",
            kernel_initializer="he_normal",  # he_normal or he_uniform
            name="Dense_Layer"
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
        '''
        
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.output_layer = tf.keras.layers.Dense(3, activation="linear")

    
    def call(self, inputs):
        input_ids, attention_mask = inputs

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs[1]

        
        # ver.1
        #VAD_1 = tf.concat([self.V_1, self.A_1, self.D_1], 1) # 0: up-down 1: side # ver.1

        hidden = self.hidden1(cls_token) #ver.1

        hidden = self.dropout(hidden)
        
        outputs = self.output_layer(hidden)

        '''
        # ver.2
        hidden_V = self.D_V1(cls_token)
        hidden_A = self.D_A1(cls_token)
        hidden_D = self.D_D1(cls_token)

        hidden_V = self.dropout(hidden_V)
        hidden_A = self.dropout(hidden_A)
        hidden_D = self.dropout(hidden_D)

        hidden = tf.concat([hidden_V, hidden_A, hidden_D], 1) # 0: up-down 1: side # ver.1
        outputs = self.output_layer(hidden)
        '''        

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "model_name": self.model_name,
            "units": self.units,
            "kernel_l2_lambda": self.kernel_l2_lambda,
            "activity_l2_lambda": self.activity_l2_lambda,
            "dropout_rate": self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Set Callback function
dir_name = "Learning_log\Model_S"
file_name = "Model_S_1_" + datetime.now().strftime("%Y%m%d-%H%M%S") # <<<<< Edit

def make_tensorboard_dir(dir_name):
    root_logdir = os.path.join(os.curdir, dir_name)
    return os.path.join(root_logdir, file_name)

# Define callbacks
TB_log_dir = make_tensorboard_dir(dir_name)
TensorB = tf.keras.callbacks.TensorBoard(log_dir=TB_log_dir)
ES = tf.keras.callbacks.EarlyStopping(monitor="val_mse", mode="min", patience=4, restore_best_weights=True, verbose=1)

    
# Define KFold object
kf = KFold(n_splits=30, shuffle=True, random_state=1225)

# Resume KFold test when bug is occured
resume_fold = 0

# Define Model's Hyper-parameters
dic = {'units': 750, 'dropout_rate': 0.13, 'kernel_l2_lambda': 0.0004, 'activity_l2_lambda': 0.0002, 'learning_rate': 3.3e-05, 'weight_decay': 0.0003}

# Validate model
for i, (train_index, test_index) in enumerate(kf.split(input_ids, y_datas)):

    if i < resume_fold:
        continue

    with open(dir_name+"\\Model_MSE&Loss.txt","a") as f:
        f.write(f"\nEnumerate {i}\n")

    # Split input_ids
    input_ids_train, input_ids_test = input_ids[train_index], input_ids[test_index]
    
    # Split input_masks
    input_masks_train, input_masks_test = input_masks[train_index], input_masks[test_index]
    
    # Combine input_ids and input_masks for training and testing
    X_train = (input_ids_train, input_masks_train)
    X_test = (input_ids_test, input_masks_test)

    # Split y_datas
    y_train, y_test = y_datas[train_index], y_datas[test_index]

    # Define Model & Compile
    model = TF_RoBERTa_VAD_Classification("roberta-base", units=dic["units"], kernel_l2_lambda=dic["kernel_l2_lambda"], activity_l2_lambda=dic["activity_l2_lambda"], dropout_rate=dic["dropout_rate"])
    
    optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=dic["learning_rate"], weight_decay=dic["weight_decay"])
    
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss, metrics = ['mse'])

    # Train the model
    model.fit(x=X_train, y=y_train, validation_split=0.075, epochs=model_H_param.num_epochs, batch_size=model_H_param.num_batch_size , callbacks=[TensorB, ES])

    # Test model
    loss, mse = model.evaluate(x=X_test, y=y_test)

    pred = model.predict(X_test)

    print(f"mse: {mse}, loss: {loss}")

    # Note log
    with open(dir_name+"\\Model_MSE&Loss.txt","a") as f:
        f.write(f"mse: {mse}, loss: {loss}\n")

    with open(dir_name+"\\Model_pred.txt","a") as f:
        f.write(f"\nEnumerate {i}\n")

        for j in range(len(pred)):
            f.write(f"predict: {pred[j]}, answer: {y_test[j]}\n")
        f.write('\n')

    # Clear Keras session
    tf.keras.backend.clear_session()


# Save Model
model_path = os.path.join(os.curdir, "Model\Model_S", file_name)
model.save(model_path)