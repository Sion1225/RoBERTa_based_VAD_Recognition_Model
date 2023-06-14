import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, TFRobertaModel

# Load RoBERTa's Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Encode datas
def convert_datas_to_features(inputs, max_seq_len, tokenizer):
    
    input_ids, attention_masks, = [], []
    
    for i, input in enumerate(tqdm(inputs, total=len(inputs))):

        if input is None or input != input:  # Check if input is None or NaN
            print(f"An error occurred at iteration {i} with input {input}")
            continue

        input_id = tokenizer.encode(input, max_length=max_seq_len, padding="max_length")

        # attention mask (padding mask)
        padding_count = input_id.count(tokenizer.pad_token_id) #pad_token_id: 1
        attention_mask = [1] * (max_seq_len - padding_count) + [0] * padding_count #other tokens:1, [pad]:0

        assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)

    return (input_ids, attention_masks)

# Set RoBERTa's Hyper-parameter
class H_parameter:
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len

# RoBERTa's sequence length is 512
RoBERTa_hyper = H_parameter(max_seq_len=512)

# Read and Split data
df = pd.read_csv("Assinging_VAD_scores_BERT\DataSet\emobank.csv")
#print(df.isnull().sum())
VAD = df[["V","A","D"]]
V, A, D = df["V"], df["A"], df["D"]
texts = df["text"]

# Encode Datas
input_ids, input_masks = convert_datas_to_features(texts, max_seq_len=RoBERTa_hyper.max_seq_len, tokenizer=tokenizer)
y_datas = np.array(VAD)

# Split Datas for Train and Test
X_id_train, X_id_test, X_mask_train, X_mask_test, y_train, y_test = train_test_split(input_ids, input_masks, y_datas, test_size=0.1, random_state=1225)

# Assemble ids and masks
X_train = (X_id_train, X_mask_train)
X_test = (X_id_test, X_mask_test)

# load pre-trained model and fine-tuning
class TF_RoBERTa_VAD_Classification(tf.keras.Model):
    def __init__(self, model_name):
        super(TF_RoBERTa_VAD_Classification, self).__init__()

        self.roberta = TFRobertaModel.from_pretrained(model_name, from_pt=True)

        self.predict_V_1 = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02), activation="linear", name="predict_V_1")
        self.predict_A_1 = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02), activation="linear", name="predict_A_1")
        self.predict_D_1 = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02), activation="linear", name="predict_D_1")
        # Learn Correlation Layers
        self.corr_layer_1 = tf.keras.layers.Dense(64, kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02), activation="gelu")
        self.drop_layer_1 = tf.keras.layers.Dropout(0.8)
        self.corr_layer_2 = tf.keras.layers.Dense(3, kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02), activation="gelu")

        self.predict_V_2 = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02), activation="linear", name="predict_V_1")
        self.predict_A_2 = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02), activation="linear", name="predict_A_1")
        self.predict_D_2 = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02), activation="linear", name="predict_D_1")
    
    def call(self, inputs):
        input_ids, attention_mask = inputs

        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs[1]

        V_1 = self.predict_V_1(cls_token)
        A_1 = self.predict_A_1(cls_token)
        D_1 = self.predict_D_1(cls_token)

        Output_of_corr_layer = self.corr_layer_1(V_1, A_1, D_1)
        Output_of_corr_layer = self.drop_layer_1(Output_of_corr_layer)
        Output_of_corr_layer = self.corr_layer_2(Output_of_corr_layer)
        


