import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaModel

# Load RoBERTa's Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Encode datas
def convert_datas_to_features(inputs, labels, max_seq_len, tokenizer):
    
    input_ids, attention_mask, toekn_type_dis, data_labels = [], [], [], []
    
    for input, label in tqdm(zip(inputs, labels), total=len(inputs)):
        #--------------------<Progressing>--------------------------------------

# Set RoBERTa's Hyper-parameter
class H_parameter:
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len

# Read and Split data
df = pd.read_csv("Assinging_VAD_scores_BERT\DataSet\emobank.csv")
V, A, D = df["V"], df["A"], df["D"]
text = df["text"]

# RoBERTa's sequence length is 512
RoBERTa_hyper = H_parameter(512)
X_Data = tokenizer.encode("And why I’m writing you today to ask you to renew your support for the pivotal year ahead.",max_length=RoBERTa_hyper.max_seq_len, padding="max_length")

print(X_Data)