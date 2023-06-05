import pandas as pd
import numpy as np
import tensorflow as tf
import transformers import RobertaTokenizer, TFRobertaModel

tokenzier = RobertaTokenizer.from_pretrained("roberta-base")

print(tokenzier.tokenize("I am a student of Tokai."))