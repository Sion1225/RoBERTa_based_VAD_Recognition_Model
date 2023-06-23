"""Encode datas"""

import numpy as np
from tqdm import tqdm

def convert_datas_to_features(inputs: list, max_seq_len: int, tokenizer):
    
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
