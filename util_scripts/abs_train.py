from transformers import AutoTokenizer
import os
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")

ds_dir = os.path.join(os.getcwd(), "data")
df = pd.read_csv(os.path.join(ds_dir, "dials_abs_2607_1312_train_spc.csv"), names=['conv_id', 'dialogue', 'summary'], encoding='utf-8', dtype={'conv_id': 'string', 'dialogue': 'string', 'summary': 'string'})

# For each item in df add the length of the tokenized summary column as a new column
df['t5_lengths'] = df['summary'].apply(lambda x: len(tokenizer.encode(x)))

print(df['t5_lengths'].describe())