import json
import pandas as pd
import polars as pl
from tweet_sum_processor import TweetSumProcessor
processor = TweetSumProcessor("./tweet_sum_data_files/archive/twcs/twcs.csv")
with open("./tweet_sum_data_files/final_train_tweetsum.jsonl") as f:
  dialog_with_summaries = processor.get_dialog_with_summaries(f.readlines())
  dials = []
  conv_ids = []
  for dialog_with_summary in dialog_with_summaries:
#     print(dialog_with_summary)
#     break
# print("987")

    # turns = dialog_with_summary.get_dialog().get_turns()
    # sents = []
    # for turn in turns:
    #   turn.g
    #   dial = " ".join(turn.get_sentences())
    #   print(dial)
    json_format = dialog_with_summary.get_json()
    string_format = str(dialog_with_summary)
    # print(string_format)
    # print(dir(json_format), type(json_format), json_format, sep='\n')
    # print(type(string_format), string_format, sep='\n')
    lines = string_format.splitlines()
    conv_id = lines[0]
    dial_lines = []
    # print(len(lines))
    ct = 0
    for idx, l in enumerate(lines[1:]):
      if l.startswith("Extractive"):
        break
      if l.startswith("\tCustomer:") or l.startswith("\tAgent:"):
        dial_lines.append(l.strip().replace('\t',' '))
    if len(dial_lines) == 0:
      continue
    dial = "[CLS] " + dial_lines[0] + " " + " [SEP] [CLS] ".join(dial_lines[1:]) + " [SEP]"
    dials.append(dial)
    conv_ids.append(conv_id)
  
df = pd.DataFrame({'conv_id': conv_ids, 'dialogue': dials}, columns=['conv_id','dialogue'])
df = df.convert_dtypes()
print(df.dtypes)
df_pl = pl.from_pandas(df)
df_pl.write_csv('dials_2007_1226_train_sepcls_pl.csv', quote_style='non_numeric')
