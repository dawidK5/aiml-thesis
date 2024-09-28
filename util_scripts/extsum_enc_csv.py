# Extracts extractive summaries

import json
import pandas as pd
import polars as pl
from tweet_sum_processor import TweetSumProcessor
processor = TweetSumProcessor("./notebooks/util_scripts/tweet_sum_data_files/archive/twcs/twcs.csv")
with open("./.notebooks/util_scripts/tweet_sum_data_files/final_train_tweetsum.jsonl") as f:
  dialog_with_summaries = processor.get_dialog_with_summaries(f.readlines())
  ext_summs = []
  conv_ids = []
  for dialog_with_summary in dialog_with_summaries:
    
    all_ext_sums_turns = dialog_with_summary.get_extractive_summaries()
    if len(all_ext_sums_turns) == 0:
      print(dialog_with_summary.get_dialog().get_dialog_id())
      continue
    cust_sents = []
    agent_sents = []

    for t in all_ext_sums_turns[0]:
      sents_t = t.get_sentences()
      if t.is_agent():
        agent_sents = agent_sents + sents_t
      else:
        cust_sents = cust_sents + sents_t
        
    cust_sents = [s.replace("\n","") for s in cust_sents]
    agent_sents = [s.replace("\n", "") for s in agent_sents]
    all_sents_sum =  "Customer: " + " ".join(cust_sents) + " Agent: " + " ".join(agent_sents)
    ext_summs.append(all_sents_sum)
    conv_ids.append(dialog_with_summary.get_dialog().get_dialog_id())
  
df = pd.DataFrame({'summary': ext_summs}, columns=['summary'])
df = df.convert_dtypes()
print(df.dtypes)
df_pl = pl.from_pandas(df)
# df_pl.write_csv('extsums_2107_1240_train_raw.txt', include_header=False, quote_style='never')
df_pl.write_csv('extsums_2207_2048_train.csv', include_header=False, quote_style='always')