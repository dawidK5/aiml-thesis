import json
import pandas as pd
import polars as pl
from tweet_sum_processor import TweetSumProcessor

processor = TweetSumProcessor("./notebooks/util_scripts/tweet_sum_data_files/archive/twcs/twcs.csv")
with open("./notebooks/util_scripts/tweet_sum_data_files/final_train_tweetsum.jsonl") as f:
  dialog_with_summaries = processor.get_dialog_with_summaries(f.readlines())
  dials = []
  conv_ids = []
  skipped_dialogues = []
  for dialog_with_summary in dialog_with_summaries:
    if len(dialog_with_summary.get_extractive_summaries()) == 0 or len(dialog_with_summary.get_abstractive_summaries()) == 0:
      skipped_dialogues.append(dialog_with_summary.get_dialog().get_dialog_id())
      continue
    all_sents_dial = []
    dial_obj = dialog_with_summary.get_dialog()
    turns = dial_obj.get_turns()
    for id_t, t in enumerate(turns):
      sents_t = t.get_sentences()
      role = "Agent: " if t.is_agent() else "Customer: "
      sents_t[0] = (" \n " if id_t != 0 else "") + role + sents_t[0]
      sents_t = [ sent + " \n " if sent_id != len(sents_t) - 1 else sent for sent_id, sent in enumerate(sents_t)]
      all_sents_dial = all_sents_dial + sents_t
    dial = "".join(all_sents_dial)
    dials.append(dial)
    conv_ids.append(dial_obj.get_dialog_id())
print("Skipped dialogues:", skipped_dialogues)
df = pd.DataFrame({'dialogue': dials}, columns=['dialogue'])
df = df.convert_dtypes()
print(df.dtypes)
df_pl = pl.from_pandas(df)
df_pl.write_csv('dials_2307_1928_train_sent_n.csv', include_header=False, quote_style='always')
