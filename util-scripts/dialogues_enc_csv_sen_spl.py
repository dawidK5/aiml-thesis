import json
import pandas as pd
import polars as pl
from tweet_sum_processor import TweetSumProcessor
processor = TweetSumProcessor("./tweet_sum_data_files/archive/twcs/twcs.csv")
with open("./tweet_sum_data_files/final_train_tweetsum.jsonl") as f:
  dialog_with_summaries = processor.get_dialog_with_summaries(f.readlines())
  dials = []
  conv_ids = []
  skipped_dialogues = []
  for dialog_with_summary in dialog_with_summaries:
    if len(dialog_with_summary.get_extractive_summaries()) == 0 or len(dialog_with_summary.get_abstractive_summaries()) == 0:
      skipped_dialogues.append(dialog_with_summary.get_dialog().get_dialog_id())
      continue
#     print(dialog_with_summary)
#     break
# print("987")

    # turns = dialog_with_summary.get_dialog().get_turns()
    # sents = []
    # for turn in turns:
    #   turn.g
    #   dial = " ".join(turn.get_sentences())
    #   print(dial)
    all_sents_dial = []
    dial_obj = dialog_with_summary.get_dialog()
    turns = dial_obj.get_turns()
    for t in turns:
      sents_t = t.get_sentences()
      # Add word "Customer: " if turn by agent
      if t.is_agent():
        sents_t[0] = "Agent: " + sents_t[0]
      else:
        sents_t[0] = "Customer: " + sents_t[0]
        
      all_sents_dial = all_sents_dial + sents_t
    
    all_sents_dial = [s.replace("\n","") for s in all_sents_dial]
    dial = "[CLS] " + all_sents_dial[0] + " " + " [SEP] [CLS] ".join(all_sents_dial[1:]) + " [SEP]"
    # print(dial)
    # break
    dials.append(dial)
    conv_ids.append(dial_obj.get_dialog_id())
    # quit()
    
    # string_format = str(dialog_with_summary)
    # # print(string_format)
    # # print(dir(json_format), type(json_format), json_format, sep='\n')
    # # print(type(string_format), string_format, sep='\n')
    # lines = string_format.splitlines()
    # conv_id = lines[0]
    # dial_lines = []
    # # print(len(lines))
    # ct = 0
    # for idx, l in enumerate(lines[1:]):
    #   if l.startswith("Extractive"):
    #     break
    #   if l.startswith("\tCustomer:") or l.startswith("\tAgent:"):
    #     dial_lines.append(l.strip().replace('\t',' '))
    # if len(dial_lines) == 0:
    #   continue
    # dial = "[CLS] " + dial_lines[0] + " " + " [SEP] [CLS] ".join(dial_lines[1:]) + " [SEP]"
    # dials.append(dial)
    # conv_ids.append(conv_id)
  
# df = pd.DataFrame({'conv_id': conv_ids, 'dialogue': dials}, columns=['conv_id','dialogue'])
print("Skipped dialogues:", skipped_dialogues)
df = pd.DataFrame({'dialogue': dials}, columns=['dialogue'])
df = df.convert_dtypes()
print(df.dtypes)
df_pl = pl.from_pandas(df)
df_pl.write_csv('dials_2007_1744_train_raw.txt', include_header=False, quote_style='never')
