import pandas as pd
from tweet_sum_processor import *



def dialogue_to_text(dial_obj: Dialog, add_newlines:bool = False) -> str:
  all_sents_dial = []
  turns = dial_obj.get_turns()
  for id_t, t in enumerate(turns):
    sents_t = t.get_sentences()
    role = "Agent: " if t.is_agent() else "Customer: "

    turn_line = role + " ".join(sents_t)

    if add_newlines:
      sents_t[0] = ("\n" if (id_t != 0) else "") + role + sents_t[0]
      sents_t = [ sent + "\n" if sent_id != len(sents_t) - 1 else sent for sent_id, sent in enumerate(sents_t)]
      turn_line = "\n".join(sents_t)
    
    all_sents_dial.append(turn_line)

  dial = "\n".join(all_sents_dial)
  return dial


def ext_summary_to_text(ext_summ_obj: List[Turn], highlights:bool=False) -> str:
  cust_sents = []
  agent_sents = []
  assert ext_summ_obj[0] is not None
  for t in ext_summ_obj[0]:
    sents_t = t.get_sentences()
    if t.is_agent():
      agent_sents = agent_sents + sents_t
    else:
      cust_sents = cust_sents + sents_t
  
  cust_sents = [s.replace("\n","") for s in cust_sents]
  agent_sents = [s.replace("\n", "") for s in agent_sents]
  
  if highlights:
    all_sents_sum =  "@highlight\n\nCustomer: " + "\n\n@highlight\n\n".join(cust_sents) + "\n\n@highlight\n\n"+ "Agent: " + "\n\n@highlight\n\n".join(agent_sents)
  else:
    all_sents_sum =  "Customer: " + "\n\n".join(cust_sents) + "\nAgent: " + " ".join(agent_sents)
  return all_sents_sum


def abs_summary_to_text(abs_summ_obj: List[str]) -> str:
  return abs_summ_obj[0][0] + " " + abs_summ_obj[0][1]


def main():
  import os
  print("Preparing data for BERT")
  DATA_DIR = os.path.join(os.getcwd(), 'data')
  processor = TweetSumProcessor("./temp/archive/twcs/twcs.csv")
  for data_split in ("train", "valid", "test"):
    # dials = []
    # ext_summs = []
    # abs_summs = []
    # conv_ids = []
    skipped_dialogues = []

    with open(f"./temp/Tweetsumm/tweet_sum_data_files/final_{data_split}_tweetsum.jsonl") as f:
      dialogues_with_summaries = processor.get_dialog_with_summaries(f.readlines())
      
      for dialogue_with_summary in dialogues_with_summaries:
        dial_obj = dialogue_with_summary.get_dialog()
        ext_summ_obj = dialogue_with_summary.get_extractive_summaries()
        abs_summ_obj = dialogue_with_summary.get_abstractive_summaries()

        if len(ext_summ_obj) == 0 or len(abs_summ_obj) == 0:
          skipped_dialogues.append(dial_obj.get_dialog_id())
          continue
        with open(os.path.join(DATA_DIR, 'raw_dialogues', data_split, dial_obj.get_dialog_id()+".story"),'wb') as fl:
          fl.write(bytes(dialogue_to_text(dial_obj, add_newlines=True)+"\n\n"+ext_summary_to_text(ext_summ_obj, highlights=True), 'utf-8', errors='replace'))
        # dials.append()
        # print(dials)
        # break
        # ext_summs.append(ext_summary_to_text(ext_summ_obj))
        # abs_summs.append(abs_summary_to_text(abs_summ_obj))
                        
        # conv_ids.append(dial_obj.get_dialog_id())
    
    print(f"Skipped {data_split} dataset dialogues:", skipped_dialogues)
  print("Done")
    # df_ext = pd.DataFrame({'conv_id': conv_ids, 'dialogue': dials, 'summary': ext_summs}, columns=['conv_id', 'dialogue','summary'])

    # df_ext = df_ext.convert_dtypes()
    # print(df_ext.dtypes)

    # df_abs = pd.DataFrame({'conv_id': conv_ids, 'dialogue': dials, 'summary': abs_summs}, columns=['conv_id', 'dialogue','summary'])
    # df_abs = df_abs.convert_dtypes()
    # print(df_abs.dtypes)

    # df_ext.to_csv(os.path.join(DATA_DIR, f"dials_ext_2607_1312_{data_split}_spc.csv"), index=False, header=False, quoting=csv.QUOTE_ALL)
    # df_abs.to_csv(os.path.join(DATA_DIR, f"dials_abs_2607_1312_{data_split}_spc.csv"), index=False, header=False, quoting=csv.QUOTE_ALL)

if __name__ == "__main__":
  main()