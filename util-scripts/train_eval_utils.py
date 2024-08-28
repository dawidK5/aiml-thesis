class TrainEvalUtils:
    import os
    import pandas as pd
    from rouge_score import rouge_scorer

    def __init__(self, split='train'):
        self.split = split

    def calc_metrics_df(df, rouge_only=True):
        df_test_results_lst = []
        for idx, row in df.iterrows():
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
            rouge_scores = scorer.score(row['summary'], row['response'])
            rouges = dict()
            for k,v in rouge_scores.items():
                rouges[f"{k}_pr"] = round(v.precision, 4)
                rouges[f"{k}_re"] = round(v.recall, 4)
                rouges[f"{k}_f1"] = round(v.fmeasure, 4)
            if rouge_only == False:
                bert_scores = bertscore.compute(predictions=[row['response']], references=[row['summary']], lang="en")
                bert_scores.pop('hashcode')
                result = {
                    **rouges,
                    **{f"bertscore_{k[:2]}": round(v[0], 4) for k,v in bert_scores.items()},
                    'meteor': round(meteor.compute(predictions=[row['response']], references=[row['summary']])['meteor'], 4),
                }
                row_res = {
                    'conv_id': row['conv_id'],
                    **result,
                }
              
            else:
                result = {
                    **rouges,
                }
                row_res = {
                    **result,
                }
            df_test_results_lst.append(row_res)
        return df_test_results_lst

    def evaluate_rouge(split_name):
      for root, _, files in os.walk(bertsum_src_dir, topdown=True):
          file_data = []
          for results_file in files:
              if results_file.startswith("results_step") and results_file.endswith(".gold"):
                  step_num = results_file[12:-5]
                  cand_filename = results_file[:-5] + ".candidate"
                  with open(os.path.join(root, results_file), 'r') as gold_file:
                      with open(os.path.join(root, cand_filename), 'r') as cand_file:
                          df = pd.DataFrame(
                              {
                                  'summary': gold_file.readlines(),
                                  'response': cand_file.readlines(),
                              }
                          )
                  results_df = pd.DataFrame(calc_metrics_df(df))
                  results_df.to_csv(f"./results/{split_name}_res_bertsum_s{step_num}_2408_1327.csv", index=False, header=True)
          print("Finished saving evaluation results")
          break