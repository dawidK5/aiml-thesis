import wandb, os, shutil

api = wandb.Api()
CWD = os.getcwd()
results_t5_dir = os.path.join(CWD, 'results', 't5_results_2309')
os.makedirs(results_t5_dir, exist_ok=True)
os.chdir(results_t5_dir)
results_t5 = api.artifact('aiml-thesis-train-t5-abs-2309-1054/t5-abs-2309-1054:v0')
results_t5.download()
shutil.make_archive(results_t5.name, 'zip', results_t5_dir)
