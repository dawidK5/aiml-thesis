# %% [markdown]
# # Abstractive summaries - Train DistilBART on TWEETSUMM dataset

# %%
from huggingface_hub import login
import pandas as pd
import numpy as np
import os, time, datetime

from datasets import Dataset, DatasetDict

from transformers import DataCollatorForSeq2Seq, AutoTokenizer, set_seed
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

import wandb

# %%
# !pip freeze > requirements_bart.txt

# %%
ds_dir = os.path.join(os.getcwd(), 'data')
try:
    HF_TOKEN =  os.environ['HF_TOKEN']
except:
    HF_TOKEN = ""

# if 'google.colab' in str(get_ipython()):
#     print("Running on Colab")
#     from google.colab import drive, userdata
#     drive.mount('/content/drive')
#     HF_TOKEN = userdata.get('HF_TOKEN')
# elif os.environ.get('KAGGLE_KERNEL_RUN_TYPE') != None:
#     ds_dir = '/kaggle/input/bertdata2207/'
#     # ds_dir="/kaggle/input/bertdata2207/"
#     from kaggle_secrets import UserSecretsClient
#     print("Running on Kaggle")
#     # ds_dir = "/kaggle/input/tweet-data-2106-1512/"
#     user_secrets = UserSecretsClient()
#     HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
#     WANDB_API_KEY = user_secrets.get_secret("WANDB_API_KEY")
#     os.environ['WANDB_API_KEY'] = WANDB_API_KEY
#     os.makedirs(os.path.join(os.getcwd(), "results"), exist_ok=True)


# %%
set_seed(17)

# %%
def get_current_time():
    return datetime.datetime.now().strftime("%d%m-%H%M")

# %%
run_name = f"bart-abs-{get_current_time()}"

# %%
os.environ["WANDB_PROJECT"] = "aiml-thesis-train-test-temp"
os.environ["WANDB_WATCH"] = "all"
wandb.init(settings=wandb.Settings(start_method="thread"), id=run_name)

# %%
login(token=HF_TOKEN)

# %% [markdown]
# ## Load data

# %%
print(ds_dir)

# %%
checkpoint_bart = "sshleifer/distilbart-xsum-12-6"

# %%
train_df_temp = pd.read_csv(os.path.join(ds_dir,"dials_abs_2607_1312_train_spc.csv"), names=['conv_id','dialogue','summary'], encoding='utf-8', dtype={'conv_id':'string', 'dialogue':'string', 'summary': 'string'})
train_df_temp = train_df_temp.convert_dtypes()
train_df_temp.drop(columns=['conv_id'], inplace=True)
train_df_temp.reset_index(drop=True, inplace=True)

val_df_temp = pd.read_csv(os.path.join(ds_dir,"dials_abs_2607_1312_valid_spc.csv"), names=['conv_id','dialogue','summary'], encoding='utf-8', dtype={'conv_id':'string', 'dialogue':'string', 'summary': 'string'})
val_df_temp = val_df_temp.convert_dtypes()
val_df_temp.drop(columns=['conv_id'], inplace=True)
val_df_temp.reset_index(drop=True, inplace=True)

test_df_temp = pd.read_csv(os.path.join(ds_dir,"dials_abs_2607_1312_test_spc.csv"), names=['conv_id','dialogue','summary'], encoding='utf-8', dtype={'conv_id':'string', 'dialogue':'string', 'summary': 'string'})
test_df_temp = test_df_temp.convert_dtypes()
test_df_temp.reset_index(drop=True, inplace=True)

print(train_df_temp.dtypes)
print(train_df_temp.head())

PD_DATASETS = {'train': train_df_temp, 'validation': val_df_temp, 'test': test_df_temp}

# %%
tweetsumm_abs = DatasetDict(
    {
        'train': Dataset.from_pandas(train_df_temp),
        'validation': Dataset.from_pandas(val_df_temp),
        'test': Dataset.from_pandas(test_df_temp)
    }
)

# %%
tokenizer = AutoTokenizer.from_pretrained(checkpoint_bart)
print(tokenizer)

# %%
# Source: https://huggingface.co/docs/transformers/en/tasks/summarization

def preprocess_function(examples):
    prefix = "summarize: "
    inputs = [str(prefix) + str(dial) for dial in examples["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True) # same params as tweetsumm paper
    labels = tokenizer(text_target=examples["summary"], max_length=80, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# %%
tokenized_tweetsumm_abs = tweetsumm_abs.map(preprocess_function, batched=True, remove_columns=['dialogue','summary'])
print(tokenized_tweetsumm_abs["train"][1])

# %% [markdown]
# ## Setup Training Evaluation

# %%
# !pip install -U nltk

# %%
# !pip install evaluate pyrouge rouge_score bert_score meteor

# %%
import evaluate, nltk, csv
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")

nltk.download('punkt_tab')

# %%
def compute_metrics_abs(eval_pred):
    predictions, labels = eval_pred
    # Extra line added to address an overflow: https://github.com/huggingface/transformers/issues/22634
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]

    rouge_scores = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
    bert_scores = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
    bert_scores.pop('hashcode')
    result = {
      **{f"rouge/{k}": round(v, 4) for k,v in rouge_scores.items()},
      **{f"bertscore/bertscore-{k}": round(np.mean(v), 4) for k,v in bert_scores.items()},
      'meteor': round(meteor.compute(predictions=decoded_preds, references=decoded_labels)['meteor'], 4),
    }
   
    result["gen_len"] = np.mean(prediction_lens)
    return result


# %% [markdown]
# ## Train and Evaluate

# %%
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_bart)

# %%
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# %%
my_batch = data_collator(tokenized_tweetsumm_abs['train'])
assert len(my_batch) == 4 # default setting for the model

# %%
EXPERIMENT_PARAMS = []
BASE_PARAMS = {'lr':3e-5, 'batch_size':4, 'epochs': 6}
EXPERIMENT_PARAMS.append(BASE_PARAMS)

# %%
LEARN_RATES = (3e-5, 3e-4, 3e-6)
BATCH_SIZES = (4, 2, 8)
EPOCHS = (6,10)

for lr in LEARN_RATES:
    for batch_size in BATCH_SIZES:
        for epoch in EPOCHS:
            if lr == BASE_PARAMS['lr'] and batch_size == BASE_PARAMS['batch_size'] and epoch == BASE_PARAMS['epochs']:
                continue
            experiment = {'lr':lr, 'batch_size':batch_size, 'epochs': epoch}
            EXPERIMENT_PARAMS.append(experiment)

# %%
def run_post_training(split, test_details, test_df_temp: pd.DataFrame, tokenizer, experiment, run_name_model, epoch):
    # First line added due to label error, see 
    predictions = np.where(test_details.predictions != -100, test_details.predictions, tokenizer.pad_token_id)
    preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    test_df_temp['response'] = preds
    exp_res = None
    csv_items = {**experiment, **(test_details.metrics)}
    if not exp_res:
        exp_res = {k: list() for k in csv_items.keys()}
    else:
        for k, v in csv_items.items():
            exp_res[k].append(v)

    test_metrics_df = pd.DataFrame(exp_res)
    test_df_temp.convert_dtypes()
    test_metrics_df.convert_dtypes()
    wandb.log({run_name_model: test_details.metrics})
    preds_name = f"{split}_preds_{run_name_model.replace('-','_')}_{epoch}_bart.csv"
    metrics_name =  f"{split}_metrics_{run_name_model.replace('-','_')}_{epoch}_bart.csv"
    test_df_temp.to_csv(os.path.join(os.getcwd(), 'results', preds_name), index=False, header=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
    test_metrics_df.to_csv(os.path.join(os.getcwd(), 'results', metrics_name), index=False, header=True, encoding='utf-8', quoting=csv.QUOTE_ALL)
    # Using wandb documentation: https://docs.wandb.ai/guides/artifacts
    for root, dirs, files in os.walk(os.path.join(os.getcwd(), 'results')):
        for file in files:
            artifact = wandb.Artifact(name=run_name_model, type="predictions")
            artifact.add_file(local_path=os.path.join(root, file), name=file)
            wandb.log_artifact(artifact)


# %%
class ExtraCallback(TrainerCallback):
    def __init__(self):
        self.experiment_rows = []
        
#     def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
#         print(len(state.log_history), state.log_history)
#         self.experiment_rows.append(state.log_history[-1])
#         wandb.log({'run_name': args.run_name, **state.log_history[-1]})
        
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Save loss from state, log current epoch to wandb
        
        # 'lr': args['learning_rate'], 'batch_size': args['per_device_train_batch_size'], 'max_epochs' args['num_train_epochs']
        wandb.log({'run_name': args.run_name, **state.log_history[-1]})
#         df = pd.DataFrame(self.experiment_rows)
#         df = df.convert_dtypes()
#         df.to_csv(os.path.join('.', 'results', args['run_name'] + ".csv", header=True, index=False))
    
    def on_train_end(self, args, state, control, **kwargs):
        # Save and upload CSVs
        df = pd.DataFrame(state.log_history)
        df = df.convert_dtypes()
        df = df.groupby(['epoch'], as_index=False).mean()
        df.to_csv(os.path.join('.', 'results', args.run_name + ".csv"), header=True, index=False)
        
        
#         for split in ('train', 'validation', 'test'):
#             test_details = trainer.predict(tokenized_tweetsumm_abs[split], metric_key_prefix=split)
#             run_post_training(split, test_details, PD_DATASETS[split], tokenizer, exp, run_name_model, state.epoch)
#         if epoch in EPOCHS:
#             trainer.push_to_hub()
        

# %%
exp_res = None
for count, exp in enumerate(EXPERIMENT_PARAMS):
    current_time = get_current_time()
    run_name_model = f"temp-bart-abs-{current_time}-lr-{exp['lr']}-bs-{exp['batch_size']}-maxep-{exp['epochs']}"
    print("Starting experiment", count, run_name_model, "training")
    wandb.run.name = run_name_model
    wandb.run.save()

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join('.', run_model_name),
        eval_strategy="epoch",
        logging_strategy="epoch",
        # logging_steps=100,
        learning_rate=exp['lr'],
        per_device_train_batch_size=exp['batch_size'],
        per_device_eval_batch_size=exp['batch_size'],
        weight_decay=0.01,
        save_strategy="epoch", # "epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=exp['epochs'],
        predict_with_generate=True,
        fp16=True,
        generation_max_length=80,
        push_to_hub=False,
        report_to="none",
        run_name=run_name_model
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_tweetsumm_abs["train"].select(range(0,50)),
        eval_dataset=tokenized_tweetsumm_abs["validation"].select(range(0,10)),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_abs,
    )
    trainer.add_callback(ExtraCallback)
    training_start = time.time()
    trainer.train()
    training_end = time.time()
    print("Finished",  run_name_model, "time it took for training:", str(datetime.timedelta(seconds=(training_end-training_start))))

# %%
def log_csv_wandb(results_path, run_name_model):
    for root, dirs, files in os.walk(results_path):
        for file in files:
            artifact = wandb.Artifact(name=run_name_model, type="predictions")
            artifact.add_file(local_path=os.path.join(root, file), name=file)
            wandb.log_artifact(artifact)

# %%
# !ls results

# %%
log_csv_wandb(os.path.join(os.getcwd(), 'results'), run_name_model)

# %%
print("Finished all training and evaluation for", run_name)
wandb.finish()

# %%
print("Results uploaded")


