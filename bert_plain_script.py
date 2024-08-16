# %% [markdown]
# # BERT extractive summarization

# %%
import os

# %%
os.chdir("/kaggle/working")

# %%
!ls -la

# %%
!wget https://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip

# %%
!mv stanford-corenlp-full-2017-06-09.zip.1 stanford-corenlp-full-2017-06-09.zip

# %%
!unzip stanford-corenlp-full-2017-06-09.zip

# %%
!ls -la stanford-corenlp-full-2017-06-09

# %%
%env CLASSPATH=/kaggle/working/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar:/kaggle/working/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0-models.jar

# %%
!git clone https://github.com/nlpyang/BertSum.git

# %%
!ls -la

# %%
wkdir_prefix = "/kaggle/working"

# %%
import os
os.chdir(wkdir_prefix + "/BertSum/src")

# %%
!ls -la

# %%
!pip install pytorch_pretrained_bert pyrouge

# %%
import pandas as pd
data_prefix = "/kaggle/input/gptdata/"

# %%
df_train = pd.read_csv(data_prefix+"dials_ext_2607_1312_train_spc.csv", names=['conv_id','dialogue','summary'], dtype={'conv_id': 'string', 'dialogue':'string', 'summary':'string'})
df_train.convert_dtypes()
print(df_train.dtypes)

# %%
!mkdir raw_dialogues

# %%
df_train['dialogue'][0].split('\nAgent')

# %%
!wget https://github.com/dawidK5/aiml-thesis/raw/v001/data/raw_dialogues.zip

# %%
!unzip raw_dialogues.zip

# %%
!mkdir /kaggle/working/logs

# %%
!mkdir -p ./tokenized_dialogues/train

# %%
!mkdir -p ./json_dialogues/train

# %%
!ls -la

# %%
!ls -la

# %%
!python preprocess.py -mode tokenize -raw_path "./raw_dialogues/train" -save_path "./tokenized_dialogues/train"

# %%
!zip -r tokenized_dials_bertsum.zip "./tokenized_dialogues/train"

# %%
# !python preprocess.py -mode format_to_lines -raw_path "./tokenized_dialogues/train" -save_path "./json_dialogues/train" -map_path "./" -lower
# Doesn't work, have to redo manually

# %%
os.chdir("./prepro")
import data_builder

# %%
!wget https://github.com/dawidK5/aiml-thesis/raw/v001/data/tokenized_dials_bertsum.zip

# %%
!unzip tokenized_dials_bertsum.zip

# %%
import os,json

# %%
!git clone https://github.com/nlpyang/BertSum.git

# %%
os.chdir("./src/prepro")

# %%
os.chdir("../")

# %%
import preprocess, prepro.data_builder

# %%
!ls -la

# %%
os.getcwd()

# %%
!mkdir -p ./json_dialogues/train

# %%
!rm -r ./tokenized_dialogues

# %%
!rm -r ./json_dialogues

# %%
for root, _, files in os.walk(os.path.join(os.getcwd(),'tokenized_dialogues', 'train')):
    file_data = []
    for dialsum_file in files:
        src, tgt = prepro.data_builder.load_json(os.path.join(root, dialsum_file), lower=True)
        file_data.append({'src': src, 'tgt': tgt})
    with open(f"./json_dialogues/train/bertsumdata.train.0.json", 'w', encoding='utf-8') as jfile:
        jfile.write(json.dumps(file_data))

# %%
os.chdir("./BertSum/")

# %%
!ls -la ./json_dialogues/train

# %%
!pwd && ls -la && head -n 5 ./json_dialogues/train/bertsum_ready_data_1608.train.0.json

# %%
!mkdir -p ./bert_pt

# %%
!mkdir -p ./logs/

# %%
os.chdir("../")

# %%
os.chdir("./BertSum/src/prepro")

# %%
os.chdir("./src")

# %%
!ls -la ./json_dialogues/train

# %%
!python preprocess.py -mode format_to_bert -raw_path ./json_dialogues/train -save_path ./bert_pt -oracle_mode greedy -n_cpus 4 -log_file ./logs/preprocess.log

# %%
!ls -la

# %%
!cat ./logs/preprocess.log

# %%
os.chdir("./json_dialogues/train")

# %%
os.chdir("../")

# %%
os.chdir("./train")

# %%
!ls -la


