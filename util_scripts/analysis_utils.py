import re, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_file_to_df(results_dir):
    exp_epochs = dict()
    test_metrics = dict()
    for root, _, files in os.walk(results_dir):
        count = 0
        for file in files:
            if 'log' in file and 'csv' in file:
                exp_epochs[file] = pd.read_csv(os.path.join(root, file))
            elif 'metrics' in file:
                # multi bar plots
                test_metrics[file] = pd.read_csv(os.path.join(root, file))
    exp_epochs = dict(sorted(exp_epochs.items(), key=lambda x: get_params_from_name(x[0])))
    # print(list(exp_epochs.keys()))
    test_metrics = dict(sorted(test_metrics.items(), key=lambda x: get_params_from_name(x[0])))
    # print(list(test_metrics.keys()))
    return exp_epochs, test_metrics

def get_params_from_name(file):
    file_pattern = r"lr_(\d+[\.e_\d]+)_bs_(\d+)_maxep_(\d+)"
    if 'metrics' in file:
        file_pattern += r"_s(\d+)"
    match = re.search(file_pattern, file)
    try:
        lr = float(match.group(1).replace("_", "-"))
        lr = float(f"{lr:.2e}")
        batch_size = int(match.group(2))
        max_epoch = int(match.group(3))
        step = int(match.group(4)) if 'metrics' in file else int(0)
        # print(f"lr: {lr}, batch_size: {batch_size}, max_epoch: {max_epoch}")
        return lr, batch_size, max_epoch, step
    except AttributeError:
        print(file, "not matched with regex")

def params_to_shortname(lr, bs, ep, step):
    return f"lr:{lr:1.0e} bs:{bs} ep:{ep} st:{step}" if step != 0 else f"lr:{lr:1.0e} bs:{bs} ep:{ep}"

def model_table(model_name, name_to_df, col_to_label):
    print(f"{model_name} model params", '\t\t', "\t".join(col_to_label.values()))
    for name, res_df in name_to_df.items():
        lr, bs, ep, step = get_params_from_name(name)
        short_name = params_to_shortname(lr, bs, ep, step)
        print(short_name, '\t', [f"{res_df[k].values[0]:.2f}"for k in col_to_label.keys()][0])

def metrics_columns_to_names(file_to_df, model_name, metrics, blue=False, detailed=False):
    plt.figure(figsize=(8, 4))
    colours = ['gold','orange','red', 'purple']
    if blue:
        colours = ['green','blue','cyan']
        plt.ylim(0.82, 0.9)
    if detailed:
        plt.grid(axis='both', linestyle='--', alpha=0.5, which='both')
        plt.ylim(0.15, 0.46)
        plt.yticks(np.arange(0.15, 0.46, 0.01))
        plt.y.set_tick_params(which='minor', labelleft=False)
    bar_width = 0.2
    num_models = len(file_to_df)
    x = np.arange(num_models)

    for idx, metric in enumerate(metrics):
        values = []
        for df in file_to_df.values():
            values.append(df[metric].values[0])
        plt.bar(x + idx*bar_width, values, width=bar_width, label=metric.replace('test_', '').split('/')[-1], color=colours[idx])
    
    plt.title(f"{model_name} Metrics on Test Set")
    plt.xlabel('Models')
    exp_names = []
    for k in file_to_df.keys():
        lr, bs, ep, step = get_params_from_name(k)
        exp_names.append(f"lr:{lr} bs:{bs} ep:{ep} st:{step}") 
    plt.xticks(x + bar_width, labels=exp_names, rotation=70, ha='right')
    plt.ylabel('Metric scores')
    plt.legend(loc='lower right')
    # plt.tight_layout()
    plt.show()

def plot_loss(df_dict: dict[str, pd.DataFrame]):
    # training vs evaluation loss for each experiment
    #get keys
    lrs = set()
    batch_sizes = set()
    max_epochs = set()

    file_names = df_dict.keys()
    for name in file_names:
        lr, bs, maxep, _ = get_params_from_name(name)
        lrs.add(lr)
        batch_sizes.add(bs)
        max_epochs.add(maxep)
    
    lrs = sorted(list(lrs))
    batch_sizes = sorted(list(batch_sizes))
    max_epochs = sorted(list(max_epochs))

    figure, axes = plt.subplots(len(batch_sizes), 1, figsize=(10, 8), sharex=True)

    for name, df in df_dict.items():
        lr, bs, maxep, _ = get_params_from_name(name)
        if maxep == 6:
            ax = axes[batch_sizes.index(bs)]
            ax.plot(df["epoch"], np.log10(df["loss"] + 1), label=f"Train, lr {lr}", linestyle='--', marker='o')
            ax.plot(df["epoch"], np.log10(df["eval_loss"] + 1), label=f"Validation, lr {lr}", marker='s')
            # ax.set_yscale('symlog', linthresh=1e-5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Log10 loss offset 1")
            ax.legend(loc='right')
            # ax.set_ylim(0, 3)
            ax.set_title(f"Batch size {bs}")
    figure.tight_layout(pad=1)
    plt.show()

def shorten_metric(full_metric_name):
    return full_metric_name.split('/')[-1].replace('test_', '').replace('precision', 'pr').replace('recall', 're').replace('bertscore', 'bert')

def best_scores_table(files_to_df, metrics):
    print("Model", '\t\t\t\t', "\t".join([shorten_metric(m) for m in metrics]))
    for name, df in files_to_df.items():
        lr, bs, ep, step = get_params_from_name(name)
        short_name = params_to_shortname(lr, bs, ep, step)
        print(short_name, '\t', "\t".join([f"{df[k].values[0]:.4f}" for k in metrics]))