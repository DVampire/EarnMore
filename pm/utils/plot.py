import numpy as np
import plotly.io as pio
import os
import json
import pandas as pd
import seaborn as sns
from copy import deepcopy
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from glob import glob

def plot_figures(fig_list, output_path):
    pio.write_image(fig_list[0], output_path, format='pdf')

def plot_metrics(exp_path, frac = 0.1, topk = 5):
    train_stats = ["train_obj_critics", "train_obj_actors"]
    val_stats = [[f"val_ARR%_env{i}", f"val_SR_env{i}", f"val_CR_env{i}", f"val_MDD%_env{i}", f"val_VOL_env{i}", f"val_DD_env{i}", f"val_SOR_env{i}"] for i in range(4)]
    val_stats = sum(val_stats, [])

    test_stats = [[f"test_ARR%_env{i}", f"test_SR_env{i}", f"test_CR_env{i}", f"test_MDD%_env{i}", f"test_VOL_env{i}", f"test_DD_env{i}", f"test_SOR_env{i}"] for i in range(4)]
    test_stats = sum(test_stats, [])

    df = pd.DataFrame(columns=["episode"] + train_stats + val_stats + test_stats)

    train_log_path = os.path.join(exp_path, "train_log.txt")
    with open(train_log_path, "r") as f:
        for line in f.readlines():
            stat = json.loads(line)

            stat = {k:float(v[0]) for k,v in stat.items()}

            df = df.append(stat, ignore_index=True)

    df = df.drop_duplicates(subset=["episode"], keep="last")
    df = df.dropna(axis=1)

    # compute topk of df "ARR%"
    topk_df = deepcopy(df)
    topk_df["ARR%"] = topk_df[f"val_ARR%_env0"]
    topk_df = topk_df.sort_values(by=["ARR%"], ascending=False)
    topk_df = topk_df.iloc[:topk]
    topk_df.to_csv(os.path.join(exp_path, "topk.csv"), index=False)

    plot_combine_val_test(exp_path)
    #
    # cols = 2
    # rows = int(np.ceil((len(val_stats) + len(train_stats)) / cols))
    # fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 8))
    #
    # colors = ["#ADD8E6", "#FFA07A"]
    #
    # for i in range(len(val_stats)):
    #     ax = axes[i // cols, i % cols]
    #     ax.set_title(val_stats[i].split("_")[1])
    #
    #     sns.scatterplot(data=df[[val_stats[i], test_stats[i]]], palette=colors, ax=ax)
    #
    #     smoothed_val = lowess(df[val_stats[i]].values, df["episode"].values, frac=frac)
    #     smoothed_test = lowess(df[test_stats[i]].values, df["episode"].values, frac=frac)
    #
    #     df[val_stats[i]] = smoothed_val[:, 1]
    #     df[test_stats[i]] = smoothed_test[:, 1]
    #
    #     sns.lineplot(data=df[[val_stats[i], test_stats[i]]], ax=ax)
    #
    # for i in range(len(val_stats),len(val_stats) + len(train_stats)):
    #     colors = ["#ADD8E6"]
    #     ax = axes[i // cols, i % cols]
    #     ax.set_title("train")
    #
    #     train_index = i - len(val_stats)
    #     if train_stats[train_index] in df.columns:
    #         sns.scatterplot(data=df[[train_stats[train_index]]], palette=colors, ax=ax)
    #         smoothed = lowess(df[train_stats[train_index]].values, df["episode"].values, frac=frac)
    #         df[train_stats[train_index]] = smoothed[:, 1]
    #         sns.lineplot(data=df[[train_stats[train_index]]], ax=ax)
    #
    # plt.savefig(os.path.join(exp_path, "metrics.pdf"))
    # plt.close()

def plot_combine_val_test(exp_path):
    topk_path = os.path.join(exp_path, "topk.csv")
    topk_df = pd.read_csv(topk_path)

    data = {
        "val_ARR%": [],
        "val_SR": [],
        "val_CR": [],
        "val_MDD%": [],
        "val_VOL": [],
        "val_DD": [],
        "val_SOR": [],
        "test_ARR%": [],
        "test_SR": [],
        "test_CR": [],
        "test_MDD%": [],
        "test_VOL": [],
        "test_DD": [],
        "test_SOR": [],
    }

    for i in range(4):
        for k in data.keys():
            data[k].append(topk_df[f"{k}_env{i}"].iloc[0])
    df = pd.DataFrame(data=data, index=range(4))
    df.to_csv(os.path.join(exp_path, "top1_index.csv"), index=False)


if __name__ == '__main__':

    workdir = "/Users/wentaozhang/workspace/RA/PortfolioManagement/workdir"

    for path in glob(os.path.join(workdir, "sac*")):
        plot_metrics(path)
        plot_combine_val_test(path)