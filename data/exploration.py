import matplotlib.gridspec as gridspec
from tools.utils import load_catalog
import matplotlib.pyplot as plt
from global_settings import DATA_FOLDER
import numpy as np
import pandas as pd
import seaborn as sns
import os

sns.set()
plt.rcParams['axes.labelsize'] = 17
plt.rcParams['axes.titlesize'] = 17
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)


def distribution(dataset_name):
    catalog = load_catalog(dataset_name, 'whole')
    fig = plt.figure(figsize=(16, 9)); gs = gridspec.GridSpec(3, 2)
    ax1 = plt.subplot(gs[0:2, 0]); ax3 = plt.subplot(gs[2, :])
    ax2 = plt.subplot(gs[0:2, 1]); ax2_ = ax2.twinx(); ax2_.set_yticks([])

    sns.countplot(catalog['Class'], ax=ax1)
    sns.distplot(catalog['N'], bins=25, hist=True, kde=False, ax=ax2)
    sns.distplot(catalog['N'], hist=False, ax=ax2_, label='All')
    for _ in list(set(catalog['Class'])):
        sns.distplot(catalog[catalog['Class'] == _]['N'], bins=25, hist=False, kde=True, ax=ax3, label=_)
    plt.tight_layout()

    return fig


def minmax(dataset_name):
    catalog = load_catalog(dataset_name, 'whole')
    cats, paths = list(catalog['Class']), list(catalog['Path'])
    uniq_cats = sorted(list(set(cats))); N = len(uniq_cats)

    min_mag, max_mag = [], []
    for uniq_cat in uniq_cats:
        indices = [idx for idx, cat in enumerate(cats) if cat == uniq_cat]
        cats_, paths_ = [cats[i] for i in indices], [paths[i] for i in indices]
        min_mag_, max_mag_ = [], []
        for cat_, path_ in list(zip(cats_, paths_)):
            data_df = pd.read_csv(os.path.join(DATA_FOLDER, 'ASAS', path_))
            min_mag_.append(min(list(data_df['mag']))); max_mag_.append(max(list(data_df['mag'])))
        min_mag.append(round(min(min_mag_), 2)), max_mag.append(round(max(max_mag_), 2))

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(np.arange(N), max_mag, width=0.35, label='max')
    rects2 = ax.bar(np.arange(N), min_mag, width=0.35, label='min')
    plt.legend(); plt.ylabel('mag')
    plt.xticks(np.arange(N), uniq_cats)
    plt.yticks(np.arange(0, 21, 10))
    _annotate(ax, rects1, color='black')
    _annotate(ax, rects2, color='white')
    fig.tight_layout()

    return fig


def _annotate(ax, rects, color):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(format(height, '.2f'), xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', color=color)
