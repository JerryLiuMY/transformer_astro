from global_settings import RAW_FOLDER
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
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
    catalog = pd.read_csv(os.path.join(RAW_FOLDER, dataset_name, 'catalog.csv'), index_col=0)
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
    catalog = pd.read_csv(os.path.join(RAW_FOLDER, dataset_name, 'catalog.csv'), index_col=0)
    cats, paths = list(catalog['Class']), list(catalog['Path'])
    uniq_cats = sorted(list(set(cats))); N = len(uniq_cats)

    min_mag, max_mag = [], []
    for uniq_cat in uniq_cats:
        indices = [idx for idx, cat in enumerate(cats) if cat == uniq_cat]
        cats_, paths_ = [cats[i] for i in indices], [paths[i] for i in indices]
        min_mag_, max_mag_ = [], []
        for cat_, path_ in list(zip(cats_, paths_)):
            data_df = pd.read_csv(os.path.join(RAW_FOLDER, 'ASAS', path_))
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


def positional():
    sin_mtx, cos_mtx, alt_mtx = _matrix()
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(3, 2); ax1 = plt.subplot(gs[0:2, 0])
    ax2 = plt.subplot(gs[0:2, 1]); ax3 = plt.subplot(gs[2:, :])
    sns.heatmap(sin_mtx, ax=ax1)
    sns.heatmap(cos_mtx, ax=ax2)
    sns.heatmap(alt_mtx, ax=ax3, cbar=False)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Embedding Position')
        ax.set_ylabel('Time-steps')
    plt.tight_layout()
    plt.show()

    return fig


def product():
    _, _, alt_mtx = _matrix()
    fig, ax = plt.subplots(figsize=(12, 10))
    score = np.dot(alt_mtx, alt_mtx.T)
    sns.heatmap(score, ax=ax)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Steps')
    plt.tight_layout()
    plt.show()

    return fig


def _matrix():
    seq, dim = 50, 300
    sin_mtx = np.zeros((seq, dim))
    cos_mtx = np.zeros((seq, dim))
    alt_mtx = np.zeros((seq, dim))

    for t in range(seq):
        for i in range(dim):
            sin = np.sin(t * (1 / (10000 ** (i / dim))))
            cos = np.cos(t * (1 / (10000 ** ((i - 1) / dim))))
            alt = np.sin(t * (1 / (10000 ** (i / dim)))) if i % 2 == 0 else np.cos(t * (1 / (10000 ** ((i - 1) / dim))))
            sin_mtx[t, i], cos_mtx[t, i], alt_mtx[t, i] = sin, cos, alt

    return sin_mtx, cos_mtx, alt_mtx
