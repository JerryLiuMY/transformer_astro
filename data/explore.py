import matplotlib.gridspec as gridspec
from tools.utils import load_catalog
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
plt.rcParams['axes.labelsize'] = 17
plt.rcParams['axes.titlesize'] = 17
plt.rc('xtick',labelsize=13)
plt.rc('ytick',labelsize=13)


def explore(dataset_name):
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
