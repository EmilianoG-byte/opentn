"""
A module containing utility functions
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

def set_discrete_labels(labels:list[str], ax=None, rotation=65):
    "must be called after plotting the data (e.g. right before `plt.legend()`)"
    # NOTE: it does not work if we used an x array when plotting
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_ticks(np.arange(len(labels)))
    ax.xaxis.set_ticklabels(labels, rotation=rotation)

def set_integer_labels(ax=None):
    "must be called after plotting the data (e.g. right before `plt.legend()`)"
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))