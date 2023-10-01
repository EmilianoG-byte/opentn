"""
A module containing utility functions
"""

import matplotlib.pyplot as plt
import numpy as np

def set_discrete_labels(labels:list[str], ax=None):
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_ticks(np.arange(len(labels)))
    ax.xaxis.set_ticklabels(labels, rotation=65)

