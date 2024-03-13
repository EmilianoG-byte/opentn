"""
A module containing utility functions
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


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

def plot_cost_function_vs_iter(cost_list:list[float], labels:list[str]=None):
    "plot the cost function of each array in the list agains the number of iterations"
    cost_list = np.array(cost_list)
    if cost_list.ndim == 1: # takes care if cost_list is just one single cost function list (instead of list of lists)
        cost_list = [cost_list]
    if labels is  None:
        labels = [f"method {i}" for i in range(len(cost_list))]
    plt.figure(dpi=200)
    plt.ylabel('cost function')
    plt.xlabel('iterations')
    for cost, label in zip(cost_list, labels):
        plt.plot(cost, '--', label=label)
    plt.legend()