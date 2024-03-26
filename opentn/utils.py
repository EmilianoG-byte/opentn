"""
A module containing utility functions
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, AutoLocator, MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from typing import Union

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


def plot_pretty(ydatas:list[list[float]],
                labels:list[str],
                ylabel:str,
                xlabel:str,
                title:str=None,
                xdatas:list[list[float]]=None,
                integers:bool=False,
                legend_out:bool=True,
                use_semilogy:bool=False,
                optimize:bool=True,
                marker_step:int=5,
                loglog:bool=False,
                idx_main:int=2,
                horizontal_value:float=None,
                comparison_value:Union[float, list, np.ndarray]=None,
                comparison_label:str="",
                inset:bool=False,
                inset_idx:int=10,
                inset_label:bool=False
                ):
    """
    Utility functino to plot prettily a list of lists of data
    """

    # Define your color palette

    color_palette = ['#2EC4B6', '#B9C7DF', '#E71D36', '#88D498', '#FF8C42', '#AEA0CF']
    color_palette.insert(idx_main, '#294C60')

    # Define marker styles
    marker_styles = ['o', 's', '^', 'D', 'v', '>', '<', 'x', '+', '*']

    # Increase figure size
    plt.figure(figsize=(8, 6), dpi=200)



    if use_semilogy:
        plot_function = plt.semilogy
    elif loglog:
        plot_function = plt.loglog
    else:
        plot_function = plt.plot

    if horizontal_value:
        plt.axhline(y=horizontal_value, color='#E71D36', linestyle='--', alpha=0.65)
    if isinstance(comparison_value, (float, int, )):
        plt.axhline(y=comparison_value, color='gray', linestyle='--', label=comparison_label)
    elif isinstance(comparison_value, tuple):
        assert len(comparison_value)==2, "comparison should be a tuple with x and y data"
        plot_function(comparison_value[0], comparison_value[1], '--', color='gray', label=comparison_label)
    elif isinstance(comparison_value, (list, np.ndarray)):
        plot_function(range(1, len(comparison_value)+1), comparison_value, '--', color='gray', label=comparison_label)

    # Plot the data with custom colors, line styles, and markers
    for i, ydata in enumerate(ydatas):
        if xdatas is None:
            xdata = range(1, len(ydata)+1)
        else:
            xdata = xdatas[i]
        marker_style = marker_styles[i % len(marker_styles)]
        if optimize:
            if len(ydata) > 25:
                ydata = ydata[::marker_step]
                xdata = xdata[::marker_step]

        plot_function(xdata, ydata, '-'+ marker_style, color=color_palette[i], label=labels[i], linewidth=2)

     # Add title and labels
    if title:
        plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    # Set y-axis tick formatter and locator
    if not use_semilogy and not loglog:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 3))  # Set the exponent range
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.gca().yaxis.set_major_locator(AutoLocator())


    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))  # Set the exponent range
    plt.gca().xaxis.set_major_formatter(formatter)
    if integers:
        # Set x-axis ticks to integers
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # see: https://stackoverflow.com/questions/21001088/how-to-add-different-graphs-as-an-inset-in-another-python-graph

    if inset:
        main_axis = plt.gca()
        min_value = 100
        max_value = 0
        ax_inset = inset_axes(main_axis, width="40%", height="40%", loc='upper right')
        for i, ydata in enumerate(ydatas):
            if xdatas is None:
                xdata = range(1, len(ydata)+1)
            else:
                xdata = xdatas[i]

            min_value = min(ydata[inset_idx:][-1], min_value)
            max_value = max(ydata[inset_idx:][0], max_value)
            if use_semilogy:
                ax_inset.semilogy(xdata[inset_idx:], ydata[inset_idx:], '-'+ marker_style, color=color_palette[i], label=labels[i], linewidth=2)
            else:
                ax_inset.plot(xdata[inset_idx:], ydata[inset_idx:], '-'+ marker_style, color=color_palette[i], label=labels[i], linewidth=2)

        # ax_inset.set_xlim(xdata[inset_x], xdata[-1])
        # ax_inset.set_ylim(min(ydata[inset_y:]), max(ydata[inset_y:]))  # Adjusted y-axis limits
        ax_inset.set_xlabel(xlabel)

        if inset_label:
        # Add x 10^{-5} to the label
            ax_inset.set_yticks([min_value, max_value])
            ax_inset.set_yticklabels([f"{min_value}"[:3], f"{max_value}"[:3]])

        # Remove tick labels from other ticks
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html
        # from
        ax_inset.tick_params(axis='y', which='both',labelleft=inset_label, left=True, right=False)

        # Mark the inset in the main plot
        # https://stackoverflow.com/questions/13583153/how-to-zoomed-a-portion-of-image-and-insert-in-the-same-plot-in-matplotlib
        # from
        mark_inset(main_axis, ax_inset, loc1=3, loc2=4, fc="none", ec="0.5")

        height_legend = 0.75
    else:
        height_legend = 0.825

    # Add legend with larger font size and place it to the right of the plot
    if legend_out:
        plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, height_legend))
    else:
        plt.legend(fontsize=12, loc='best')
    # Return the figure
    return plt.gcf()