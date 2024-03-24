"""
A module containing utility functions
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, AutoLocator, MultipleLocator, MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

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
                comparison_value:float=None,
                inset:bool=False,
                inset_idx:int=10,
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

    if comparison_value is not None:
        plt.axhline(y=comparison_value, color='gray', linestyle='--')

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
        if use_semilogy:
            plot_function = plt.semilogy
        elif loglog:
            plot_function = plt.loglog
        else:
            plot_function = plt.plot
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

    # Set x-axis tick formatter
    formatter = ScalarFormatter(useMathText=False)
    formatter.set_powerlimits((-2, 3))  # Set the exponent range
    plt.gca().xaxis.set_major_formatter(formatter)
    if integers:
        # Set x-axis ticks to integers
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

     # Add legend with larger font size and place it to the right of the plot
    if legend_out:
        plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.85))
    else:
        plt.legend(fontsize=12)

    main_axis = plt.gca()
    # Add inset if requested
    if inset:
        yticks = [ydata[inset_idx], ydata[-1]]
        ax_inset = inset_axes(main_axis, width="40%", height="40%", loc='upper right', axes_kwargs={"yticks":yticks})
        for i, ydata in enumerate(ydatas):
            if xdatas is None:
                xdata = range(1, len(ydata)+1)
            else:
                xdata = xdatas[i]
            ax_inset.semilogy(xdata[inset_idx:], ydata[inset_idx:], '-'+ marker_style, color=color_palette[i], label=labels[i], linewidth=2)

        # ax_inset.set_xlim(xdata[inset_x], xdata[-1])
        # ax_inset.set_ylim(min(ydata[inset_y:]), max(ydata[inset_y:]))  # Adjusted y-axis limits
        ax_inset.set_xlabel(xlabel)
        # Set y-axis tick locator for inset plot
        # ax_inset.tick_params(labelleft=False, labelbottom=False)
        # ax_inset.set_yticks([])
        # ax_inset.yaxis.set_major_locator(MultipleLocator(0.1))  # Set the interval between major ticks
         # Remove old ticks from the inset plot
        # ax_inset.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax_inset.tick_params(axis='y', which='both',labelleft=True, left=True, right=False)
        # ax_inset.set_yticks(yticks, minor=False)



        mark_inset(main_axis, ax_inset, loc1=3, loc2=4, fc="none", ec="0.5")
    # Return the figure
    return plt.gcf()