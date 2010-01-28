"""Tools for visualization of time-series data.

Depends on matplotlib.pyplot


"""

from nitime import timeseries as ts
from matplotlib import pyplot as plt
import numpy as np

def plot_tseries(time_series,fig=None,axis=0,
                 xticks=None,xunits=None,yticks=None,yunits=None,xlabel=None,
                 ylabel=None):

    """plot a timeseries object

    Arguments
    ---------

    time_series: a nitime time-series object

    fig: a figure handle, opens a new figure if None

    subplot: an axis number (if there are several in the figure to be opened),
        defaults to 0.
        
    xticks:

    yticks: 

    xlabel:

    ylabel:

    
    """  

    if fig is None:
        fig=plt.figure()

    if not fig.get_axes():
        ax = fig.add_subplot(1,1,1)
    else:
        ax = fig.get_axes()[axis]

    if xlabel is None:
        #Make sure that time displays on the x axis with the units you want:
        conv_fac = time_series.time._conversion_factor
        time_label = time_series.time/float(conv_fac)
        ax.plot(time_label,time_series.data.T)
        ax.set_xlabel('Time (%s)' %time_series.time_unit) 
    else:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    return fig






