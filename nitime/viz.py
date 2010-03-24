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


def matshow_roi(m,roi_names=None,fig=None,x_tick_rot=90,size=None,
                cmap=plt.cm.PuBuGn):
    """This is the typical format to show a bivariate quantity (such as
    correlation or coherency between two different ROIs""" 
    N = len(roi_names)
    ind = np.arange(N)  # the evenly spaced plot indices
    
    def roi_formatter(x,pos=None):
        thisind = np.clip(int(x), 0, N-1)
        return roi_names[thisind]

    if fig is None:
        fig=plt.figure()

    if size is not None:
        fig.set_figwidth(size[0])
        fig.set_figheight(size[1])

    #The call to matshow produces the matrix plot:
    plt.matshow(m,fignum=fig.number,cmap=cmap)
    
    #Formatting:
    ax = fig.axes[0]
    ax.set_xticks(np.arange(len(roi_names)))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(roi_formatter))
    fig.autofmt_xdate(rotation=x_tick_rot)
    ax.set_yticks(np.arange(len(roi_names)))
    ax.set_yticklabels(roi_names)
    ax.set_ybound([-0.5,len(roi_names)-0.5])

    #Make the tick-marks invisible:
    for line in ax.xaxis.get_ticklines():
        line.set_markeredgewidth(0)

    for line in ax.yaxis.get_ticklines():
      line.set_markeredgewidth(0)

    plt.colorbar()

    return fig




