"""Tools for visualization of time-series data.

Depends on matplotlib


"""

from nitime import timeseries as ts
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
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

    #Make sure that time displays on the x axis with the units you want:
    conv_fac = time_series.time._conversion_factor
    this_time = time_series.time/float(conv_fac)
    ax.plot(this_time,time_series.data.T)
        
    if xlabel is None:
        ax.set_xlabel('Time (%s)' %time_series.time_unit) 
    else:
        ax.set_xlabel(xlabel)

    
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    return fig


def matshow_roi(m,roi_names=None,fig=None,x_tick_rot=90,size=None):
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
    plt.matshow(m,fignum=fig.number)

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
 
    return fig

def plot_xcorr(xc,ij,fig=None,line_labels=None,xticks=None,yticks=None,xlabel=None,ylabel=None):

    """ Visualize the cross-correlation function"""
   
    if fig is None:
        fig=plt.figure()

    if not fig.get_axes():
        ax = fig.add_subplot(1,1,1)
    else:
        ax = fig.get_axes()[axis]

    if line_labels is not None:
        #Reverse the order, so that pop() works:
        line_labels.reverse()
        this_labels = line_labels


    #Make sure that time displays on the x axis with the units you want:
    conv_fac = xc.time._conversion_factor
    this_time = xc.time/float(conv_fac)
    
    for (i,j) in ij:
        if this_labels is not None:
            #Use pop() to get the first one and remove it:
            ax.plot(this_time,xc[i,j].squeeze(),label=this_labels.pop())
        else:
            ax.plot(this_time,xc[i,j].squeeze())
        
    ax.set_xlabel('Time(sec)')
    ax.set_ylabel('Correlation(normalized)')

    if xlabel is None:
        #Make sure that time displays on the x axis with the units you want:
        conv_fac = xc.time._conversion_factor
        time_label = xc.time/float(conv_fac)
        ax.set_xlabel('Time (%s)' %xc.time_unit) 
    else:
        time_label = xlabel
        ax.set_xlabel(xlabel)

    if line_labels is not None:
        plt.legend()

    if ylabel is None:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel(ylabel)
    
    return fig
    
    

