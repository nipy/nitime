"""Tools for visualization of time-series data.

Depends on matplotlib.pyplot


"""

from nitime import timeseries as ts
from matplotlib import pyplot as plt

def plot_tseries(time_series,fig=None,axis=0,
                 xticks=None,xunits=None,yticks=None,yunits=None):
    """plot a timeseries object

    Arguments
    ---------

    time_series: a nitime time-series object

    fig: a figure handle, opens a new figure if None

    subplot: an axis number (if there are several in the figure to be opened),
        defaults to 0.
        
    xticks:

    yticks: 
    
    """  

    if fig is None:
        fig=plt.figure()

    if not fig.get_axes():
        ax = fig.add_subplot(1,1,1)
    else:
        ax = fig.get_axes()[axis]
        
    ax.plot(time_series.time,time_series.data.T)
    
##     ax.set_xticks([])        
##     ax.set_xticklabels([])
##     ax.xaxis.set_ticks_position('bottom')
##     ax.yaxis.set_ticks_position('left')
##     ax.set_xlim()

    return fig






