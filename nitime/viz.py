"""Tools for visualization of time-series data.

Depends on matplotlib.pyplot


""" 

from nitime import timeseries as ts
from matplotlib import pyplot as plt

def plot(time_series,fig=None):
    """plot a timeseries object

    Arguments
    ---------

    time_series: a nitime time-series object

    
    """  

    if fig is None:
        fig=plt.figure()

#    ax = fig.
    plt.plot(time_series.time,time_series.data.T)
        
##     for loc, spine in ax.spines.iteritems():
##         if loc in ['left','bottom']:
##              spine.set_position(('outward',10)) # outward by 10 points
##         elif loc in ['right','top']:
##             spine.set_color('none') # don't draw spine

##     ax.set_xticks([0,11,22,45,90])        
##     ax.set_xticklabels(['0','11','22','45','90'])
##     ax.xaxis.set_ticks_position('bottom')
##     ax.yaxis.set_ticks_position('left')
##     ax.set_xlim(0,95)

    return fig






