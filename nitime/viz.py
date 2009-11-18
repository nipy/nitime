"""Tools for visualization of time-series data.

Depends on matplotlib.pyplot


""" 

from nitime import timeseries as ts
from matplotlib import pyplot as plt

def plot(T,figure=None):
    """plot a timeseries object """  

    if figure is None:
        figure=plt.figure()
        ax = plt.






