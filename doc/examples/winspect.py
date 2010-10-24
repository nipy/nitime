import numpy as np
import matplotlib.pyplot as plt

def winspect(win, f,name=None):
    """Inspect a window by showing it and its spectrum"""
    npts = len(win)
    ax1,ax2 = f.add_subplot(1,2,1),f.add_subplot(1,2,2)
    ax1.plot(win)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Window amplitude')
    ax1.set_ylim(-0.1,1.1)
    ax1.set_xlim(0, npts)
    wf = np.fft.fft(win)
    ax1.set_xticks(np.arange(npts/8.,npts,npts/8.))
    toplot = np.abs(np.fft.fftshift(wf).real)
    toplot /= np.max(toplot)
    toplot = np.log(toplot)
    ax2.plot(toplot,label=name)
    ax2.set_xlim(0, npts)
    ax2.set_xticks(np.arange(npts/8.,npts,npts/8.))
    ax2.set_xticklabels(np.arange((-1/2.+1/8.),1/2.,1/8.))
    ax2.set_xlabel('Relative frequency')
    ax2.set_ylabel('Relative attenuation (log scale)')
    ax2.grid()
    ax2.legend(loc=4)
    f.set_size_inches([12,8])

