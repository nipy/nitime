import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as sig

def winspect(win, f,name=None):
    """Inspect a window by showing it and its spectrum"""
    npts = len(win)
    ax1,ax2 = f.add_subplot(1,2,1),f.add_subplot(1,2,2)
    ax1.plot(win)
    ax1.set_xlabel('Time')
    if name:
        tt = 'Window: %s' % name
    else:
        tt = 'Window'
    ax1.set_title(tt)
    ax1.set_xlim(0, npts)
    wf = np.fft.fft(win)
    ax2.plot(np.log(np.abs(np.fft.fftshift(wf).real)))
    ax2.axhline(0, color='k')
    ax2.axhline(-5, color='k')
    ax2.set_xlim(0, npts)
    ax2.set_title('Window spectrum')
    ax2.grid()


def kaiser_inspect(npts, f, beta):
    name = r'Kaiser, $\beta=%1.1f$' % beta
    winspect(sig.kaiser(npts, beta), f,name)

if __name__ == '__main__':

    f = plt.figure()
    # Window size
    npts = 128

    # Boxcar with zeroed out fraction
    b = sig.boxcar(npts)
    zfrac = 0.15
    zi = int(npts*zfrac)
    b[:zi] = b[-zi:] = 0
    name = 'Boxcar - zero fraction=%.2f' % zfrac
    winspect(b, f, name)

    # Hanning
    winspect(sig.hanning(npts), f, 'Hanning')

    # Various Kaiser windows
    kaiser_inspect(npts,f, 0.1)
#    kaiser_inspect(npts,f, 1)
#    kaiser_inspect(npts,f, 10)
#    kaiser_inspect(npts,f, 100)
