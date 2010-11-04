import numpy as np
import matplotlib.pyplot as plt
import scipy.stats.distributions as dist

def dB(x, out=None):
    if out is None:
        return 10 * np.log10(x)
    else:
        np.log10(x, out)
        np.multiply(out, 10, out)

def plot_estimate(f, sdf, sdf_ests, limits=None, elabels=()):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax_limits = 2*sdf.min(), 1.25*sdf.max()
    ax.plot(f, sdf, 'c', label='True S(f)')

    if not elabels:
        elabels = ('',) * len(sdf_ests)
    colors = 'bgkmy'
    for e, l, c in zip(sdf_ests, elabels, colors):
        ax.plot(f, e, color=c, linewidth=2, label=l)

    if limits is not None:
        ax.fill_between(f, limits[0], y2=limits[1], color=(1,0,0,.3),
                        alpha=0.5)

    ax.set_ylim(ax_limits)
    ax.figure.set_size_inches([8,6])
    ax.legend()


### Log-to-dB conversion factor ###
ln2db = dB(np.e)
