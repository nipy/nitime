from __future__ import print_function
import numpy as np
from scipy.special import factorial


def gamma_hrf(duration, A=1., tau=1.08, n=3, delta=2.05, Fs=1.0):
    r"""A gamma function hrf model, with two parameters, based on
    [Boynton1996]_


    Parameters
    ----------

    duration: float
        the length of the HRF (in the inverse units of the sampling rate)

    A: float
        a scaling factor, sets the max of the function, defaults to 1

    tau: float
        The time constant of the gamma function, defaults to 1.08

    n: int
        The phase delay of the gamma function, defaults to 3

    delta: float
        A pure delay, allowing for an additional delay from the onset of the
        time-series to the beginning of the gamma hrf, defaults to 2.05

    Fs: float
        The sampling rate, defaults to 1.0


    Returns
    -------

    h: the gamma function hrf, as a function of time

    Notes
    -----
    This is based on equation 3 in Boynton (1996):

    .. math::

        h(t) =
        \frac{(\frac{t-\delta}{\tau})^{(n-1)}
        e^{-(\frac{t-\delta}{\tau})}}{\tau(n-1)!}


    Geoffrey M. Boynton, Stephen A. Engel, Gary H. Glover and David J. Heeger
    (1996). Linear Systems Analysis of Functional Magnetic Resonance Imaging in
    Human V1. J Neurosci 16: 4207-4221

    """
    # XXX Maybe change to take out the time (Fs, duration, etc) from this and
    # instead implement this in units of sampling interval (pushing the time
    # aspect to the higher level)?
    if type(n) is not int:
        print(('gamma_hrf received unusual input, converting n from %s to %i'
               % (str(n), int(n))))

        n = int(n)

    #Prevent negative delta values:
    if delta < 0:
        raise ValueError('in gamma_hrf, delta cannot be smaller than 0')

    #Prevent cases in which the delta is larger than the entire hrf:
    if delta > duration:
        e_s = 'in gamma_hrf, delta cannot be larger than the duration'
        raise ValueError(e_s)

    t_max = duration - delta

    t = np.hstack([np.zeros((delta * Fs)), np.linspace(0, t_max, t_max * Fs)])

    t_tau = t / tau

    h = (t_tau ** (n - 1) * np.exp(-1 * (t_tau)) /
         (tau * factorial(n - 1)))

    return A * h / max(h)


def polonsky_hrf(A, B, tau1, f1, tau2, f2, t_max, Fs=1.0):
    r""" HRF based on Polonsky (2000):

    .. math::

       H(t) = exp(\frac{-t}{\tau_1}) sin(2\cdot\pi f_1 \cdot t) -a\cdot
       exp(-\frac{t}{\tau_2})*sin(2\pi f_2 t)

    Alex Polonsky, Randolph Blake, Jochen Braun and David J. Heeger
    (2000). Neuronal activity in human primary visual cortex correlates with
    perception during binocular rivalry. Nature Neuroscience 3: 1153-1159

    """
    sampling_interval = 1 / float(Fs)

    t = np.arange(0, t_max, sampling_interval)

    h = (np.exp(-t / tau1) * np.sin(2 * np.pi * f1 * t) -
            (B * np.exp(-t / tau2) * np.sin(2 * np.pi * f2 * t)))

    return A * h / max(h)
