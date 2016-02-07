import numpy as np


def boxcar_filter(time_series, lb=0, ub=0.5, n_iterations=2):
    """
    Filters data into a frequency range.

    For each of the two bounds, a low-passed version is created by convolving
    with a box-car and then the low-passed version for the upper bound is added
    to the low-passed version for the lower bound subtracted from the signal,
    resulting in a band-passed version

    Parameters
    ----------

    time_series: float array
       the signal
    ub : float, optional
      The cut-off frequency for the low-pass filtering as a proportion of the
      sampling rate. Default to 0.5 (Nyquist)
    lb : float, optional
      The cut-off frequency for the high-pass filtering as a proportion of the
      sampling rate. Default to 0
    n_iterations: int, optional
      how many rounds of smoothing to do. Default to 2.

    Returns
    -------
    float array:
      The signal, filtered
    """

    n = time_series.shape[-1]

    len_boxcar_ub = np.ceil(1 / (2.0 * ub))
    boxcar_ub = np.empty(int(len_boxcar_ub))
    boxcar_ub.fill(1.0 / len_boxcar_ub)
    boxcar_ones_ub = np.ones_like(boxcar_ub)

    if lb == 0:
        lb = None
    else:
        len_boxcar_lb = np.ceil(1 / (2.0 * lb))
        boxcar_lb = np.empty(int(len_boxcar_lb))
        boxcar_lb.fill(1.0 / len_boxcar_lb)
        boxcar_ones_lb = np.ones_like(boxcar_lb)

    #If the time_series is a 1-d, we add a dimension, so that we can iterate
    #over 2-d inputs:
    if len(time_series.shape) == 1:
        time_series = np.array([time_series])
    for i in range(time_series.shape[0]):
        if ub:
            # Start by applying a low-pass to the signal.  Pad the signal on
            # each side with the initial and terminal signal value:
            pad_s = np.hstack((boxcar_ones_ub *
                               time_series[i, 0], time_series[i]))
            pad_s = np.hstack((pad_s, boxcar_ones_ub * time_series[i, -1]))

            # Filter operation is a convolution with the box-car(iterate,
            # n_iterations times over this operation):
            for iteration in range(n_iterations):
                conv_s = np.convolve(pad_s, boxcar_ub)

            # Extract the low pass signal by excising the central
            # len(time_series) points:
            time_series[i] = conv_s[conv_s.shape[-1] // 2 -
                                    int(np.floor(n / 2.)):
                                    conv_s.shape[-1] // 2 +
                                    int(np.ceil(n / 2.))]

        # Now, if there is a high-pass, do the same, but in the end subtract
        # out the low-passed signal:
        if lb:
            pad_s = np.hstack((boxcar_ones_lb *
                               time_series[i, 0], time_series[i]))
            pad_s = np.hstack((pad_s, boxcar_ones_lb * time_series[i, -1]))

            #Filter operation is a convolution with the box-car(iterate,
            #n_iterations times over this operation):
            for iteration in range(n_iterations):
                conv_s = np.convolve(pad_s, boxcar_lb)

            #Extract the low pass signal by excising the central
            #len(time_series) points:
            s_lp = conv_s[conv_s.shape[-1] // 2 - int(np.floor(n / 2.)):
                          conv_s.shape[-1] // 2 + int(np.ceil(n / 2.))]

            #Extract the high pass signal simply by subtracting the high pass
            #signal from the original signal:
            time_series[i] = time_series[i] - s_lp + np.mean(s_lp)  # add mean
            #to make sure that there are no negative values. This also seems to
            #make sure that the mean of the signal (in % signal change) is
            #close to 0

    return time_series.squeeze()
