===============================
Multi-taper spectral estimation
===============================

The distribution of power in a signal, as a function of frequency, known as the
power spectrum (or PSD, for power spectral density) can be estimated using
variants of the discrete Fourier transform (DFT). The naive estimate of the
power spectrum, based on the values of the DFT estimated directly from the
signal, using the fast Fourier transform algorithm (FFT) is referred to as a
periodogram (see :func:`algorithms.periodogram`). This estimate suffers from
several problems [NR2007]_:

- Inefficiency: In most estimation problems, additional samples, or a denser
  sampling grid would usually lead to a better estimate (smaller variance of
  the estimate, given a constant level of noise). However, this is not the case
  for the periodogram. Even as we add more samples to our signal, or increase
  our sampling rate, our estimate at frequency $f_k$ does not improve. This is
  because of the effects these kinds of changes have on spectral
  estimates. Adding additional samples, will improve the frequency domain
  resolution of our estimate and sampling at a finer rate will change the
  Nyquist frequency, the highest frequency for which the spectrum can be
  estimated. Thus, these changes do not improve the estimate at frequency
  $f_k$.  

The inefficiency problem can be solved by treating different parts of the
signal as different samples from the same distribution, while assuming
stationarity of the signal. In this method, a shorter sliding window is applied
to different parts of the signal and the windowed spectrum is averaged from
these different samples. This is sometimes referred to as Welch's periodogram
[Welch1967]_ and it is the default method used in
:func:`algorithms.get_spectra` (with the hanning window as the window function
used and no overlap between the windows).  However, it leads to the following
problem:

- Spectral leakage and bias: Spectral leakage refers to the fact that the
  estimate of the spectrum at any given frequency bin is contaminated with the
  power from other frequency bands. This is a consequence of the fact that we
  always look at a time-limited signal. In the naive peridogram estimate all
  the samples within the time-limited signal are taken as they are (implicitly
  multiplied by 1) and all the samples outside of this time-limited signal are
  not taken at all (implicitly multiplied by 0). This is akin to what would
  happen if the signal were multiplied sample-by-sample with a 'boxcar' window,
  so called because the shape of this window is square, going from 0 to 1 over
  one sampling window. Multiplying the signal with a boxcar window in the
  time-domain is equivalent (due to the convolution theorem) to convolving it
  in the frequency domain with the spectrum of the boxcar window. The spectral
  leakage induced by this operation is demonstrated in the following example:

.. plot:: examples/boxcar_inspect.py
   :include-source:

   The figure on the left shows a boxcar window and the figure on the right
   shows the spectrum of the boxcar function (in dB units, relative to the
   frequency band of interest).  
   
These two problems can together be mitigated through the use of windowed
spectral estimates. The idea behind this method is that instead of using the
boxcar window, one can design windows whose effect on spectral leakage is less
deleterious. The following example demonstrates the spectral leakage for several
different windows (including the boxcar):

.. plot:: examples/window_compare.py
   :include-source:
 
As before, the left figure displays the windowing function in the temporal
domain and the figure on the left displays the attentuation of spectral leakage
in the other frequency bands in the spectrum. Notice that though different
windowing functions have different spectral attenuation profiles, trading off
attenuation of leakage from frequency bands near the frequency of interest
(narrow-band leakage) with leakage from faraway frequency bands (broad-band
leakage) they are all superior in both of these respects to the boxcar window
used in the naive periodogram. 

The inefficiency problem can be solved by treating different parts of the
signal as different samples from the same distribution. In this method, a
shorter sliding window is applied to different parts of the signal and the
windowed spectrum is averaged from these different samples. This is sometimes
referred to as Welch's periodogram [Welch1967]_ and it is the default method
used in :func:`algorithms.get_spectra` (with the hanning window as the window
function used and no overlap between the windows).

However, this approach trades off the reliability of the measurement without
leakage for the resolution of measurement, which is lost due to the smaller
window size. Another approach is to instead multiply the entire signal segment
by a taper function. Similar to the window functions, these functions start at
0 and end at 0. The entire signal is multiplied by a taper function. In
addition, these functions can be constructed to be orthogonal to each other,
constructing maximally independent samples at the length of the signal. As we
will see below, this makes this construction useful for statistical estimation
of the distribution of the spectrum.

Discrete prolate spheroidal sequences (also known as Slepian sequences)
[Slepian1978]_ are a class of taper functions which are constructed as a
solution to the problem of concentrating the spectrum to within a pre-specified
bandwidth. These tapers can be constructed using
:func:`algorithms.DPSS_windows`, but for the purpose of spectral estimation, it
is sufficient to specify the bandwidth desired as an input to the function, or
rely on the these tapers are orthogonal allows using the estimates based on
each window as a pseudo-indpendent sample for the purposes of statistical
estimation. Thus, in addition to estimating the spectrum itself, an estimate of
the confidence interval of the spectrum can be generated using a jack-knifing
procedure [Thomson2007]_.

Let us define the following:

| **simple sample estimate**
| :math:`\hat{\theta} = \dfrac{1}{n}\sum_i Y_i`

This is the estimate of the parameter estimate averaged from all the samples in
the distribution (all the tapered spectra).

| **leave-one-out measurement**
| :math:`\hat{\theta}_{-i} = \dfrac{1}{n-1}\sum_{k \neq i}Y_k`

This defines a group of estimates, where each sample is based on leaving one
measurement (one tapered spectrum) out.

| **pseudovalues**
| :math:`\hat{\theta}_i = n\hat{\theta} - (n-1)\hat{\theta}_{-i}`

The jackknifed esimator is computed as:

:math:`\tilde{\theta} = \dfrac{1}{n}\sum_i \hat{\theta}_i = n\hat{\theta} - \dfrac{n-1}{n}\sum_i \hat{\theta}_{-i}`

This estimator is known [Thomson2007]_ to be distributed about the true parameter \theta approximately as a Student's t distribution, with standard error defined as:

:math:`s^{2} = \dfrac{n-1}{n}\sum_i \left(\hat{\theta}_i - \tilde{\theta}\right)^{2}`


the power spectrum can be estimated, in addition to a confidence interval on
the values of the spectrum. In addition, if the 'adaptive' flag is set to True,
an iterative adaptive method is used in order to correct bias in the spectrum.


.. plot:: examples/multi_taper_sdf.py
   :include-source:


.. [NR2007] W.H. Press, S.A. Teukolsky, W.T Vetterling and B.P. Flannery (2007)
   	    Numerical Recipes: The Art of Scientific Computing. Cambridge:
   	    Cambridge University Press. 3rd Ed.

.. [Thomson2007] D.J. Thomson, Jackknifing Multitaper Spectrum Estimates, IEEE
   		 Signal Processing Magazine, 2007, pp. 20-30.

.. [Welch1967] P.D. Welch (1967), The use of the fast fourier transform for the
   	       estimation of power spectra: a method based on time averaging
   	       over short modified periodograms. IEEE Transcations on Audio and
   	       Electroacoustics.

.. [Slepian1978] Slepian, D. Prolate spheroidal wave functions, Fourier
		 analysis, and uncertainty V: The discrete case. Bell System
		 Technical Journal, Volume 57 (1978), 1371430
