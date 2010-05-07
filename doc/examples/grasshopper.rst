=====================================
 Auditory processing in grasshoppers
=====================================

Two data files are used in this example. The first contains the times of action
potentials ('spikes'), recorded intra-cellularly from primary auditory
receptors in the grasshopper *Locusta Migratoria*. The other data file contains
the stimulus that was played during the recording. Briefly, the stimulus played
was a pure-tone in the cell's preferred frequency amplitude modulated by
Gaussian white-noise, up to a cut-off frequency (200 Hz in this case, for
details on the experimental procedures and the stimulus see [Rokem2006]_). This
data is available on the `CRCNS data sharing web-site <http://crcns.org/>`_.

In the following code-snippet, we demonstrate the calculation of the
spike-triggered average (STA). This is the average of the stimulus wave-form
preceding the emission of a spike in the neuron and can be thought of as the
stimulus 'preferred' by this neuron.

.. plot:: examples/grasshopper1.py
   :include-source:

The code example makes use of :class:`EventRelatedAnalyzer` and, in particular,
of the :meth:`EventRelatedAnalyzer.eta` method of this object. This method gets
evaluated as an instance of the :class:`TimeSeries` class. In addition,
vizualization, using :func:`viz.plot_tseries` is demonstrated. This function
plots the values of the :attr:`TimeSeries.data`, with the appropriate
time-units and their values on the x axis. 

In the following example, a second channel has been added to both the stimulus
and the spike-train time-series. This is the response of the same cell, to a
different stimulus, in which the frequency modulation has a higher frequency
cut-off (800 Hz). The :class:`EventRelatedAnalyzer` simply calculates the STA
for both of these. Likewise, the plotting function :func:`viz.plot_tseries`
simply adds another line in another color for this second channel. 

.. plot:: examples/grasshopper2.py
   :include-source:

   
.. [Rokem2006] Ariel Rokem, Sebastian Watzl, Tim Gollisch, Martin Stemmler,
  Andreas V M Herz and Ines Samengo (2006). Spike-timing precision underlies
   the coding efficiency of auditory receptor neurons. J Neurophysiol, 95:
   2541--52
