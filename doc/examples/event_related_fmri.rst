==================
Event-related fMRI
==================

The following example is taken from an experiment in which a subject was
viewing a motion stimulus, while fMRI BOLD was recorded. The time-series in
this data set were extracted from motion-sensitive voxels near area MT (a
region containing motion-sensitive cells) in this subject's brain. 6 different
kinds of trials could occur in this experiment (designating different
directions and locations of motion). The following example shows the extraction
of the time-dependent responses of the voxels in this region to the different
stimuli.  

.. plot:: examples/event_related_fmri.py
   :include-source:

The example uses the EventRelated analyzer (also used in the grasshopper
example), but now, instead of providing an :class:`Events` object as input,
another :class:`TimeSeries` object is provided, containing an equivalent
time-series with the same dimensions as the time-series on which the analysis
is done, with '0' wherever no event of interest occured and an integer wherever
an even of interest occured (sequential different integers for the different
kinds of events).

Two different methods of the :attr:`E.eta` refers to the event-triggered
average of the activity and :attr:`E.ets` refers to the event-triggered
standard error of the mean (where the degrees of freedom are set by the number
of trials). Note that you can also extract the event-triggered data itself as a
list, by referring instead to :attr:`E.et_data`. 

In the following example two alternative approaches are taken to calculating
the event-related activity. The first is based on the finite impulse-response
model (see [Burock2000]_ for details) and the other is based on a
cross-correlation method (thanks to Lavi Secundo for providing a previous
implementation of this idea):

.. .. plot:: examples/event_related_fmri_fir.py
..   :include-source:

As you can see, the cross-correlation method can be applied directly to the %
change, or can be applied to the zscore of the % signal change, by setting the
zscore flag. 

.. [Burock2000] M.A. Burock and A.M.Dale (2000). Estimation and Detection of
        Event-Related fMRI Signals with Temporally Correlated Noise: A
        Statistically Efficient and Unbiased Approach. Human Brain Mapping,
        11:249-260
