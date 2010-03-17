=========
fMRI data
=========

The fMRI data-set analyzed in the following examples was contributed by Beth
Mormino. The data is taken from a single subject in a "steady-state" scan, in
which subjects are fixating on a cross and maintaining alert wakefulness, but
not performing any other behavioral task.

The data was pre-processed and time-series of BOLD responses were extracted
from different regions of interest (ROIs) in the brain. The data is organized
in csv file, where each column corresponds to an ROI and each row corresponds
to a sampling point.

In the following, we will demonstrate some simple time-series analysis and
visualization techniques which can be applied to this kind of data.

This kind of data is sampled regularly, so we make use of the
:class:`UniformTimeSeries` class. We extract the data 

First, we examine the correlations between the time-series extracted from
different parts of the brain. The following script extracts the data (using
anddisplays the correlation matrix with the ROIs labeled.   

.. plot:: examples/fmri1.py
   :include-source:


We notice that the left caudate nucleus (labeled 'lcau') has an interesting
pattern of correlations. It has a high correlation with both the left putamen
('lput', which is located nearby) and also with the right caudate nucleus
('lcau'), which is the homologours region in the other hemisphere. Are these
two correlation values related to each other? The right caudate and left
putamen seem to have a moderately low correlation value. One way to examine
this question is by looking at the temporal structure of the cross-correlation
functions. In the following script, we plot the normalized cross-correlation We
might be interested in examining the temporal structure of these

.. plot:: examples/fmri2.py
   :include-source:

