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
:class:`UniformTimeSeries` class. We extract the data using
:meth:`mlab.csv2rec`, which generates a :class:`recarray` object. This data
structure contains in its :class:`dtype` a field :class:`names`, which contains
the first row in each column. In this case, that is the labels of the ROIs from
which the data in each column was extracted. The data from the
:class:`recarray` is extracted into an :class:`ndarray` and, for each ROI, it
is normalized to percent signal change, using the :func:`utils.percent_change`
function. 

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
functions. In the following script, we plot the normalized
cross-correlation.

.. plot:: examples/fmri2.py
   :include-source:

Note that the correlation is normalized to the zero-lag point (time = 0 sec)
and that there are larger correlations occuring at other time-points preceding
and following the . This could arise because of a more complex interplay of
activity between two areas, which is not captured by the correlation and can
also arise because of differences in the characteristics of the HRF in the two
ROIs. One method of analysis which can mitigate these issues is analysis of
coherency [Sun2004]_.

Next we compute the coherency between all the areas

