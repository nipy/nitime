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

Note that the correlation is normalized, so that the the value of the
cross-correation functions at the zero-lag point (time = 0 sec) is equal to the
pearson correlation between the two time-series.  We observe that there are
correlations larger than the zero-lag correlation occuring at other time-points
preceding and following the zero-lag. This could arise because of a more complex
interplay of activity between two areas, which is not captured by the
correlation and can also arise because of differences in the characteristics of
the HRF in the two ROIs. One method of analysis which can mitigate these issues
is analysis of coherency between time-series [Sun2005]_. This analysis computes
an equivalent of the correlation in the frequency domain: 

.. math::

        R_{xy} (\lambda) = \frac{f_{xy}(\lambda)}
        {\sqrt{f_{xx} (\lambda) \cdot f_{yy}(\lambda)}}

Because this is a complex number, this computation results in two
quantities. First, the magnitude of this number, also referred to as
"coherence":  

.. math::

   Coh_{xy}(\lambda) = |{R_{xy}(\lambda)}|^2 =
        \frac{|{f_{xy}(\lambda)}|^2}{f_{xx}(\lambda) \cdot f_{yy}(\lambda)}

This is a measure of the parwise coupling between the two time-series. It can
vary between 0 and 1, with 0 being complete independence and 1 being complete
coupling. A time-series would have a coherence of 1 with itself, but not only:
since this measure is independent of the relative phase of the two time-series,
the coherence between a time-series and any phase-shifted version of itself
will also be equal to 1.

However, the relative phase is another quantitiy which can be derived from this
computation:

.. math::

   \phi(\lambda) = arg [R_{xy} (\lambda)] = arg [f_{xy} (\lambda)]

	
This value can be used in order to infer which area is leading and which area
is lagging (according to the sign of the relative phase) and, can be used to
compute the temporal delay between activity in one ROI and the other.

First, let's look at the pair-wise coherence between all our ROIs. This can be
done by creating a :class:`CoherenceAnalyzer` object. Once this object is
initialized with the :class:`UniformTimeSeries` object, the mid-frequency of
the frequency bands represented in the spectral decomposition of the
time-series can be accessed in the :attr:`C.frequencies` attribute of the
object. The spectral resolution of this representation is the same one used in
the computation of the coherence. The :attr:`C.coherence` attribute is an
:class:`ndarray` of dimensions $n_{ROI}$ by $n_{ROI}$ by
$n_{frequencies}$. Since the fMRI BOLD data contains data in frequencies which
are not physilogically relevant (presumably due to machine noise and
fluctuations in physilogical measures unrelated to neural activity), we focus
our analysis on a band of frequencies between 0.02 and 0.15 Hz. This is easily
achieved by determining the values of the indices in :attr:`C.frequencies` and
using those indices in accessing the data in :attr:`C.coherence`. The coherence
is then averaged across all these frequency bands: 

.. plot:: examples/fmri3.py
   :include-source:

   
.. [Sun2005] F.T. Sun and L.M. Miller and M. D'Esposito(2005). Measuring
           temporal dynamics of functional networks using phase spectrum of
           fMRI data. Neuroimage, 28: 227-37.

