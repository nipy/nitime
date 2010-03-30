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
functions. In order to do that, from the :class:`CorrelationAnalyzer` object,
we extract the normalized cross-correlation function. This results in another
:class:`UniformTimeSeries` object, which contains the full time-series of the
cross-correlation between any combination of time-series from the different
channels in the time-series object. We can pass the resulting object, together
with a list of indices to the :func:`viz.plot_xcorr` function, which visualizes
the chosen combinations of series:  

.. code-block:: python

    xc = C.xcorr_norm

    idx_lcau = np.where(roi_names=='lcau')[0]
    idx_rcau = np.where(roi_names=='rcau')[0]
    idx_lput = np.where(roi_names=='lput')[0]

    plot_xcorr(xc,((idx_lcau,idx_rcau),(idx_lcau,idx_lput)),
			           line_labels = ['rcau','lput'])

.. plot:: examples/fmri2.py

   
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
is then averaged across all these frequency bands.  

In order to do that, we first import the :class:`CoherenceAnalyzer` object and
generate a an object of this class:

.. code-block:: python

   from nitime.analysis import CoherenceAnalyzer
   C = CoherenceAnalyzer(T)

In this case, we will examine the coherence at frequencies between 0.02 and
0.15 Hz, which are considered to be the physiologically relevant band in the
fMRI BOLD time series (see `here <http://imaging.mrc-cbu.cam.ac.uk/imaging/DesignEfficiency>`_):

We extract the indices of these frequencies from the
:attr:`CoherenceAnalyzer.frequencies` attribute:

.. code-block:: python

   freq_idx = np.where((C.frequencies>0.02) * (C.frequencies<0.15))[0]

Then, we extract the coherence in these frequency bands and average on the last
dimension, which is the frequency dimension: 

.. code-block:: python

   coh = np.mean(C.coherence[:,:,freq_idx],-1) 

Finally, we use the :func:`viz.matshow_roi` function to display the coherence
matrix:

.. code-block:: python

   matshow_roi(coh,roi_names,size=[10.,10.])

.. plot:: examples/fmri3.py

We can also focus in on the ROIs we were interested in. This requires a little
bit more manipulation of the indices into the coherence matrix:

.. code-block:: python

   idx = np.hstack([idx_lcau,idx_rcau,idx_lput,idx_rput])
   idx1 = np.vstack([[idx[i]]*4 for i in range(4)]).ravel()
   idx2 = np.hstack(4*[idx])

   coh = C.coherence[idx1,idx2].reshape(4,4,C.frequencies.shape[0])

Extract the coherence and average across the same frequency bands as before: 

.. code-block:: python

  coh = np.mean(coh[:,:,freq_idx],2) #Averaging on the last dimension

Finally, in this case, we visualize the adjacency matrix, by creating a network
graph of these ROIs (this is done by using the function
:func:`viz.drawgraph_roi` which relies on `networkx
<http://networkx.lanl.gov>`_):

.. code-block:: python

   drawgraph_roi(coh,roi_names[idx])

.. plot:: examples/fmri4.py

This shows us that there is a stronger connectivity between the left putamen and
the left caudate than between the homologous regions in the other
hemisphere. In particular, in contrast to the relatively high correlation
between the right caudate and the left caudate, there is a rather low coherence
between the time-series in these two regions, in this frequency range.

Note that the connectivity described by coherency (and other measures of
functional connectivity could arise because of neural connectivity between the
two regions, but also due to a common blood supply, or common fluctuations in
other physiological measures which affect the BOLD signal measured in both
regions. In order to be able to differentiate these two options, we would have
to conduct a comparison between two different behavioral states that affect the
neural activity in the two regions, without affecting these common
physiological factors, such as common blood supply (for an in-depth discussion
of these issues, see [Silver2010]_). In this case, we will simply assume that
the connectivity matrix presented represents the actual neural connectivity
between these two brain regions.

We notice that there is indeed a stronger coherence betwen left putamen and the
left caudate than between the left caudate and the right caudate. Next, we
might ask whether the moderate coherence between the left putamen and the right
caudate can be accounted for by the coherence these two time-series share with
the time-series derived from the left caudate. This kind of question can be
answered using an analysis of partial coherency. For the time series $x$ and
$y$, the partial coherence, given a third time-series $r$, is defined as:

.. math::

        Coh_{xy|r} = \frac{|{R_{xy}(\lambda) - R_{xr}(\lambda)
        R_{ry}(\lambda)}|^2}{(1-|{R_{xr}}|^2)(1-|{R_{ry}}|^2)}


In this case, we extract the partial coherence between the three regions,
excluding common effects of the left caudate. In order to do that, we generate
the partial-coherence attribute of the :class:`CoherenceAnalyzer` object, while
indexing on the additional dimension which this object had (the coherence
between time-series $x$ and time-series $y$, *given* time series $r$):


.. code-block:: python

   idx3 = np.hstack(16*[idx_lcau])
   coh = C.coherence_partial[idx1,idx2,idx3].reshape(4,4,C.frequencies.shape[0])
   coh = np.mean(coh[:,:,freq_idx],-1)

Again, we visualize the result, using both the :func:`viz.drawgraph_roi` and
the :func:`matshow_roi` functions:

.. plot:: examples/fmri5.py

As can be seen, the resulting partial coherence between left putamen and right
caudate, given the activity in the left caudate is smaller than the coherence
between these two areas, suggesting that part of this coherence can be
explained by their common connection to the left caudate.

.. [Sun2005] F.T. Sun and L.M. Miller and M. D'Esposito(2005). Measuring
           temporal dynamics of functional networks using phase spectrum of
           fMRI data. Neuroimage, 28: 227-37.

.. [Silver2010] M.A Silver, AN Landau, TZ Lauritzen, W Prinzmetal, LC
   Robertson(2010) Isolating human brain functional connectivity associated
   with a specific cognitive process, in Human Vision and Electronic Imaging
   XV, edited by B.E. Rogowitz and T.N. Pappas, Proceedings of SPIE, Volume
   7527, pp. 75270B-1 to 75270B-9

