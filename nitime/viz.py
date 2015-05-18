"""Tools for visualization of time-series data.

Depends on matplotlib. Some functions depend also on networkx

"""
from __future__ import print_function

# If you are running nosetests right now, you might want to use 'agg' as a backend:
import sys
from nitime.six.moves import map
from nitime.six.moves import zip
if "nose" in sys.modules:
    import matplotlib
    matplotlib.use('agg')
     
# Then do all the rest of it:
import numpy as np
from scipy import fftpack
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from mpl_toolkits.axes_grid import make_axes_locatable

from nitime import timeseries as ts
import nitime.utils as tsu
from nitime.utils import threshold_arr, minmax_norm, rescale_arr
import nitime.analysis as nta

# Matplotlib 1.3 has a bug in it, so if that's what you have, we'll replace it
# for you with a fixed version of that module:
import matplotlib
if matplotlib.__version__[:3] == '1.3' or matplotlib.__version__[:3] == '1.4':
    import nitime._mpl_units as mpl_units
    import matplotlib.axis as ax
    ax.munits = mpl_units
    
from nitime.utils import triu_indices

#Some visualization functions require networkx. Import that if possible:
try:
    import networkx as nx
    #If not, throw an error and get on with business:
except ImportError:
    e_s = "Networkx is not available. Some visualization tools might not work"
    e_s += "\n To download networkx: http://networkx.lanl.gov/"
    print(e_s)
    class NetworkxNotInstalled(object):
        def __getattribute__(self, x):
            raise ImportError(e_s)
    nx = NetworkxNotInstalled()


def plot_tseries(time_series, fig=None, axis=0,
                 xticks=None, xunits=None, yticks=None, yunits=None,
                 xlabel=None, ylabel=None, yerror=None, error_alpha=0.1,
                 time_unit=None, **kwargs):

    """plot a timeseries object

    Arguments
    ---------

    time_series: a nitime time-series object

    fig: a figure handle, opens a new figure if None

    subplot: an axis number (if there are several in the figure to be opened),
        defaults to 0.

    xticks: optional, list, specificies what values to put xticks on. Defaults
    to the matlplotlib-generated.

    yticks: optional, list, specificies what values to put xticks on. Defaults
    to the matlplotlib-generated.

    xlabel: optional, list, specificies what labels to put on xticks

    ylabel: optional, list, specificies what labels to put on yticks

    yerror: optional, UniformTimeSeries with the same sampling_rate and number
    of samples and channels as time_series, the error will be displayed as a
    shading above and below the plotted time-series

    """

    if fig is None:
        fig = plt.figure()

    if not fig.get_axes():
        ax = fig.add_subplot(1, 1, 1)
    else:
        ax = fig.get_axes()[axis]

    #Make sure that time displays on the x axis with the units you want:
    #If you want to change the time-unit on the visualization from that used to
    #represent the time-series:
    if time_unit is not None:
        tu = time_unit
        conv_fac = ts.time_unit_conversion[time_unit]
    #Otherwise, get the information from your input:
    else:
        tu = time_series.time_unit
        conv_fac = time_series.time._conversion_factor

    this_time = time_series.time / float(conv_fac)
    ax.plot(this_time, time_series.data.T, **kwargs)

    if xlabel is None:
        ax.set_xlabel('Time (%s)' % tu)
    else:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if yerror is not None:
        if len(yerror.data.shape) == 1:
            this_e = yerror.data[np.newaxis, :]
        else:
            this_e = yerror.data
        delta = this_e
        e_u = time_series.data + delta
        e_d = time_series.data - delta
        for i in range(e_u.shape[0]):
            ax.fill_between(this_time, e_d[i], e_u[i], alpha=error_alpha)

    return fig


def matshow_tseries(time_series, fig=None, axis=0, xtick_n=5, time_unit=None,
                    xlabel=None, ylabel=None):

    """Creates an image of the time-series, ordered according to the first
    dimension of the time-series object

    Parameters
    ----------

    time_series: a nitime time-series object

    fig: a figure handle, opens a new figure if None

    axis: an axis number (if there are several in the figure to be opened),
        defaults to 0.

    xtick_n: int, optional, sets the number of ticks to be placed on the x axis
    """

    if fig is None:
        fig = plt.figure()

    if not fig.get_axes():
        ax = fig.add_subplot(1, 1, 1)
    else:
        ax = fig.get_axes()[axis]

    #Make sure that time displays on the x axis with the units you want:
    #If you want to change the time-unit on the visualization from that used to
    #represent the time-series:
    if time_unit is not None:
        tu = time_unit
        conv_fac = ts.time_unit_conversion[time_unit]
    #Otherwise, get the information from your input:
    else:
        tu = time_series.time_unit
        conv_fac = time_series.time._conversion_factor

    this_time = time_series.time / float(conv_fac)
    ax.matshow(time_series.data)

    ax.set_xticks(list(range(len(this_time)))[::len(this_time) / xtick_n])
    ax.set_xticklabels(this_time[::len(this_time) / xtick_n])

    if xlabel is None:
        ax.set_xlabel('Time (%s)' % tu)
    else:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    return fig


## Helper functions for matshow_roi and for drawgraph_roi, in order to get the
## right cmap for the colorbar:

##Currently not used at all - should they be removed?
def rgb_to_dict(value, cmap):
    return dict(zip(('red', 'green', 'blue', 'alpha'), cmap(value)))


def subcolormap(xmin, xmax, cmap):
    '''Returns the part of cmap between xmin, xmax, scaled to 0,1.'''
    assert xmin < xmax
    assert xmax <= 1
    cd = cmap._segmentdata.copy()
    colornames = ('red', 'green', 'blue')
    rgbmin, rgbmax = rgb_to_dict(xmin, cmap), rgb_to_dict(xmax, cmap)
    for k in cd:
        tmp = [x for x in cd[k] if x[0] >= xmin and x[0] <= xmax]
        if tmp == [] or tmp[0][0] > xmin:
            tmp = [(xmin, rgbmin[k], rgbmin[k])] + tmp
        if tmp == [] or tmp[-1][0] < xmax:
            tmp = tmp + [(xmax, rgbmax[k], rgbmax[k])]
        #now scale all this to (0,1)
        square = list(zip(*tmp))
        xbreaks = [(x - xmin) / (xmax - xmin) for x in square[0]]
        square[0] = xbreaks
        tmp = list(zip(*square))
        cd[k] = tmp
    return colors.LinearSegmentedColormap('local', cd, N=256)


def drawmatrix_channels(in_m, channel_names=None, fig=None, x_tick_rot=0,
                        size=None, cmap=plt.cm.RdBu_r, colorbar=True,
                        color_anchor=None, title=None):
    r"""Creates a lower-triangle of the matrix of an nxn set of values. This is
    the typical format to show a symmetrical bivariate quantity (such as
    correlation or coherence between two different ROIs).

    Parameters
    ----------

    in_m: nxn array with values of relationships between two sets of rois or
    channels

    channel_names (optional): list of strings with the labels to be applied to
    the channels in the input. Defaults to '0','1','2', etc.

    fig (optional): a matplotlib figure

    cmap (optional): a matplotlib colormap to be used for displaying the values
    of the connections on the graph

    title (optional): string to title the figure (can be like '$\alpha$')

    color_anchor (optional): determine the mapping from values to colormap
        if None, min and max of colormap correspond to min and max of in_m
        if 0, min and max of colormap correspond to max of abs(in_m)
        if (a,b), min and max of colormap correspond to (a,b)

    Returns
    -------

    fig: a figure object

    """
    N = in_m.shape[0]
    ind = np.arange(N)  # the evenly spaced plot indices

    def channel_formatter(x, pos=None):
        thisind = np.clip(int(x), 0, N - 1)
        return channel_names[thisind]

    if fig is None:
        fig = plt.figure()

    if size is not None:

        fig.set_figwidth(size[0])
        fig.set_figheight(size[1])

    w = fig.get_figwidth()
    h = fig.get_figheight()

    ax_im = fig.add_subplot(1, 1, 1)

    #If you want to draw the colorbar:
    if colorbar:
        divider = make_axes_locatable(ax_im)
        ax_cb = divider.new_vertical(size="10%", pad=0.1, pack_start=True)
        fig.add_axes(ax_cb)

    #Make a copy of the input, so that you don't make changes to the original
    #data provided
    m = in_m.copy()

    #Null the upper triangle, so that you don't get the redundant and the
    #diagonal values:
    idx_null = triu_indices(m.shape[0])
    m[idx_null] = np.nan

    #Extract the minimum and maximum values for scaling of the
    #colormap/colorbar:
    max_val = np.nanmax(m)
    min_val = np.nanmin(m)

    if color_anchor is None:
        color_min = min_val
        color_max = max_val
    elif color_anchor == 0:
        bound = max(abs(max_val), abs(min_val))
        color_min = -bound
        color_max = bound
    else:
        color_min = color_anchor[0]
        color_max = color_anchor[1]

    #The call to imshow produces the matrix plot:
    im = ax_im.imshow(m, origin='upper', interpolation='nearest',
       vmin=color_min, vmax=color_max, cmap=cmap)

    #Formatting:
    ax = ax_im
    ax.grid(True)
    #Label each of the cells with the row and the column:
    if channel_names is not None:
        for i in range(0, m.shape[0]):
            if i < (m.shape[0] - 1):
                ax.text(i - 0.3, i, channel_names[i], rotation=x_tick_rot)
            if i > 0:
                ax.text(-1, i + 0.3, channel_names[i],
                        horizontalalignment='right')

        ax.set_axis_off()
        ax.set_xticks(np.arange(N))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(channel_formatter))
        fig.autofmt_xdate(rotation=x_tick_rot)
        ax.set_yticks(np.arange(N))
        ax.set_yticklabels(channel_names)
        ax.set_ybound([-0.5, N - 0.5])
        ax.set_xbound([-0.5, N - 1.5])

    #Make the tick-marks invisible:
    for line in ax.xaxis.get_ticklines():
        line.set_markeredgewidth(0)

    for line in ax.yaxis.get_ticklines():
        line.set_markeredgewidth(0)

    ax.set_axis_off()

    if title is not None:
        ax.set_title(title)

    #The following produces the colorbar and sets the ticks
    if colorbar:
        #Set the ticks - if 0 is in the interval of values, set that, as well
        #as the maximal and minimal values:
        if min_val < 0:
            ticks = [color_min, min_val, 0, max_val, color_max]
        #Otherwise - only set the minimal and maximal value:
        else:
            ticks = [color_min, min_val, max_val, color_max]

        #This makes the colorbar:
        cb = fig.colorbar(im, cax=ax_cb, orientation='horizontal',
                          cmap=cmap,
                          norm=im.norm,
                          boundaries=np.linspace(color_min, color_max, 256),
                          ticks=ticks,
                          format='%.2f')

    # Set the current figure active axis to be the top-one, which is the one
    # most likely to be operated on by users later on
    fig.sca(ax)

    return fig


def drawgraph_channels(in_m, channel_names=None, cmap=plt.cm.RdBu_r,
                       node_shapes=None, node_colors=None,
                       title=None, layout=None, threshold=None):

    """Draw a graph based on the matrix specified in in_m. Wrapper to
    draw_graph.

    Parameters
    ----------

    in_m: nxn array with values of relationships between two sets of channels
    or channels

    channel_names (optional): list of strings with the labels to be applied to
    the channels in the input. Defaults to '0','1','2', etc.

    cmap (optional): a matplotlib colormap to be used for displaying the values
    of the connections on the graph

    node_shapes: defaults to circle

    node_colors: defaults to white,

    title:

    layout, defaults to nx.circular_layout
    Returns
    -------
    fig: a figure object

    Notes
    -----

    The layout of the graph is done using functions from networkx
    (http://networkx.lanl.gov), which is a dependency of this function
    """
    nnodes = in_m.shape[0]
    if channel_names is None:
        node_labels = None  # [None]*nnodes
    else:
        node_labels = list(channel_names)

    if node_shapes is None:
        node_shapes = ['o'] * nnodes

    if node_colors is None:
        node_colors = ['w'] * nnodes

    #Make a copy, avoiding making changes to the original data:
    m = in_m.copy()

    #Set the diagonal values to the minimal value of the matrix, so that the
    #vrange doesn't always get stretched to 1:
    m[np.arange(nnodes), np.arange(nnodes)] = min(np.nanmin(m), -np.nanmax(m))
    range_setter = max(abs(np.nanmin(m)), abs(np.nanmax(m)))
    vrange = [-range_setter, range_setter]

    #m[np.where(np.isnan(m))] = 0
    if threshold is None:
        #If there happens to be an off-diagnoal edge in the adjacency matrix
        #which is just as small as the minimum, we don't want to drop that one:
        eps = 10 ** -10
        G = mkgraph(m, threshold=vrange[0] - eps, threshold2=None)
    else:
        G = mkgraph(m, threshold=threshold[0], threshold2=threshold[1])
    fig = draw_graph(G,
                     node_colors=node_colors,
                     node_shapes=node_shapes,
                     node_scale=2,
                     labels=node_labels,
                     edge_cmap=cmap,
                     colorbar=True,
                     vrange=vrange,
                     title=title,
                     stretch_factor=1,
                     edge_alpha=False,
                     layout=layout
                     )
    return fig


def plot_xcorr(xc, ij, fig=None, line_labels=None, xticks=None, yticks=None,
               xlabel=None, ylabel=None):
    """ Visualize the cross-correlation function"""

    if fig is None:
        fig = plt.figure()

    if not fig.get_axes():
        ax = fig.add_subplot(1, 1, 1)
    else:
        ax = fig.get_axes()[0]

    if line_labels is not None:
        #Reverse the order, so that pop() works:
        line_labels.reverse()
        this_labels = line_labels

    #Use the ij input as the labels:
    else:
        this_labels = [str(this) for this in ij].reverse()

    #Make sure that time displays on the x axis with the units you want:
    conv_fac = xc.time._conversion_factor
    this_time = xc.time / float(conv_fac)

    for (i, j) in ij:
        if this_labels is not None:
            #Use pop() to get the first one and remove it:
            ax.plot(this_time, xc.data[i, j].squeeze(),
                    label=this_labels.pop())
        else:
            ax.plot(this_time, xc.data[i, j].squeeze())

    ax.set_xlabel('Time(sec)')
    ax.set_ylabel('Correlation(normalized)')

    if xlabel is None:
        #Make sure that time displays on the x axis with the units you want:
        conv_fac = xc.time._conversion_factor
        time_label = xc.time / float(conv_fac)
        ax.set_xlabel('Time (%s)' % xc.time_unit)
    else:
        time_label = xlabel
        ax.set_xlabel(xlabel)

    if line_labels is not None:
        plt.legend()

    if ylabel is None:
        ax.set_ylabel('Correlation')
    else:
        ax.set_ylabel(ylabel)

    return fig


#-----------------------------------------------------------------------------
# Functions from brainx:
#-----------------------------------------------------------------------------
def draw_matrix(mat, th1=None, th2=None, clim=None, cmap=None):
    """Draw a matrix, optionally thresholding it.
    """
    if th1 is not None:
        m2 = tsu.thresholded_arr(mat, th1, th2)
    else:
        m2 = mat
    ax = plt.matshow(m2, cmap=cmap)
    if clim is not None:
        ax.set_clim(*clim)
    plt.colorbar()
    return ax


def draw_arrows(G, pos, edgelist=None, ax=None, edge_color='k', alpha=1.0,
                width=1):
    """Draw arrows on a set of edges"""

    if ax is None:
        ax = plt.gca()

    if edgelist is None:
        edgelist = G.edges()

    if not edgelist or len(edgelist) == 0:  # no edges!
        return

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])

    arrow_colors = (colors.colorConverter.to_rgba('k', alpha), )
    a_pos = []

    # Radius of the nodes in world coordinates
    radius = 0.5
    head_length = 0.31
    overhang = 0.1

    #ipvars('edge_pos')  # dbg

    for src, dst in edge_pos:
        dd = dst - src
        nd = np.linalg.norm(dd)
        if nd == 0:  # source and target at same position
            continue

        s = 1.0 - radius / nd
        dd *= s
        x1, y1 = src
        dx, dy = dd
        ax.arrow(x1, y1,
                 dx, dy,
                 lw=width, width=width,
                 head_length=head_length,
                 fc=edge_color, ec='none',
                 alpha=alpha, overhang=overhang)


def draw_graph(G,
               labels=None,
               node_colors=None,
               node_shapes=None,
               node_scale=1.0,
               edge_style='solid',
               edge_cmap=None,
               colorbar=False,
               vrange=None,
               layout=None,
               title=None,
               font_family='sans-serif',
               font_size=9,
               stretch_factor=1.0,
               edge_alpha=True,
               fig_size=None):
    """Draw a weighted graph with options to visualize link weights.

    The resulting diagram uses the rank of each node as its size, and the
    weight of each link (after discarding thresholded values, see below) as the
    link opacity.

    It maps edge weight to color as well as line opacity and thickness,
    allowing the color part to be hardcoded over a value range (to permit valid
    cross-figure comparisons for different graphs, so the same color
    corresponds to the same link weight even if each graph has a different
    range of weights).  The nodes sizes are proportional to their degree,
    computed as the sum of the weights of all their links.  The layout defaults
    to circular, but any nx layout function can be passed in, as well as a
    statically precomputed layout.

    Parameters
    ----------
    G : weighted graph
      The values must be of the form (v1,v2), with all v2 in [0,1].  v1 are
      used for colors, v2 for thickness/opacity.

    labels : list or dict, optional.
      An indexable object that maps nodes to strings.  If not given, the
      string form of each node is used as a label.  If False, no labels are
      drawn.

    node_colors : list or dict, optional.
      An indexable object that maps nodes to valid matplotlib color specs.  See
      matplotlib's plot() function for details.

    node_shapes : list or dict, optional.
      An indexable object that maps nodes to valid matplotlib shape specs.  See
      matplotlib's scatter() function for details.  If not given, circles are
      used.

    node_scale : float, optional
      A scale factor to globally stretch or shrink all nodes symbols by.

    edge_style : string, optional
      Line style for the edges, defaults to 'solid'.

    edge_cmap : matplotlib colormap, optional.
      A callable that returns valid color specs, like matplotlib colormaps.
      If not given, edges are colored black.

    colorbar : bool
      If true, automatically add a colorbar showing the mapping of graph weight
      values to colors.

    vrange : pair of floats
      If given, this indicates the total range of values that the weights can
      in principle occupy, and is used to set the lower/upper range of the
      colormap.  This allows you to set the range of multiple different figures
      to the same values, even if each individual graph has range variations,
      so that visual color comparisons across figures are valid.

    layout : function or layout dict, optional
      A NetworkX-like layout function or the result of a precomputed layout for
      the given graph.  NetworkX produces layouts as dicts keyed by nodes and
      with (x,y) pairs of coordinates as values, any function that produces
      this kind of output is acceptable.  Defaults to nx.circular_layout.

    title : string, optional.
      If given, title to put on the main plot.

    font_family : string, optional.
      Font family used for the node labels and title.

    font_size : int, optional.
      Font size used for the node labels and title.

    stretch_factor : float, optional
      A global scaling factor to make the graph larger (or smaller if <1).
      This can be used to separate the nodes if they start overlapping.

    edge_alpha: bool, optional
      Whether to weight the transparency of each edge by a factor equivalent to
      its relative weight

    fig_size: list of height by width, the size of the figure (in
    inches). Defaults to [6,6]

    Returns
    -------
    fig
      The matplotlib figure object with the plot.
    """
    if fig_size is None:
        figsize = [6, 6]

    scaler = figsize[0] / 6.
    # For the size of the node symbols
    node_size_base = 1000 * scaler
    node_min_size = 200 * scaler
    default_node_shape = 'o'
    # Default colors if none given
    default_node_color = 'r'
    default_edge_color = 'k'
    # Max edge width
    max_width = 13 * scaler
    min_width = 2 * scaler
    font_family = 'sans-serif'

    # We'll use the nodes a lot, let's make a numpy array of them
    nodes = np.array(sorted(G.nodes()))
    nnod = len(nodes)

    # Build a 'weighted degree' array obtained by adding the (absolute value)
    # of the weights for all edges pointing to each node:
    amat = nx.adj_matrix(G).A  # get a normal array out of it
    degarr = abs(amat).sum(0)  # weights are sums across rows

    # Map the degree to the 0-1 range so we can use it for sizing the nodes.
    try:
        odegree = rescale_arr(degarr, 0, 1)
        # Make an array of node sizes based on node degree
        node_sizes = odegree * node_size_base + node_min_size
    except ZeroDivisionError:
        # All nodes same size
        node_sizes = np.empty(nnod, float)
        node_sizes.fill(0.5 * node_size_base + node_min_size)

    # Adjust node size list.  We square the scale factor because in mpl, node
    # sizes represent area, not linear size, but it's more intuitive for the
    # user to think of linear factors (the overall figure scale factor is also
    # linear).
    node_sizes *= node_scale ** 2

    # Set default node properties
    if node_colors is None:
        node_colors = [default_node_color] * nnod

    if node_shapes is None:
        node_shapes = [default_node_shape] * nnod

    # Set default edge colormap
    if edge_cmap is None:
        # Make an object with the colormap API, that maps all input values to
        # the default color (with proper alhpa)
        edge_cmap = (lambda val, alpha:
                     colors.colorConverter.to_rgba(default_edge_color, alpha))

    # if vrange is None, we set the color range from the values, else the user
    # can specify it

    # e[2] is edge value: edges_iter returns (i,j,data)
    gvals = np.array([e[2]['weight'] for e in G.edges(data=True)])
    gvmin, gvmax = gvals.min(), gvals.max()

    gvrange = gvmax - gvmin
    if vrange is None:
        vrange = gvmin, gvmax
    # Now, construct the normalization for the colormap
    cnorm = mpl.colors.Normalize(vmin=vrange[0], vmax=vrange[1])

    # Create the actual plot where the graph will be displayed
    figsize = np.array(figsize, float)
    figsize *= stretch_factor

    fig = plt.figure(figsize=figsize)
    ax_graph = fig.add_subplot(1, 1, 1)
    fig.sca(ax_graph)

    if layout is None:
        layout = nx.circular_layout
    # Compute positions for all nodes - nx has several algorithms
    if callable(layout):
        pos = layout(G)
    else:
        # The user can also provide a precomputed layout
        pos = layout

    # Draw nodes
    for nod in nodes:
        nx.draw_networkx_nodes(G,
                               pos,
                               nodelist=[nod],
                               node_color=node_colors[nod],
                               node_shape=node_shapes[nod],
                               node_size=node_sizes[nod])
    # Draw edges
    if not isinstance(G, nx.DiGraph):
        # Undirected graph, simple lines for edges
        # We need the size of the value range to properly scale colors
        vsize = vrange[1] - vrange[0]
        gvals_normalized = G.metadata['vals_norm']
        for (u, v, y) in G.edges(data=True):
            # The graph value is the weight, and the normalized values are in
            # [0,1], used for thickness/transparency
            alpha = gvals_normalized[u, v]
            # Scale the color choice to the specified vrange, so that
            ecol = (y['weight'] - vrange[0]) / vsize
            #print 'u,v:',u,v,'y:',y,'ecol:',ecol  # dbg

            if edge_alpha:
                fade = alpha
            else:
                fade = 1.0

            edge_color = [tuple(edge_cmap(ecol, fade))]
            #dbg:
            #print u,v,y
            draw_networkx_edges(G,
                                pos,
                                edgelist=[(u, v)],
                                width=min_width + alpha * max_width,
                                edge_color=edge_color,
                                style=edge_style)
    else:
        # Directed graph, use arrows.
        # XXX - this is currently broken.
        raise NotImplementedError("arrow drawing currently broken")

        ## for (u,v,x) in G.edges(data=True):
        ##     y,w = x
        ##     draw_arrows(G,pos,edgelist=[(u,v)],
        ##                 edge_color=[w],
        ##                 alpha=w,
        ##                 edge_cmap=edge_cmap,
        ##                 width=w*max_width)

    # Draw labels.  If not given, we use the string form of the nodes.  If
    # labels is False, no labels are drawn.
    if labels is None:
        labels = map(str, nodes)

    if labels:
        lab_idx = list(range(len(labels)))
        labels_dict = dict(zip(lab_idx, labels))
        nx.draw_networkx_labels(G,
                                pos,
                                labels_dict,
                                font_size=font_size,
                                font_family=font_family)

    if title:
        plt.title(title, fontsize=font_size)

    # Turn off x and y axes labels in pylab
    plt.xticks([])
    plt.yticks([])

    # Add a colorbar if requested
    if colorbar:
        divider = make_axes_locatable(ax_graph)
        ax_cb = divider.new_vertical(size="20%", pad=0.2, pack_start=True)
        fig.add_axes(ax_cb)
        cb = mpl.colorbar.ColorbarBase(ax_cb,
                                    cmap=edge_cmap,
                                    norm=cnorm,
                                    #boundaries = np.linspace(min((gvmin,0)),
                                    #                         max((gvmax,0)),
                                    #                         256),
                                    orientation='horizontal',
                                    format='%.2f')

    # Always return the MPL figure object so the user can further manipulate it
    return fig


def lab2node(labels, labels_dict):
    return [labels_dict[ll] for ll in labels]


### Patched version for networx draw_networkx_edges, sent to Aric.
def draw_networkx_edges(G, pos,
                        edgelist=None,
                        width=1.0,
                        edge_color='k',
                        style='solid',
                        alpha=None,
                        edge_cmap=None,
                        edge_vmin=None,
                        edge_vmax=None,
                        ax=None,
                        arrows=True,
                        **kwds):
    """Draw the edges of the graph G

    This draws only the edges of the graph G.

    pos is a dictionary keyed by vertex with a two-tuple
    of x-y positions as the value.
    See networkx.layout for functions that compute node positions.

    edgelist is an optional list of the edges in G to be drawn.
    If provided, only the edges in edgelist will be drawn.

    edgecolor can be a list of matplotlib color letters such as 'k' or
    'b' that lists the color of each edge; the list must be ordered in
    the same way as the edge list. Alternatively, this list can contain
    numbers and those number are mapped to a color scale using the color
    map edge_cmap.  Finally, it can also be a list of (r,g,b) or (r,g,b,a)
    tuples, in which case these will be used directly to color the edges.  If
    the latter mode is used, you should not provide a value for alpha, as it
    would be applied globally to all lines.

    For directed graphs, 'arrows' (actually just thicker stubs) are drawn
    at the head end.  Arrows can be turned off with keyword arrows=False.

    See draw_networkx for the list of other optional parameters.

    """
    try:
        import matplotlib.pylab as pylab
        import matplotlib.cbook as cb
        from matplotlib.colors import colorConverter, Colormap
        from matplotlib.collections import LineCollection
    except ImportError:
        raise ImportError("Matplotlib required for draw()")
    except RuntimeError:
        pass  # unable to open display

    if ax is None:
        ax = pylab.gca()

    if edgelist is None:
        edgelist = G.edges()

    if not edgelist or len(edgelist) == 0:  # no edges!
        return None

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])

    if not cb.iterable(width):
        lw = (width,)
    else:
        lw = width

    if not cb.is_string_like(edge_color) \
           and cb.iterable(edge_color) \
           and len(edge_color) == len(edge_pos):
        if np.alltrue([cb.is_string_like(c)
                         for c in edge_color]):
            # (should check ALL elements)
            # list of color letters such as ['k','r','k',...]
            edge_colors = tuple([colorConverter.to_rgba(c, alpha)
                                 for c in edge_color])
        elif np.alltrue([not cb.is_string_like(c)
                           for c in edge_color]):
            # If color specs are given as (rgb) or (rgba) tuples, we're OK
            if np.alltrue([cb.iterable(c) and len(c) in (3, 4)
                             for c in edge_color]):
                edge_colors = tuple(edge_color)
                alpha = None
            else:
                # numbers (which are going to be mapped with a colormap)
                edge_colors = None
        else:
            e_s = 'edge_color must consist of either color names or numbers'
            raise ValueError(e_s)
    else:
        if len(edge_color) == 1:
            edge_colors = (colorConverter.to_rgba(edge_color, alpha),)
        else:
            e_s = 'edge_color must be a single color or list of exactly'
            e_s += 'm colors where m is the number or edges'
            raise ValueError(e_s)
    edge_collection = LineCollection(edge_pos,
                                     colors=edge_colors,
                                     linewidths=lw,
                                     antialiaseds=(1,),
                                     linestyle=style,
                                     transOffset=ax.transData,
                                     )

    # Note: there was a bug in mpl regarding the handling of alpha values for
    # each line in a LineCollection.  It was fixed in matplotlib in r7184 and
    # r7189 (June 6 2009).  We should then not set the alpha value globally,
    # since the user can instead provide per-edge alphas now.  Only set it
    # globally if provided as a scalar.
    if cb.is_numlike(alpha):
        edge_collection.set_alpha(alpha)

    # need 0.87.7 or greater for edge colormaps
    if edge_colors is None:
        if edge_cmap is not None:
            assert(isinstance(edge_cmap, Colormap))
        edge_collection.set_array(np.asarray(edge_color))
        edge_collection.set_cmap(edge_cmap)
        if edge_vmin is not None or edge_vmax is not None:
            edge_collection.set_clim(edge_vmin, edge_vmax)
        else:
            edge_collection.autoscale()
        pylab.sci(edge_collection)

#    else:
#        sys.stderr.write(\
#            """matplotlib version >= 0.87.7 required for colormapped edges.
#        (version %s detected)."""%matplotlib.__version__)
#        raise UserWarning(\
#            """matplotlib version >= 0.87.7 required for colormapped edges.
#        (version %s detected)."""%matplotlib.__version__)

    arrow_collection = None

    if G.is_directed() and arrows:

        # a directed graph hack
        # draw thick line segments at head end of edge
        # waiting for someone else to implement arrows that will work
        arrow_colors = (colorConverter.to_rgba('k', alpha),)
        a_pos = []
        p = 1.0 - 0.25  # make head segment 25 percent of edge length
        for src, dst in edge_pos:
            x1, y1 = src
            x2, y2 = dst
            dx = x2 - x1  # x offset
            dy = y2 - y1  # y offset
            d = np.sqrt(float(dx ** 2 + dy ** 2))  # length of edge
            if d == 0:  # source and target at same position
                continue
            if dx == 0:  # vertical edge
                xa = x2
                ya = dy * p + y1
            if dy == 0:  # horizontal edge
                ya = y2
                xa = dx * p + x1
            else:
                theta = np.arctan2(dy, dx)
                xa = p * d * np.cos(theta) + x1
                ya = p * d * np.sin(theta) + y1

            a_pos.append(((xa, ya), (x2, y2)))

        arrow_collection = LineCollection(a_pos,
                                colors=arrow_colors,
                                linewidths=[4 * ww for ww in lw],
                                antialiaseds=(1,),
                                transOffset=ax.transData,
                                )

    # update view
    minx = np.amin(np.ravel(edge_pos[:, :, 0]))
    maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
    miny = np.amin(np.ravel(edge_pos[:, :, 1]))
    maxy = np.amax(np.ravel(edge_pos[:, :, 1]))

    w = maxx - minx
    h = maxy - miny
    padx, pady = 0.05 * w, 0.05 * h
    corners = (minx - padx, miny - pady), (maxx + padx, maxy + pady)
    ax.update_datalim(corners)
    ax.autoscale_view()

    edge_collection.set_zorder(1)  # edges go behind nodes
    ax.add_collection(edge_collection)
    if arrow_collection:
        arrow_collection.set_zorder(1)  # edges go behind nodes
        ax.add_collection(arrow_collection)

    return ax


def mkgraph(cmat, threshold=0.0, threshold2=None):
    """Make a weighted graph object out of an adjacency matrix.

    The values in the original matrix cmat can be thresholded out.  If only one
    threshold is given, all values below that are omitted when creating edges.
    If two thresholds are given, then values in the th2-th1 range are
    ommitted.  This allows for the easy creation of weighted graphs with
    positive and negative values where a range of weights around 0 is omitted.

    Parameters
    ----------
    cmat : 2-d square array
      Adjacency matrix.
    threshold : float
      First threshold.
    threshold2 : float
      Second threshold.

    Returns
    -------
    G : a NetworkX weighted graph object, to which a dictionary called
    G.metadata is appended.  This dict contains the original adjacency matrix
    cmat, the two thresholds, and the weights
    """

    # Input sanity check
    nrow, ncol = cmat.shape
    if nrow != ncol:
        raise ValueError("Adjacency matrix must be square")

    row_idx, col_idx, vals = threshold_arr(cmat, threshold, threshold2)
    # Also make the full thresholded array available in the metadata
    cmat_th = np.empty_like(cmat)
    if threshold2 is None:
        cmat_th.fill(threshold)
    else:
        cmat_th.fill(-np.inf)
    cmat_th[row_idx, col_idx] = vals

    # Next, make a normalized copy of the values.  For the 2-threshold case, we
    # use 'folding' normalization
    if threshold2 is None:
        vals_norm = minmax_norm(vals)
    else:
        vals_norm = minmax_norm(vals, 'folding', [threshold, threshold2])

    # Now make the actual graph
    G = nx.Graph(weighted=True)
    G.add_nodes_from(list(range(nrow)))
    # To keep the weights of the graph to simple values, we store the
    # normalize ones in a separate dict that we'll stuff into the graph
    # metadata.

    normed_values = {}
    for i, j, val, nval in zip(row_idx, col_idx, vals, vals_norm):
        if i == j:
            # no self-loops
            continue
        G.add_edge(i, j, weight=val)
        normed_values[i, j] = nval

    # Write a metadata dict into the graph and save the threshold info there
    G.metadata = dict(threshold1=threshold,
                      threshold2=threshold2,
                      cmat_raw=cmat,
                      cmat_th=cmat_th,
                      vals_norm=normed_values,
                      )
    return G


def plot_snr(tseries, lb=0, ub=None, fig=None):
    """
    Show the coherence, snr and information of an SNRAnalyzer

    Parameters
    ----------
    tseries: nitime TimeSeries object
       Multi-trial data in response to one stimulus/protocol with the dims:
       (n_channels,n_repetitions,time)

    lb,ub: float
       Lower and upper bounds on the frequency range over which to
       calculate (default to [0,Nyquist]).

    Returns
    -------

    A tuple containing:

    fig: a matplotlib figure object
        This figure displays:
        1. Coherence
        2. SNR
        3. Information
    """

    if fig is None:
        fig = plt.figure()

    ax_spectra = fig.add_subplot(1, 2, 1)
    ax_snr_info = fig.add_subplot(1, 2, 2)

    A = []
    info = []
    s_n_r = []
    coh = []
    noise_spectra = []
    signal_spectra = []
    #If you only have one channel, make sure that everything still works by
    #adding an axis
    if len(tseries.data.shape) < 3:
        this = tseries.data[np.newaxis, :, :]
    else:
        this = tseries.data

    for i in range(this.shape[0]):
        A.append(nta.SNRAnalyzer(ts.TimeSeries(this[i],
                                    sampling_rate=tseries.sampling_rate)))
        info.append(A[-1].mt_information)
        s_n_r.append(A[-1].mt_snr)
        coh.append(A[-1].mt_coherence)
        noise_spectra.append(A[-1].mt_noise_psd)
        signal_spectra.append(A[-1].mt_signal_psd)

    freqs = A[-1].mt_frequencies

    lb_idx, ub_idx = tsu.get_bounds(freqs, lb, ub)
    freqs = freqs[lb_idx:ub_idx]

    coh_mean = np.mean(coh, 0)
    snr_mean = np.mean(s_n_r, 0)
    info_mean = np.mean(info, 0)
    n_spec_mean = np.mean(noise_spectra, 0)
    s_spec_mean = np.mean(signal_spectra, 0)

    ax_spectra.plot(freqs, np.log(s_spec_mean[lb_idx:ub_idx]), label='Signal')
    ax_spectra.plot(freqs, np.log(n_spec_mean[lb_idx:ub_idx]), label='Noise')
    ax_spectra.set_xlabel('Frequency (Hz)')
    ax_spectra.set_ylabel('Spectral power (dB)')

    ax_snr_info.plot(freqs, snr_mean[lb_idx:ub_idx], label='SNR')
    ax_snr_info.plot(np.nan, np.nan, 'r', label='Info')
    ax_snr_info.set_ylabel('SNR')
    ax_snr_info.set_xlabel('Frequency (Hz)')
    ax_info = ax_snr_info.twinx()
    ax_info.plot(freqs, np.cumsum(info_mean[lb_idx:ub_idx]), 'r')
    ax_info.set_ylabel('Cumulative information rate (bits/sec)')
    return fig


def plot_snr_diff(tseries1, tseries2, lb=0, ub=None, fig=None,
                  ts_names=['1', '2'],
                  bandwidth=None, adaptive=False, low_bias=True):

    """
    Show distributions of differences between two time-series in the
    amount of snr (freq band by freq band) and information. For example,
    for comparing two stimulus conditions

    Parameters
    ----------
    tseries1, tseries2 : nitime TimeSeries objects
       These are the time-series to compare, with each of them having the
       dims: (n_channels, n_reps, time), where n_channels1 = n_channels2

    lb,ub: float
       Lower and upper bounds on the frequency range over which to
       calculate the information rate (default to [0,Nyquist]).

    fig: matplotlib figure object
       If you want to do this on already existing figure. Otherwise, a new
       figure object will be generated.

    ts_names: list of str
       Labels for the two inputs, to be used in plotting (defaults to
       ['1','2'])

    bandwidth, adaptive, low_bias: See :func:`nta.SNRAnalyzer` for details


    Returns
    -------

    A tuple containing:

    fig: a matplotlib figure object
        This figure displays:
        1. The histogram of the information differences between the two
        time-series
        2. The frequency-dependent SNR for the two time-series

    info1, info2: float arrays
        The frequency-dependent information rates (in bits/sec)

    s_n_r1, s_n_r2: float arrays
         The frequncy-dependent signal-to-noise ratios

    """
    if fig is None:
        fig = plt.figure()
    ax_scatter = fig.add_subplot(1, 2, 1)
    ax_snr = fig.add_subplot(1, 2, 2)

    SNR1 = []
    s_n_r1 = []
    info1 = []
    SNR2 = []
    info2 = []
    s_n_r2 = []

    #If you only have one channel, make sure that everything still works by
    #adding an axis
    if len(tseries1.data.shape) < 3:
        this1 = tseries1.data[np.newaxis, :, :]
        this2 = tseries2.data[np.newaxis, :, :]
    else:
        this1 = tseries1.data
        this2 = tseries2.data

    for i in range(this1.shape[0]):
        SNR1.append(nta.SNRAnalyzer(ts.TimeSeries(this1[i],
                                    sampling_rate=tseries1.sampling_rate),
                                bandwidth=bandwidth,
                                adaptive=adaptive,
                                low_bias=low_bias))
        info1.append(SNR1[-1].mt_information)
        s_n_r1.append(SNR1[-1].mt_snr)

        SNR2.append(nta.SNRAnalyzer(ts.TimeSeries(this2[i],
                                    sampling_rate=tseries2.sampling_rate),
                                bandwidth=bandwidth,
                                adaptive=adaptive,
                                low_bias=low_bias))

        info2.append(SNR2[-1].mt_information)
        s_n_r2.append(SNR2[-1].mt_snr)

    freqs = SNR1[-1].mt_frequencies

    lb_idx, ub_idx = tsu.get_bounds(freqs, lb, ub)
    freqs = freqs[lb_idx:ub_idx]

    info1 = np.array(info1)
    info_sum1 = np.sum(info1[:, lb_idx:ub_idx], -1)
    info2 = np.array(info2)
    info_sum2 = np.sum(info2[:, lb_idx:ub_idx], -1)

    ax_scatter.scatter(info_sum1, info_sum2)
    ax_scatter.errorbar(np.mean(info_sum1), np.mean(info_sum2),
                 yerr=np.std(info_sum2),
                 xerr=np.std(info_sum1))

    plot_min = min(min(info_sum1), min(info_sum2))
    plot_max = max(max(info_sum1), max(info_sum2))
    ax_scatter.plot([plot_min, plot_max], [plot_min, plot_max], 'k--')
    ax_scatter.set_xlabel('Information %s (bits/sec)' % ts_names[0])
    ax_scatter.set_ylabel('Information %s (bits/sec)' % ts_names[1])

    snr_mean1 = np.mean(s_n_r1, 0)
    snr_mean2 = np.mean(s_n_r2, 0)

    ax_snr.plot(freqs, snr_mean1[lb_idx:ub_idx], label=ts_names[0])
    ax_snr.plot(freqs, snr_mean2[lb_idx:ub_idx], label=ts_names[1])
    ax_snr.legend()
    ax_snr.set_xlabel('Frequency (Hz)')
    ax_snr.set_ylabel('SNR')

    return fig, info1, info2, s_n_r1, s_n_r2


def plot_corr_diff(tseries1, tseries2, fig=None,
                  ts_names=['1', '2']):
    """
    Show the differences in *Fischer-transformed* snr correlations for two
    time-series

    Parameters
    ----------
    tseries1, tseries2 : nitime TimeSeries objects
       These are the time-series to compare, with each of them having the
       dims: (n_channels, n_reps, time), where n_channels1 = n_channels2

    lb,ub: float
       Lower and upper bounds on the frequency range over which to
       calculate the information rate (default to [0,Nyquist]).

    fig: matplotlib figure object
       If you want to do this on already existing figure. Otherwise, a new
       figure object will be generated.

    ts_names: list of str
       Labels for the two inputs, to be used in plotting (defaults to
       ['1','2'])

    bandwidth, adaptive, low_bias: See :func:`SNRAnalyzer` for details


    Returns
    -------

    fig: a matplotlib figure object
    """

    if fig is None:
        fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)

    SNR1 = []
    SNR2 = []
    corr1 = []
    corr2 = []
    corr_e1 = []
    corr_e2 = []

    for i in range(tseries1.shape[0]):
        SNR1.append(nta.SNRAnalyzer(ts.TimeSeries(tseries1.data[i],
                                    sampling_rate=tseries1.sampling_rate)))

        corr1.append(np.arctanh(np.abs(SNR1[-1].correlation[0])))
        corr_e1.append(SNR1[-1].correlation[1])

        SNR2.append(nta.SNRAnalyzer(ts.TimeSeries(tseries2.data[i],
                                    sampling_rate=tseries2.sampling_rate)))

        corr2.append(np.arctanh(np.abs(SNR2[-1].correlation[0])))
        corr_e2.append(SNR1[-1].correlation[1])

    ax.scatter(np.array(corr1), np.array(corr2))
    ax.errorbar(np.mean(corr1), np.mean(corr2),
                 yerr=np.std(corr2),
                 xerr=np.std(corr1))
    plot_min = min(min(corr1), min(corr2))
    plot_max = max(max(corr1), max(corr2))
    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--')
    ax.set_xlabel('Correlation (Fischer Z) %s' % ts_names[0])
    ax.set_ylabel('Correlation (Fischer Z) %s' % ts_names[1])

    return fig, corr1, corr2


def winspect(win, f, name=None):
    """
    Inspect a window by showing it and its spectrum

    Utility file used in building the documentation
    """
    npts = len(win)
    ax1, ax2 = f.add_subplot(1, 2, 1), f.add_subplot(1, 2, 2)
    ax1.plot(win)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Window amplitude')
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlim(0, npts)
    wf = fftpack.fft(win)
    ax1.set_xticks(np.arange(npts / 8., npts, npts / 8.))
    toplot = np.abs(fftpack.fftshift(wf).real)
    toplot /= np.max(toplot)
    toplot = np.log(toplot)
    ax2.plot(toplot, label=name)
    ax2.set_xlim(0, npts)
    ax2.set_xticks(np.arange(npts / 8., npts, npts / 8.))
    ax2.set_xticklabels(np.arange((-1 / 2. + 1 / 8.), 1 / 2., 1 / 8.))
    ax2.set_xlabel('Relative frequency')
    ax2.set_ylabel('Relative attenuation (log scale)')
    ax2.grid()
    ax2.legend(loc=4)
    f.set_size_inches([10, 6])


def plot_spectral_estimate(f, sdf, sdf_ests, limits=None, elabels=()):
    """
    Plot an estimate of a spectral transform against the ground truth.

    Utility file used in building the documentation
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax_limits = (sdf.min() - 2*np.abs(sdf.min()),
                 sdf.max() + 1.25*np.abs(sdf.max()))
    ax.plot(f, sdf, 'c', label='True S(f)')

    if not elabels:
        elabels = ('',) * len(sdf_ests)
    colors = 'bgkmy'
    for e, l, c in zip(sdf_ests, elabels, colors):
        ax.plot(f, e, color=c, linewidth=2, label=l)

    if limits is not None:
        ax.fill_between(f, limits[0], y2=limits[1], color=(1, 0, 0, .3),
                        alpha=0.5)

    ax.set_ylim(ax_limits)
    ax.legend()
    return fig
