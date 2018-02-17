import numpy as np
from nitime.lazy import scipy_stats as stats

from nitime import descriptors as desc
from nitime import utils as tsu
from nitime import algorithms as tsa
from nitime import timeseries as ts


class EventRelatedAnalyzer(desc.ResetMixin):
    """Analyzer object for reverse-correlation/event-related analysis.

    Note: right now, this class assumes the input time series is only
    two-dimensional.  If your input data is something like
    (nchannels,nsubjects, ...) with more dimensions, things are likely to break
    in hard to understand ways.
    """

    def __init__(self, time_series, events, len_et, zscore=False,
                 correct_baseline=False, offset=0):
        """
        Parameters
        ----------
        time_series : a time-series object
           A time-series with data on which the event-related analysis proceeds

        events_time_series : a TimeSeries object or an Events object
            The events which occurred in tandem with the time-series in the
            EventRelatedAnalyzer. This object's data has to have the same
            dimensions as the data in the EventRelatedAnalyzer object. In each
            sample in the time-series, there is an integer, which denotes the
            kind of event which occurred at that time. In time-bins in which no
            event occurred, a 0 should be entered. The data in this time series
            object needs to have the same dimensionality as the data in the
            data time-series

        len_et : int
            The expected length of the event-triggered quantity (in the same
            time-units as the events are represented (presumably number of TRs,
            for fMRI data). For example, the size of the block dedicated in the
            fir_matrix to each type of event

        zscore : a flag to return the result in zscore (where relevant)

        correct_baseline : a flag to correct the baseline according to the first
        point in the event-triggered average (where possible)

        offset : the offset of the beginning of the event-related time-series,
        relative to the event occurrence
        """
        #XXX Change so that the offset and length of the eta can be given in
        #units of time

        #Make sure that the offset and the len_et values can be used, by
        #padding with zeros before and after:

        if isinstance(events, ts.TimeSeries):
            #Set a flag to indicate the input is a time-series object:
            self._is_ts = True
            s = time_series.data.shape
            e_data = np.copy(events.data)

            #If the input is a one-dimensional (instead of an n-channel
            #dimensional) time-series, we will need to broadcast to make the
            #data assume the same number of dimensions as the time-series
            #input:
            if len(events.shape) == 1 and len(s) > 1:
                e_data = e_data + np.zeros((s[0], 1))

            zeros_before = np.zeros((s[:-1] + (int(offset),)))
            zeros_after = np.zeros((s[:-1] + (int(len_et),)))
            time_series_data = np.hstack([zeros_before,
                                          time_series.data,
                                          zeros_after])
            events_data = np.hstack([zeros_before,
                                     e_data,
                                     zeros_after])

            #If the events and the time_series have more than 1-d, the analysis
            #can traverse their first dimension
            if time_series.data.ndim - 1 > 0:
                self._len_h = time_series.data.shape[0]
                self.events = events_data
                self.data = time_series_data
            #Otherwise, in order to extract the array from the first dimension,
            #we wrap it in a list

            else:
                self._len_h = 1
                self.events = [events_data]
                self.data = [time_series_data]

        elif isinstance(events, ts.Events):
            self._is_ts = False
            s = time_series.data.shape
            zeros_before = np.zeros((s[:-1] + (abs(offset),)))
            zeros_after = np.zeros((s[:-1] + (abs(len_et),)))

            #If the time_series has more than 1-d, the analysis can traverse
            #the first dimension
            if time_series.data.ndim - 1 > 0:
                self._len_h = time_series.shape[0]
                self.data = time_series
                self.events = events

            #Otherwise, in order to extract the array from the first dimension,
            #we wrap it in a list
            else:
                self._len_h = 1
                self.data = [time_series]
                #No need to do that for the Events object:
                self.events = events
        else:
            err = ("Input 'events' to EventRelatedAnalyzer must be of type "
                   "Events or of type TimeSeries, %r given" % events)
            raise ValueError(err)

        self.sampling_rate = time_series.sampling_rate
        self.sampling_interval = time_series.sampling_interval
        self.len_et = int(len_et)
        self._zscore = zscore
        self._correct_baseline = correct_baseline
        self.offset = offset
        self.time_unit = time_series.time_unit

    @desc.setattr_on_read
    def FIR(self):
        """Calculate the FIR event-related estimated of the HRFs for different
        kinds of events

        Returns
        -------
        A time-series object, shape[:-2] are dimensions corresponding to the to
        shape[:-2] of the EventRelatedAnalyzer data, shape[-2] corresponds to
        the different kinds of events used (ordered according to the sorted
        order of the unique components in the events time-series). shape[-1]
        corresponds to time, and has length = len_et

        """
        # XXX code needs to be changed to use flattening (see 'eta' below)

        #Make a list to put the outputs in:
        h = [0] * self._len_h

        for i in range(self._len_h):
            #XXX Check that the offset makes sense (there can't be an event
            #happening within one offset duration of the beginning of the
            #time-series:

            #Get the design matrix (roll by the offset, in order to get the
            #right thing):

            roll_events = np.roll(self.events[i], self.offset)
            design = tsu.fir_design_matrix(roll_events, self.len_et)
            #Compute the fir estimate, in linear form:
            this_h = tsa.fir(self.data[i], design)
            #Reshape the linear fir estimate into a event_types*hrf_len array
            u = np.unique(self.events[i])
            event_types = u[np.unique(self.events[i]) != 0]
            h[i] = np.reshape(this_h, (event_types.shape[0], self.len_et))

        h = np.array(h).squeeze()

        return ts.TimeSeries(data=h,
                             sampling_rate=self.sampling_rate,
                             t0=self.offset * self.sampling_interval,
                             time_unit=self.time_unit)

    @desc.setattr_on_read
    def FIR_estimate(self):
        """Calculate back the LTI estimate of the time-series, from FIR"""
        raise NotImplementedError

    @desc.setattr_on_read
    def xcorr_eta(self):
        """Compute the normalized cross-correlation estimate of the HRFs for
        different kinds of events

        Returns
        -------

        A time-series object, shape[:-2] are dimensions corresponding to the to
        shape[:-2] of the EventRelatedAnalyzer data, shape[-2] corresponds to
        the different kinds of events used (ordered according to the sorted
        order of the unique components in the events time-series). shape[-1]
        corresponds to time, and has length = len_et (xcorr looks both back
        and forward for half of this length)

        """
        #Make a list to put the outputs in:
        h = [0] * self._len_h

        for i in range(self._len_h):
            data = self.data[i]
            u = np.unique(self.events[i])
            event_types = u[np.unique(self.events[i]) != 0]
            h[i] = np.empty((event_types.shape[0],
                             self.len_et // 2),
                            dtype=complex)
            for e_idx in range(event_types.shape[0]):
                this_e = (self.events[i] == event_types[e_idx]) * 1.0
                if self._zscore:
                    this_h = tsa.freq_domain_xcorr_zscored(
                                                data,
                                                this_e,
                                                -self.offset + 1,
                                                self.len_et - self.offset - 2)
                else:
                    this_h = tsa.freq_domain_xcorr(
                                                data,
                                                this_e,
                                                -self.offset + 1,
                                                self.len_et - self.offset - 2)
                h[i][e_idx] = this_h

        h = np.array(h).squeeze()

        ## t0 for the object returned here needs to be the central time, not
        ## the first time point, because the functions 'look' back and forth
        ## for len_et bins

        return ts.TimeSeries(data=h,
                             sampling_rate=self.sampling_rate,
                             t0=-1 * self.len_et * self.sampling_interval,
                             time_unit=self.time_unit)

    @desc.setattr_on_read
    def et_data(self):
        """The event-triggered data (all occurrences).

        This gets the time-series corresponding to the inidividual event
        occurrences. Returns a list of lists of time-series. The first dimension
        is the different channels in the original time-series data and the
        second dimension is each type of event in the event time series

        The time-series itself has the first diemnsion of the data being the
        specific occurrence, with time 0 locked to the that occurrence
        of the event and the last dimension is time.e

        This complicated structure is so that it can deal with situations where
        each channel has different events and different events have different #
        of occurrences
        """
        #Make a list for the output
        h = [0] * self._len_h

        for i in range(self._len_h):
            data = self.data[i]
            u = np.unique(self.events[i])
            event_types = u[np.unique(self.events[i]) != 0]
            #Make a list in here as well:
            this_list = [0] * event_types.shape[0]
            for e_idx in range(event_types.shape[0]):
                idx = np.where(self.events[i] == event_types[e_idx])

                idx_w_len = np.array([idx[0] + count + self.offset for count
                                      in range(self.len_et)])
                event_trig = data[idx_w_len].T
                this_list[e_idx] = ts.TimeSeries(data=event_trig,
                                 sampling_interval=self.sampling_interval,
                                 t0=self.offset * self.sampling_interval,
                                 time_unit=self.time_unit)

            h[i] = this_list

        return h

    @desc.setattr_on_read
    def eta(self):
        """The event-triggered average activity.
        """
        #Make a list for the output
        h = [0] * self._len_h

        if self._is_ts:
            # Loop over channels
            for i in range(self._len_h):
                data = self.data[i]
                u = np.unique(self.events[i])
                event_types = u[np.unique(self.events[i]) != 0]
                h[i] = np.empty((event_types.shape[0], self.len_et),
                                dtype=complex)

                # This offset is used to pull the event indices below, but we
                # have to broadcast it so the shape of the resulting idx+offset
                # operation below gives us the (nevents, len_et) array we want,
                # per channel.
                offset = np.arange(self.offset,
                                   self.offset + self.len_et)[:, np.newaxis]
                # Loop over event types
                for e_idx in range(event_types.shape[0]):
                    idx = np.where(self.events[i] == event_types[e_idx])[0]
                    event_trig = data[idx + offset]
                    #Correct baseline by removing the first point in the series
                    #for each channel:
                    if self._correct_baseline:
                        event_trig -= event_trig[0]

                    h[i][e_idx] = np.mean(event_trig, -1)

        #In case the input events are an Events:
        else:
            #Get the indices necessary for extraction of the eta:
            add_offset = np.arange(self.offset,
                                   self.offset + self.len_et)[:, np.newaxis]

            idx = (self.events.time / self.sampling_interval).astype(int)

            #Make a list for the output
            h = [0] * self._len_h

            # Loop over channels
            for i in range(self._len_h):
                #If this is a list with one element:
                if self._len_h == 1:
                    event_trig = self.data[0][idx + add_offset]
                #Otherwise, you need to index straight into the underlying data
                #array:
                else:
                    event_trig = self.data.data[i][idx + add_offset]

                h[i] = np.mean(event_trig, -1)

        h = np.array(h).squeeze()
        return ts.TimeSeries(data=h,
                             sampling_interval=self.sampling_interval,
                             t0=self.offset * self.sampling_interval,
                             time_unit=self.time_unit)

    @desc.setattr_on_read
    def ets(self):
        """The event-triggered standard error of the mean """

        #Make a list for the output
        h = [0] * self._len_h

        if self._is_ts:
            # Loop over channels
            for i in range(self._len_h):
                data = self.data[i]
                u = np.unique(self.events[i])
                event_types = u[np.unique(self.events[i]) != 0]
                h[i] = np.empty((event_types.shape[0], self.len_et),
                                dtype=complex)

                # This offset is used to pull the event indices below, but we
                # have to broadcast it so the shape of the resulting idx+offset
                # operation below gives us the (nevents, len_et) array we want,
                # per channel.
                offset = np.arange(self.offset,
                                   self.offset + self.len_et)[:, np.newaxis]
                # Loop over event types
                for e_idx in range(event_types.shape[0]):
                    idx = np.where(self.events[i] == event_types[e_idx])[0]
                    event_trig = data[idx + offset]
                    #Correct baseline by removing the first point in the series
                    #for each channel:
                    if self._correct_baseline:
                        event_trig -= event_trig[0]

                    h[i][e_idx] = stats.sem(event_trig, -1)

        #In case the input events are an Events:
        else:
            #Get the indices necessary for extraction of the eta:
            add_offset = np.arange(self.offset,
                                   self.offset + self.len_et)[:, np.newaxis]

            idx = (self.events.time / self.sampling_interval).astype(int)

            #Make a list for the output
            h = [0] * self._len_h

            # Loop over channels
            for i in range(self._len_h):
                #If this is a list with one element:
                if self._len_h == 1:
                    event_trig = self.data[0][idx + add_offset]
                #Otherwise, you need to index straight into the underlying data
                #array:
                else:
                    event_trig = self.data.data[i][idx + add_offset]

                h[i] = stats.sem(event_trig, -1)

        h = np.array(h).squeeze()
        return ts.TimeSeries(data=h,
                             sampling_interval=self.sampling_interval,
                             t0=self.offset * self.sampling_interval,
                             time_unit=self.time_unit)
