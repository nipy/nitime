====================
 Time-series access
====================

`func`:ts.at: : access with a time-point will give you back an array

AND access with an array of time-points will give you back a non-uniform

Using intervals (see `ref`:interval_object_discussion.rst: ), will give you
back a uniform time-series objects with the time being of length of
t_start-t_end and with the ts.t0 offset by the intervals t_offset.

Access with integers? Maybe make an additional method which indexes into t

We would need a function: `func`:ts.time2index: which would generate the
integer index into the data, based on input which is the time.


