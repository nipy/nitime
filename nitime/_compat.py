# np.trapezoid was introduced and np.trapz deprecated in numpy 2.0
try:  # NP2
    from numpy import trapezoid
except ImportError:  # NP1
    from numpy import trapz as trapezoid
