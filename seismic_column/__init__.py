"""seismic_column

Pure-Python tool to analyze and optimize circular reinforced-concrete columns
supported on Type II (enlarged) shafts for seismic checks per Caltrans SDC 2.1
using the Equivalent Static Analysis (ESA) method.

Units convention (US customary) used consistently throughout the package:
    force    : kip
    length   : in
    stress   : ksi
    mass     : kip*s^2/in
    accel.   : in/s^2  (g = 386.088 in/s^2)
"""

G_IN_S2 = 386.088  # gravitational acceleration, in/s^2

__all__ = ["G_IN_S2"]
