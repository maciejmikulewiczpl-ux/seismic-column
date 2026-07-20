"""Type II shaft geometry and the two-segment equivalent cantilever.

A single-column bent on a Type II (enlarged) shaft is idealised as a cantilever
fixed at an assumed point of fixity a depth ``Df`` below the top of the shaft,
with the lateral load applied at the top of the column.  The plastic hinge forms
in the *column* at the top of the shaft (the shaft is capacity-protected).

The equivalent cantilever has two segments with different cracked flexural
rigidities:

    * column segment   : length ``Hcol``      , rigidity ``EI_col``
    * shaft segment     : length ``Df``         , rigidity ``EI_shaft``

The depth to the point of fixity is ``Df = multiplier * D_shaft`` where the
multiplier is typically bracketed (default 3 = upper-bound stiffness, 6 =
lower-bound stiffness).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Geometry:
    """Column-on-Type-II-shaft geometry.

    Parameters
    ----------
    Hcol:
        Column height from top of shaft to the point of load application
        (point of contraflexure), in.
    D_shaft:
        Shaft diameter, in.
    """

    Hcol: float
    D_shaft: float

    def fixity_depth(self, multiplier: float) -> float:
        """Depth from top of shaft to the point of fixity, in."""
        return multiplier * self.D_shaft

    def effective_length(self, multiplier: float) -> float:
        """Equivalent cantilever length to the point of fixity, in."""
        return self.Hcol + self.fixity_depth(multiplier)

    def tip_flexibility(self, EI_col: float, EI_shaft: float, multiplier: float) -> float:
        """Lateral flexibility (tip displacement per unit lateral load), in/kip.

        Uses the unit-load method for a cantilever with a point load at the top,
        M(x) = F*x, integrated over the two segments:

            d/F = (1/EI_col) * Hcol^3/3
                + (1/EI_shaft) * ((Hcol+Df)^3 - Hcol^3)/3
        """
        Df = self.fixity_depth(multiplier)
        Le = self.Hcol + Df
        term_col = (self.Hcol ** 3) / (3.0 * EI_col)
        term_shaft = (Le ** 3 - self.Hcol ** 3) / (3.0 * EI_shaft)
        return term_col + term_shaft

    def lateral_stiffness(self, EI_col: float, EI_shaft: float, multiplier: float) -> float:
        """Lateral stiffness of the equivalent cantilever, kip/in."""
        return 1.0 / self.tip_flexibility(EI_col, EI_shaft, multiplier)
