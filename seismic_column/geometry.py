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

A **column silo** (isolation casing) lowers the top of shaft by ``silo`` and adds
that much free column length, so the free length used by every mechanics
calculation is ``H_free = Hcol + silo``.  The plastic hinge stays at the top of
shaft, i.e. at the bottom of the silo.  This is the "adjust effective column
lengths (lower footings using isolation casing)" tuning technique of Caltrans
SDC 2.1 C7.1.2 / AASHTO SGS 4.1.4, used here to satisfy the balanced-stiffness
and balanced-frame-geometry rules between adjacent piers.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Geometry:
    """Column-on-Type-II-shaft geometry.

    Parameters
    ----------
    Hcol:
        Column height from the original ground/soffit line to the point of load
        application (point of contraflexure), in.  This is the value the user
        enters; it excludes any column silo.
    D_shaft:
        Shaft diameter, in.
    silo:
        Column silo (isolation casing) depth, in.  Lowers the top of shaft by
        this much and lengthens the free column by the same amount; the embedded
        shaft length is unchanged (the tip goes deeper).  0 = no silo.
    """

    Hcol: float
    D_shaft: float
    silo: float = 0.0

    @property
    def H_free(self) -> float:
        """Free column length above the top of shaft, in.

        ``Hcol + silo`` — the length that governs Lp, the self-weight, the
        displacement capacity and the shear/P-Delta demands.  Equals ``Hcol``
        when there is no silo.
        """
        return self.Hcol + self.silo

    def fixity_depth(self, multiplier: float) -> float:
        """Depth from top of shaft to the point of fixity, in."""
        return multiplier * self.D_shaft

    def effective_length(self, multiplier: float) -> float:
        """Equivalent cantilever length to the point of fixity, in."""
        return self.H_free + self.fixity_depth(multiplier)

    def tip_flexibility(self, EI_col: float, EI_shaft: float, multiplier: float) -> float:
        """Lateral flexibility (tip displacement per unit lateral load), in/kip.

        Uses the unit-load method for a cantilever with a point load at the top,
        M(x) = F*x, integrated over the two segments:

            d/F = (1/EI_col) * H_free^3/3
                + (1/EI_shaft) * ((H_free+Df)^3 - H_free^3)/3
        """
        H = self.H_free
        Df = self.fixity_depth(multiplier)
        Le = H + Df
        term_col = (H ** 3) / (3.0 * EI_col)
        term_shaft = (Le ** 3 - H ** 3) / (3.0 * EI_shaft)
        return term_col + term_shaft

    def lateral_stiffness(self, EI_col: float, EI_shaft: float, multiplier: float) -> float:
        """Lateral stiffness of the equivalent cantilever, kip/in."""
        return 1.0 / self.tip_flexibility(EI_col, EI_shaft, multiplier)
