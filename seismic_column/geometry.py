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

    def _ei_moments(self, EI_col: float, EI_shaft: float, multiplier: float):
        """``(A, B, C)`` = ∫dx/EI, ∫x·dx/EI, ∫x²·dx/EI over the two segments.

        ``x`` runs from the top of the column down to the point of fixity, so
        the column contributes over ``0…H_free`` and the shaft over
        ``H_free…H_free+Df``.  Every flexibility below is built from these.
        """
        H = self.H_free
        L = H + self.fixity_depth(multiplier)
        A = H / EI_col + (L - H) / EI_shaft
        B = H ** 2 / (2.0 * EI_col) + (L ** 2 - H ** 2) / (2.0 * EI_shaft)
        C = H ** 3 / (3.0 * EI_col) + (L ** 3 - H ** 3) / (3.0 * EI_shaft)
        return A, B, C

    def tip_flexibility(self, EI_col: float, EI_shaft: float, multiplier: float,
                        end_fixity: str = "free") -> float:
        """Lateral flexibility (tip displacement per unit lateral load), in/kip.

        ``end_fixity`` is the rotational restraint at the **top** of the column:

        ``"free"`` — a cantilever, free to rotate at the deck.  Unit-load method
        with ``M(x) = F·x``::

            d/F = C = H³/3EI_col + ((H+Df)³ − H³)/3EI_shaft

        ``"fixed"`` — rotation restrained at the deck, i.e. an **integral**
        (monolithic, moment-connected) bent swaying longitudinally, where the
        deep continuous girder holds the column head against rotation.  Solving
        the redundant head moment ``M₀`` from zero head rotation
        (``V·B + M₀·A = 0``) and substituting back gives::

            d/F = C − B²/A

        For a prismatic member those reduce to ``L³/3EI`` and ``L³/12EI``, i.e.
        Caltrans SDC C7.1.2-1 ``ke = 3EcIeff/L³`` and C7.1.2-2
        ``ke = 12EcIeff/L³`` — a factor of 4 — and the stepped column-on-shaft
        member lands between those two limits.

        *Caveat:* the depth to fixity ``Df`` is itself derived for a free-head
        member (a 3×/6× multiplier, or a free-head p-y solve).  Re-using it with
        a fixed head is the usual simplification, not an exact equivalence.
        """
        A, B, C = self._ei_moments(EI_col, EI_shaft, multiplier)
        if end_fixity == "free":
            return C
        if end_fixity == "fixed":
            return C - B * B / A
        raise ValueError("end_fixity must be 'free' or 'fixed'")

    def lateral_stiffness(self, EI_col: float, EI_shaft: float,
                          multiplier: float, end_fixity: str = "free") -> float:
        """Lateral stiffness of the equivalent cantilever, kip/in."""
        return 1.0 / self.tip_flexibility(EI_col, EI_shaft, multiplier,
                                          end_fixity)
