"""Displacement check of a FRAME, on the frame's own period and mechanism.

The per-bent suite in :mod:`sdc_capacity` treats every column as a stand-alone
fixed-free cantilever: one hinge at the top of shaft, demand from that bent's own
SDOF period.  For bents that are integral with a continuous deck neither half of
that is right longitudinally.

**Demand.**  The members of a frame sway together, so the frame has one period::

    K_frame = sum k_i     (each member at ITS end condition in this direction)
    M_frame = sum m_i
    T_frame = 2*pi*sqrt(M_frame / K_frame)

and, taking the deck as rigid, every member is pushed to the same displacement.

**Capacity.**  This is where the end condition really bites.  With x measured
down from the deck to the point of fixity, statics gives a LINEAR moment
diagram::

    M(x) = V*x + M0
    fixed-free    M0 = 0            -> M peaks at the base
    fixed-fixed   M0 = -V*B/A       -> contraflexure at x = B/A

with ``A = int(dx/EI)`` and ``B = int(x dx/EI)`` over the two segments.  The
first hinge forms where ``M / Mp`` peaks, and Mp is NOT constant: the column
carries ``Mp_col`` while the capacity-protected shaft below carries far more.
So the candidate sections are the deck, the top of shaft, and the point of
fixity, and which one governs has to be read off the diagram rather than
assumed.

A fixed-free member is determinate in sway, so ONE hinge is already a mechanism.
A fixed-fixed member is indeterminate to degree one, so it takes TWO, and the
second one has to be found by redistribution rather than assumed at the other
end.  After the first hinge at ``x1`` the moment there is pinned to its
capacity, so the diagram becomes::

    M(x) = V*(x - x1) + sigma*Mp1          sigma = sign of M at the first hinge

and the second hinge forms at whichever remaining section first satisfies
``|M(x)| = Mp(x)``.  The mechanism load is the V that does it, and the plastic
displacement follows the hinge SPACING::

    two hinges   Dp = theta_p * (x2 - x1 - Lp)
    cantilever   Dp = theta_p * (H_free - Lp/2)

A hinge that lands in the shaft rather than the column is reported as a
**failure**, not as capacity: Type II detailing exists precisely to keep the
hinge in the column above, so a mechanism that relies on the shaft yielding
violates the premise the whole design rests on.

**What this is not.**  It is closed-form plastic analysis, not an incremental
pushover.  It gives the section that yields first and the load at which a
mechanism forms; it does not trace the sequence, redistribution, or the
post-mechanism response.  Where the answer matters, run the real pushover.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .demand import displacement_demand, magnified_demand
from .geometry import Geometry

DECK = "deck (column top)"
TOP_OF_SHAFT = "top of shaft (column)"
FIXITY = "point of fixity (shaft)"


@dataclass
class Section:
    """One candidate hinge location on the moment diagram."""

    name: str
    x: float                 # in, measured down from the deck
    arm: float               # |M| per unit shear at this section, in
    Mp: float                # plastic moment available here, kip-in
    V_yield: float           # shear that brings this section to Mp, kip


@dataclass
class MemberCheck:
    """One member of the frame, checked against the frame displacement."""

    name: str
    end_fixity: str
    k: float                 # this member's contribution to K_frame, kip/in
    m: float                 # ... and to M_frame, kip*s^2/in
    sections: list[Section]
    hinges: list[Section]    # in order of formation; 1 fixed-free, 2 fixed-fixed
    V_mech: float            # shear at which the mechanism forms, kip
    delta_y: float
    delta_p: float
    delta_c: float
    delta_d: float
    mu_d: float
    pdelta_ok: bool
    type_ii_ok: bool = True  # every hinge stays in the column
    warnings: list[str] = field(default_factory=list)

    @property
    def first_hinge(self) -> Section:
        return self.hinges[0]

    @property
    def mechanism(self) -> str:
        return " + ".join(h.name for h in self.hinges)

    @property
    def ratio(self) -> float:
        return self.delta_c / self.delta_d if self.delta_d > 0 else float("nan")

    @property
    def passed(self) -> bool:
        return (self.ratio >= 1.0 and math.isfinite(self.mu_d)
                and self.pdelta_ok and self.type_ii_ok)


@dataclass
class FrameCheck:
    """A frame, one direction, one fixity bound."""

    frame_key: str
    direction: str
    bound_label: str
    member_names: tuple[str, ...]
    end_conditions: str
    K: float
    M: float
    T: float
    Sa: float
    delta_d: float
    members: list[MemberCheck] = field(default_factory=list)

    @property
    def W(self) -> float:
        return self.M * 386.088

    @property
    def passed(self) -> bool:
        return all(m.passed for m in self.members)

    @property
    def worst(self) -> float:
        return min((m.ratio for m in self.members), default=float("nan"))


def _sections(geom: Geometry, EI_col: float, EI_shaft: float, multiplier: float,
              Mp_col: float, Mp_shaft: float, end_fixity: str) -> list[Section]:
    """Candidate hinge sections with the shear each needs to reach its Mp."""
    A, B, _ = geom._ei_moments(EI_col, EI_shaft, multiplier)
    H = geom.H_free
    L = H + geom.fixity_depth(multiplier)
    M0 = 0.0 if end_fixity == "free" else -B / A     # per unit shear
    out = []
    for name, x, Mp in ((DECK, 0.0, Mp_col),
                        (TOP_OF_SHAFT, H, Mp_col),
                        (FIXITY, L, Mp_shaft)):
        arm = abs(x + M0)
        out.append(Section(name, x, arm,
                           Mp, Mp / arm if arm > 1e-9 else float("inf")))
    return out


def _second_hinge(sections: list[Section], first: Section, V1: float,
                  contraflexure: float) -> tuple[Section, float]:
    """Redistribute past the first hinge and find the second.

    With the first hinge pinned at its capacity the diagram becomes
    ``M(x) = V*(x - x1) + sigma*Mp1``, so each remaining section yields at
    ``V = (±Mp - sigma*Mp1) / (x - x1)``.  The mechanism forms at the smallest
    such V that is not below the load which formed the first hinge.
    """
    sigma = 1.0 if first.x >= contraflexure else -1.0
    best, best_V = None, float("inf")
    for s in sections:
        dx = s.x - first.x
        if abs(dx) < 1e-9:
            continue
        for target in (s.Mp, -s.Mp):
            V = (target - sigma * first.Mp) / dx
            if V >= V1 - 1e-9 and V < best_V:
                best, best_V = s, V
    if best is None:                       # degenerate; fall back to the first
        return first, V1
    return best, best_V


def check_frame(frame, direction: str, bound: int, assessments: dict, spectrum,
                provisions) -> FrameCheck:
    """Check every member of ``frame`` against the frame's displacement demand.

    ``assessments`` maps pier name -> ``ColumnAssessment``.  Each member is taken
    at its own end condition in this direction, so a mixed frame (an integral
    bent beside a bearing) is handled member by member.
    """
    K = sum(b.stiffness(direction, bound) for b in frame.members)
    M = frame.M()
    T = 2.0 * math.pi * math.sqrt(M / K) if K > 0 and M > 0 else float("nan")
    dem = displacement_demand(spectrum, K, M * 386.088)

    fc = FrameCheck(frame_key=frame.key, direction=direction,
                    bound_label=frame.label(bound),
                    member_names=frame.names,
                    end_conditions=frame.end_conditions,
                    K=K, M=M, T=T, Sa=dem.Sa, delta_d=dem.disp_demand)

    for b in frame.members:
        a = assessments[b.name]
        bb = a.bounds[bound]
        geom = Geometry(Hcol=a.Hcol_entered, D_shaft=a.shaft_D, silo=a.silo)
        H = geom.H_free
        L = H + bb.fixity_depth
        ef = b.end_fixity(direction)
        Mp_col, Mp_shaft = a.mc_col.Mp, a.mc_shaft.Mp
        secs = _sections(geom, a.EI_col, a.EI_shaft, bb.multiplier,
                         Mp_col, Mp_shaft, ef)
        first = min(secs, key=lambda s: s.V_yield)
        warn: list[str] = []

        theta_p = a.Lp * (a.mc_col.phi_u - a.mc_col.phi_y)
        flex = geom.tip_flexibility(a.EI_col, a.EI_shaft, bb.multiplier,
                                    end_fixity=ef)

        if ef == "free":
            # determinate in sway, so the first hinge IS the mechanism
            hinges, V_mech = [first], first.V_yield
            delta_p = theta_p * (H - a.Lp / 2.0)
            if first.name != TOP_OF_SHAFT:
                warn.append(f"the moment diagram peaks at the {first.name}, not "
                            f"the top of shaft")
        else:
            # indeterminate: redistribute past the first hinge to find the second
            A_, B_, _ = geom._ei_moments(a.EI_col, a.EI_shaft, bb.multiplier)
            second, V_mech = _second_hinge(secs, first, first.V_yield, B_ / A_)
            hinges = [first, second]
            delta_p = theta_p * max(abs(second.x - first.x) - a.Lp, 0.0)
            if first.name == DECK:
                warn.append("first yield is at the DECK connection, not the top "
                            "of shaft — the joint needs capacity protection, and "
                            "the shaft demand no longer follows from Mo at the "
                            "interface")

        if any(h.name == FIXITY for h in hinges):
            warn.append("the sway mechanism relies on a plastic hinge IN THE "
                        "SHAFT, which Type II detailing is meant to prevent — "
                        "reported as a failure, not as capacity")

        delta_y = V_mech * flex
        d = dem
        if provisions.short_period_magnification:
            d = magnified_demand(dem, spectrum, delta_y)
        delta_d = d.disp_demand
        delta_c = delta_y + delta_p
        fc.members.append(MemberCheck(
            name=b.name, end_fixity=ef,
            k=b.stiffness(direction, bound), m=b.mass(direction),
            sections=secs, hinges=hinges,
            V_mech=V_mech, delta_y=delta_y, delta_p=delta_p,
            delta_c=delta_c, delta_d=delta_d,
            mu_d=delta_d / delta_y if delta_y > 0 else float("nan"),
            pdelta_ok=(a.P_used * delta_d
                       <= provisions.pdelta_factor * a.mc_col.Mp),
            type_ii_ok=all(h.name != FIXITY for h in hinges),
            warnings=warn))
    return fc


def check_all(balance, assessments: dict, spectrum, provisions,
              continuous_only: bool = True) -> list[FrameCheck]:
    """Every frame worth checking, both directions, every fixity bound.

    Single-bent frames reduce to the per-bent cantilever the seismic suite
    already runs — only the mass and the demand basis differ — so by default
    only CONTINUOUS frames are reported.
    """
    out = []
    for direction, frames in balance.frames.items():
        for f in frames:
            if continuous_only and not f.continuous:
                continue
            for bound in range(f.n_bounds):
                out.append(check_frame(f, direction, bound, assessments,
                                       spectrum, provisions))
    return out
