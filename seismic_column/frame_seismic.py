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

Both hinges are sought **in the column**.  The shaft is a capacity-protected
element: it is not a section competing to hinge, it is one the designer keeps
elastic by sizing it for the column's overstrength demand.  Letting it compete
would answer a shaft-SIZING question in the wrong place, and — because Dp
follows the hinge spacing — would also credit the member with the long
deck-to-fixity lever it does not have.

What the mechanism yields instead is that design demand, and for a fixed-fixed
member it is not the fixed-free one.  Both hinges of a column mechanism sit at
``Mp_col``, so the interface MOMENT is the same ``Mo``; but the mechanism shear
is ``2*Mp/H`` against ``Mp/H``, i.e. **twice** the overstrength shear the
stand-alone cantilever was sized for.  That is reported per member so the p-y
in-ground demand can be re-run at the right head condition.

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
    # Capacity-design demand the SHAFT must then be sized for, at the mechanism.
    Mo_interface: float = 0.0   # overstrength moment at the top of shaft, kip-in
    Vo_interface: float = 0.0   # overstrength shear there, kip
    Vo_cantilever: float = 0.0  # what the fixed-free suite designed it for, kip
    # Below-ground shaft demand at THIS mechanism's head condition (p-y).
    # Zero for a fixed-free member: there the mechanism is the cantilever the
    # per-bent suite already solved, so its numbers stand.
    shaft_moment: float = 0.0   # max |M| below ground, kip-in
    shaft_shear: float = 0.0    # max |V| below ground, kip
    shaft_Mp: float = 0.0       # shaft plastic moment available, kip-in
    shaft_solution: object = None
    warnings: list[str] = field(default_factory=list)

    @property
    def shaft_dc(self) -> float:
        """Below-ground flexural demand/capacity. >1 means the shaft yields."""
        return (self.shaft_moment / self.shaft_Mp if self.shaft_Mp > 0
                else float("nan"))

    @property
    def first_hinge(self) -> Section:
        return self.hinges[0]

    @property
    def mechanism(self) -> str:
        return " + ".join(h.name for h in self.hinges)

    @property
    def shear_amplification(self) -> float:
        """Vo here against the fixed-free Vo the shaft was designed for."""
        return (self.Vo_interface / self.Vo_cantilever
                if self.Vo_cantilever > 0 else float("nan"))

    @property
    def ratio(self) -> float:
        return self.delta_c / self.delta_d if self.delta_d > 0 else float("nan")

    @property
    def passed(self) -> bool:
        return (self.ratio >= 1.0 and math.isfinite(self.mu_d)
                and self.pdelta_ok)


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
    """Redistribute past the first hinge and find the second, IN THE COLUMN.

    With the first hinge pinned at its capacity the diagram becomes
    ``M(x) = V*(x - x1) + sigma*Mp1``, so each remaining section yields at
    ``V = (±Mp - sigma*Mp1) / (x - x1)``.  The mechanism forms at the smallest
    such V that is not below the load which formed the first hinge.

    The shaft is **excluded as a candidate**.  It is a capacity-protected
    element: it is not a section that competes to hinge, it is one the designer
    is required to keep elastic by sizing it for the column's overstrength
    demand.  Asking whether the as-entered ``Mp_shaft`` would yield first
    answers a shaft-SIZING question, and answering it here would both fail the
    displacement check for the wrong reason and — because the plastic
    displacement follows the hinge spacing — credit the member with the long
    deck-to-fixity lever it does not have.  The demand the shaft must then be
    designed for is reported separately by :func:`shaft_demand`.
    """
    sigma = 1.0 if first.x >= contraflexure else -1.0
    best, best_V = None, float("inf")
    for s in sections:
        if s.name == FIXITY:               # capacity-protected, cannot hinge
            continue
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


def _shaft_demand(a, Mo_int: float, Vo_int: float, soil_bounds, warn: list):
    """Below-ground shaft demand at the fixed-fixed mechanism's head condition.

    The column mechanism holds the top of shaft at ``Mp_col``, so the shaft head
    sees the SAME overstrength moment as the cantilever case but twice the
    shear.  Passing the shear alone would put ``2*Mo`` at the interface and
    overstate the demand, hence the head moment.

    The head-moment sign is verified, not assumed: the solver's DOF-1 convention
    is the opposite of the obvious hand derivation, and getting it wrong lands
    ``3*Mo`` at the interface — a 3x error that would look plausible in a table.
    So the returned interface moment is checked against ``Mo`` and the result
    discarded if it disagrees.
    """
    from .sdc_capacity import inground_demand      # circular at module level

    if a.soil_profile is None:                     # multiplier-based fixity
        return 0.0, 0.0, None
    M, V, sol = inground_demand(
        a.H_free, a.shaft_embed_length, a.EI_col, a.EI_shaft, a.shaft_D,
        a.P_used, a.soil_profile, Vo_int, soil_bounds, M_head=Mo_int)
    if sol is None:
        warn.append("the p-y solve for the fixed-fixed shaft demand was "
                    "unstable on every soil bound — below-ground demand not "
                    "established")
        return 0.0, 0.0, None
    at_interface = abs(sol.moment[sol.ground_index])
    if abs(at_interface - Mo_int) > 0.10 * Mo_int:
        warn.append(
            f"the fixed-fixed shaft solve was discarded: it put "
            f"{at_interface/12:.0f} kip-ft at the interface against the "
            f"{Mo_int/12:.0f} kip-ft the mechanism requires, so the head "
            f"condition is not being applied as intended")
        return 0.0, 0.0, None
    return M, V, sol


def check_frame(frame, direction: str, bound: int, assessments: dict, spectrum,
                provisions, soil_bounds=(2.0, 0.5)) -> FrameCheck:
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
        ef = b.end_fixity(direction)
        Mp_col, Mp_shaft = a.mc_col.Mp, a.mc_shaft.Mp
        secs = _sections(geom, a.EI_col, a.EI_shaft, bb.multiplier,
                         Mp_col, Mp_shaft, ef)
        # The shaft is capacity-protected, so it is not a candidate for EITHER
        # hinge -- including the first.  A shaft weak enough to win the elastic
        # race is a shaft that needs enlarging, which the demand table reports;
        # it is not a hinge the mechanism is allowed to use.
        first = min((s for s in secs if s.name != FIXITY),
                    key=lambda s: s.V_yield)
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
                            "of shaft — so the deck joint, not only the shaft, "
                            "has to be capacity-protected. The second hinge is "
                            "still in the column, so the Type II premise holds")

        # The shaft is held elastic by design, so what it must be sized for is
        # the mechanism's overstrength demand at the interface.  Both hinges of
        # a column mechanism sit at Mp_col, so the interface MOMENT is the same
        # Mo the fixed-free suite already used — but the SHEAR is not.
        lam = provisions.overstrength_factor
        Mo_int, Vo_int = lam * Mp_col, lam * V_mech
        Vo_cant = lam * Mp_col / H
        sh_M = sh_V = 0.0
        sh_sol = None
        if ef != "free" and Vo_int > Vo_cant * 1.01:
            warn.append(
                f"the shaft is capacity-designed for this mechanism: same "
                f"Mo = {Mo_int/12:.0f} kip-ft at the interface, but "
                f"Vo = {Vo_int:.0f} kip — {Vo_int/Vo_cant:.2f}× the "
                f"{Vo_cant:.0f} kip the fixed-free suite sized it for")
            sh_M, sh_V, sh_sol = _shaft_demand(a, Mo_int, Vo_int, soil_bounds,
                                               warn)
            if sh_sol is not None and sh_M > a.mc_shaft.Mp:
                warn.append(
                    f"the SHAFT would yield below ground under this mechanism: "
                    f"M = {sh_M/12:.0f} kip-ft against Mp_shaft = "
                    f"{a.mc_shaft.Mp/12:.0f} kip-ft, D/C = "
                    f"{sh_M/a.mc_shaft.Mp:.2f}. Reported, not failed — the "
                    f"fixed-fixed mechanism is a closed-form idealisation; a "
                    f"pushover should settle it before the shaft is resized")

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
            Mo_interface=Mo_int, Vo_interface=Vo_int, Vo_cantilever=Vo_cant,
            shaft_moment=sh_M, shaft_shear=sh_V, shaft_Mp=a.mc_shaft.Mp,
            shaft_solution=sh_sol,
            warnings=warn))
    return fc


def check_all(balance, assessments: dict, spectrum, provisions,
              continuous_only: bool = True,
              soil_bounds=(2.0, 0.5)) -> list[FrameCheck]:
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
                                       spectrum, provisions, soil_bounds))
    return out
