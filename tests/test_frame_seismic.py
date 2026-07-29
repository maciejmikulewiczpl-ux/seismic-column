"""Frame-level displacement check: the mechanism, not just the plumbing."""
from __future__ import annotations

import math

import pytest

from seismic_column.balance import BentStiffness, Frame
from seismic_column.frame_seismic import (DECK, FIXITY, TOP_OF_SHAFT, Section,
                                          _second_hinge, _sections, check_frame)
from seismic_column.geometry import Geometry
from seismic_column.io_schema import LONGITUDINAL, TRANSVERSE


# ---------------------------------------------------------------- moment diagram
def _geom(Hcol=300.0, D_shaft=120.0, silo=0.0):
    return Geometry(Hcol=Hcol, D_shaft=D_shaft, silo=silo)


def test_fixed_free_moment_diagram_peaks_at_the_base():
    """M0 = 0, so the lever grows monotonically with depth."""
    g = _geom()
    secs = _sections(g, 1e9, 3e9, 3.0, Mp_col=1e5, Mp_shaft=3e5,
                     end_fixity="free")
    by = {s.name: s for s in secs}
    assert by[DECK].arm == pytest.approx(0.0)
    assert by[TOP_OF_SHAFT].arm == pytest.approx(g.H_free)
    assert by[FIXITY].arm > by[TOP_OF_SHAFT].arm
    # a zero lever cannot yield
    assert math.isinf(by[DECK].V_yield)


def test_fixed_fixed_puts_contraflexure_between_the_ends():
    """The deck lever is B/A and it must sit inside the member."""
    g = _geom()
    A, B, _ = g._ei_moments(1e9, 3e9, 3.0)
    secs = _sections(g, 1e9, 3e9, 3.0, Mp_col=1e5, Mp_shaft=3e5,
                     end_fixity="fixed")
    by = {s.name: s for s in secs}
    L = g.H_free + g.fixity_depth(3.0)
    assert by[DECK].arm == pytest.approx(B / A)
    assert 0.0 < B / A < L
    # the deck now carries moment, so it is a real candidate
    assert math.isfinite(by[DECK].V_yield)


def test_fixed_fixed_levers_straddle_the_contraflexure():
    g = _geom()
    A, B, _ = g._ei_moments(1e9, 3e9, 3.0)
    c = B / A
    secs = _sections(g, 1e9, 3e9, 3.0, Mp_col=1e5, Mp_shaft=3e5,
                     end_fixity="fixed")
    by = {s.name: s for s in secs}
    assert by[TOP_OF_SHAFT].arm == pytest.approx(abs(g.H_free - c))
    assert by[FIXITY].arm == pytest.approx(
        abs(g.H_free + g.fixity_depth(3.0) - c))


# ------------------------------------------------------------------ second hinge
def _diagram(Mp_col, Mp_shaft, H, L, c):
    """Three sections on a fixed-fixed diagram with contraflexure at ``c``."""
    return [Section(DECK, 0.0, c, Mp_col, Mp_col / c),
            Section(TOP_OF_SHAFT, H, abs(H - c), Mp_col, Mp_col / abs(H - c)),
            Section(FIXITY, L, L - c, Mp_shaft, Mp_shaft / (L - c))]


def test_second_hinge_reproduces_the_work_equation_when_the_deck_yields_first():
    """Deck first ⇒ V = 2·Mp_col/H for a column mechanism (classic result)."""
    Mp_col, H, L, c = 1.0e5, 300.0, 700.0, 250.0
    secs = _diagram(Mp_col, 9.0e5, H, L, c)
    first = min(secs, key=lambda s: s.V_yield)
    assert first.name == DECK               # the fixture says what it claims
    second, V = _second_hinge(secs, first, first.V_yield, contraflexure=c)
    assert second.name == TOP_OF_SHAFT
    assert V == pytest.approx(2.0 * Mp_col / H)


def test_the_shaft_is_excluded_even_when_it_would_win():
    """A weak shaft would take the second hinge — but it is capacity-protected.

    Without the exclusion this fixture gives FIXITY at V = (Mpc+Mps)/L = 314,
    below the column mechanism's 333.  The solver must skip it anyway.
    """
    Mp_col, Mp_shaft, H, L, c = 1.0e5, 1.2e5, 600.0, 700.0, 400.0
    secs = _diagram(Mp_col, Mp_shaft, H, L, c)
    first = min(secs, key=lambda s: s.V_yield)
    assert first.name == DECK
    assert (Mp_col + Mp_shaft) / L < 2.0 * Mp_col / H     # the shaft would win
    second, V = _second_hinge(secs, first, first.V_yield, contraflexure=c)
    assert second.name == TOP_OF_SHAFT
    assert V == pytest.approx(2.0 * Mp_col / H)


def test_second_hinge_never_precedes_the_first():
    secs = _diagram(1.0e5, 3.0e5, 300.0, 700.0, 250.0)
    first = min(secs, key=lambda s: s.V_yield)
    _, V = _second_hinge(secs, first, first.V_yield, contraflexure=250.0)
    assert V >= first.V_yield


# ------------------------------------------------------------------- integration
class _MC:
    def __init__(self, Mp, phi_y=1.0e-5, phi_u=6.0e-5):
        self.Mp, self.phi_y, self.phi_u = Mp, phi_y, phi_u


class _Bound:
    def __init__(self, multiplier=3.0, fixity_depth=360.0):
        self.multiplier, self.fixity_depth = multiplier, fixity_depth


class _Assessment:
    def __init__(self, Hcol=300.0, silo=0.0, Mp_col=2.0e5, Mp_shaft=9.0e5):
        self.Hcol_entered, self.silo, self.shaft_D = Hcol, silo, 120.0
        self.EI_col, self.EI_shaft = 2.0e9, 6.0e9
        self.mc_col, self.mc_shaft = _MC(Mp_col), _MC(Mp_shaft)
        self.Lp, self.P_used = 30.0, 500.0
        self.bounds = [_Bound()]
        # multiplier-based fixity: no p-y model, so the fixed-fixed shaft
        # demand is skipped and only the head forces are reported
        self.soil_profile, self.shaft_embed_length = None, 0.0
        self.H_free = Hcol + silo


class _Provisions:
    short_period_magnification = False
    pdelta_factor = 0.25
    overstrength_factor = 1.2


class _Spectrum:
    Ts = None

    def Sa(self, T):                      # flat, so Sd scales with T^2
        return 0.6


def _bent(name, deck_link, k=100.0, k_fixed=400.0, mass=4.0):
    return BentStiffness(name=name, frame="C1", order=0, Hcol=300.0, silo=0.0,
                         k=(k,), mass_long=mass, mass_trans=mass,
                         deck_link=deck_link, bound_labels=("upper",),
                         k_fixed=(k_fixed,))


def _frame(direction, links):
    members = [_bent(f"P{i}", link) for i, link in enumerate(links)]
    members = [b for b in members if b.participates(direction)]
    return Frame(key="C1", direction=direction, members=members, order=0)


def _run(direction, links, **kw):
    frame = _frame(direction, links)
    rows = {b.name: _Assessment(**kw) for b in frame.members}
    return check_frame(frame, direction, 0, rows, _Spectrum(), _Provisions())


def test_frame_stiffness_and_mass_are_the_member_sums():
    fc = _run(LONGITUDINAL, ["integral", "integral", "integral"])
    assert fc.K == pytest.approx(3 * 400.0)          # fixed-fixed longitudinally
    assert fc.M == pytest.approx(3 * 4.0)
    assert fc.T == pytest.approx(2 * math.pi * math.sqrt(12.0 / 1200.0))


def test_bearing_bents_drop_out_longitudinally_but_not_transversely():
    lon = _frame(LONGITUDINAL, ["bearing", "integral", "integral", "bearing"])
    trn = _frame(TRANSVERSE, ["bearing", "integral", "integral", "bearing"])
    assert len(lon.members) == 2
    assert len(trn.members) == 4


def test_transverse_uses_fixed_free_even_for_integral_bents():
    fc = _run(TRANSVERSE, ["integral", "integral"])
    assert fc.end_conditions == "fixed-free"
    assert fc.K == pytest.approx(2 * 100.0)
    assert all(m.end_fixity == "free" for m in fc.members)


def test_fixed_free_member_has_one_hinge_at_the_top_of_shaft():
    fc = _run(TRANSVERSE, ["integral"])
    m = fc.members[0]
    assert len(m.hinges) == 1
    assert m.hinges[0].name == TOP_OF_SHAFT
    assert m.V_mech == pytest.approx(2.0e5 / 300.0)


def test_fixed_fixed_member_needs_two_hinges():
    fc = _run(LONGITUDINAL, ["integral"])
    m = fc.members[0]
    assert m.end_fixity == "fixed"
    assert len(m.hinges) == 2
    assert m.hinges[0].name == DECK


def test_fixed_fixed_carries_more_shear_than_fixed_free():
    """Two hinges take more load than one — the whole point of the fixity."""
    lon = _run(LONGITUDINAL, ["integral"]).members[0]
    trn = _run(TRANSVERSE, ["integral"]).members[0]
    assert lon.V_mech > trn.V_mech


def test_plastic_displacement_follows_the_hinge_spacing():
    m = _run(LONGITUDINAL, ["integral"]).members[0]
    a = _Assessment()
    theta_p = a.Lp * (a.mc_col.phi_u - a.mc_col.phi_y)
    spacing = abs(m.hinges[1].x - m.hinges[0].x)
    assert m.delta_p == pytest.approx(theta_p * (spacing - a.Lp))


def test_cantilever_plastic_displacement_uses_the_half_Lp_lever():
    m = _run(TRANSVERSE, ["integral"]).members[0]
    a = _Assessment()
    theta_p = a.Lp * (a.mc_col.phi_u - a.mc_col.phi_y)
    assert m.delta_p == pytest.approx(theta_p * (300.0 - a.Lp / 2.0))


def test_every_member_of_a_frame_shares_the_displacement_demand():
    fc = _run(TRANSVERSE, ["integral", "integral", "integral"])
    assert len({round(m.delta_d, 9) for m in fc.members}) == 1
    assert fc.members[0].delta_d == pytest.approx(fc.delta_d)


def test_the_shaft_never_hinges_however_weak_it_is():
    """It is capacity-protected: a sizing question, not a hinge candidate.

    A shaft weak enough that it would win the elastic race must still be
    excluded, or Δp picks up the long deck-to-fixity lever it does not have.
    """
    weak = _run(LONGITUDINAL, ["integral"], Mp_col=2.0e5, Mp_shaft=2.1e5)
    strong = _run(LONGITUDINAL, ["integral"], Mp_col=2.0e5, Mp_shaft=9.0e5)
    for fc in (weak, strong):
        m = fc.members[0]
        assert [h.name for h in m.hinges] == [DECK, TOP_OF_SHAFT]
    # and the shaft strength therefore cannot move the displacement capacity
    assert weak.members[0].delta_c == pytest.approx(strong.members[0].delta_c)
    assert weak.passed and strong.passed


def test_deck_first_yield_is_warned_but_not_a_failure_by_itself():
    fc = _run(LONGITUDINAL, ["integral"])
    m = fc.members[0]
    assert [h.name for h in m.hinges] == [DECK, TOP_OF_SHAFT]
    assert m.passed
    assert any("DECK" in w for w in m.warnings)


def test_fixed_fixed_doubles_the_overstrength_shear_on_the_shaft():
    """Same Mo at the interface, 2x the Vo — the number the shaft needs."""
    m = _run(LONGITUDINAL, ["integral"]).members[0]
    assert m.Mo_interface == pytest.approx(1.2 * 2.0e5)
    assert m.Vo_interface == pytest.approx(1.2 * 2.0 * 2.0e5 / 300.0)
    assert m.Vo_cantilever == pytest.approx(1.2 * 2.0e5 / 300.0)
    assert m.shear_amplification == pytest.approx(2.0)
    assert any("capacity-designed" in w for w in m.warnings)


def test_fixed_free_member_gets_no_shear_amplification_warning():
    m = _run(TRANSVERSE, ["integral"]).members[0]
    assert m.shear_amplification == pytest.approx(1.0)
    assert not any("capacity-designed" in w for w in m.warnings)


def test_ratio_and_ductility_are_consistent():
    for direction in (LONGITUDINAL, TRANSVERSE):
        for m in _run(direction, ["integral", "integral"]).members:
            assert m.ratio == pytest.approx(m.delta_c / m.delta_d)
            assert m.mu_d == pytest.approx(m.delta_d / m.delta_y)
            assert m.delta_c == pytest.approx(m.delta_y + m.delta_p)


def test_worst_is_the_minimum_ratio_over_the_members():
    fc = _run(TRANSVERSE, ["integral", "integral", "integral"])
    assert fc.worst == pytest.approx(min(m.ratio for m in fc.members))


# --- geometry chart is keyed on FRAMES, not bents ---------------------------
def test_geometry_checks_pair_frames_and_orphan_nobody():
    """Every frame is an endpoint of a link, and every pier sits in a frame.

    The old chart resolved a frame to its first member, so the other members of
    a continuous frame were never a link endpoint and drew with no line at all.
    """
    from seismic_column.balance import (BalanceCriteria, GEOMETRY_CHECK,
                                        balance_checks, frames_for)
    bents = [_bent(f"P{i}", link) for i, link in enumerate(
        ["pinned", "pinned", "bearing", "integral", "integral", "bearing",
         "pinned"])]
    for i, b in enumerate(bents):
        b.order = i
        b.frame = "C1" if 2 <= i <= 5 else f"F{i}"
    checks = balance_checks(bents, BalanceCriteria())
    for direction in (LONGITUDINAL, TRANSVERSE):
        frames = frames_for(bents, direction)
        keys = {f.key for f in frames}
        geo = [c for c in checks
               if c.name == GEOMETRY_CHECK and c.direction == direction]
        linked = {k for c in geo for k in c.pair}
        assert linked <= keys                    # every pair resolves to a frame
        assert {f.key for f in frames} == linked  # and no frame is left out
        covered = {n for f in frames for n in f.names}
        assert covered == {b.name for b in bents}
