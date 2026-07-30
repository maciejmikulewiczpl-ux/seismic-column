"""Balanced stiffness / balanced frame geometry between adjacent piers, and the
column silo that tunes them."""
import math

import pytest

from seismic_column.balance import (
    GEOMETRY_CHECK,
    STIFFNESS_ANY_CHECK,
    STIFFNESS_CHECK,
    BalanceCriteria,
    BentStiffness,
    adjacent_pairs,
    balance_checks,
    frames_for,
    quantise_silo,
    required_silo,
)
from seismic_column.batch import run_batch_balanced
from seismic_column.geometry import Geometry
from seismic_column.io_schema import (
    DIRECTIONS,
    LONGITUDINAL,
    TRANSVERSE,
    GlobalConfig,
    default_dataframe,
    in_frame,
    project_from_json,
    project_to_json,
    validate,
)
from seismic_column.soil import SoilLayer, SoilProfile

CODES = ["SDC 2.1", "AASHTO SGS 3rd Ed."]

# the balance stages re-run whole rows, so keep the (slow) column-diameter
# search out of it unless a test is specifically about diameter.
# These tests were written when the optimiser ALWAYS re-derived the shaft from
# the column.  It now holds an entered shaft unless told otherwise, so declare
# it variable here to keep exercising what these tests are about -- the balance
# rules and the silo search -- rather than foundation sizing.  (The default
# table pairs a 48 in column with an 84 in shaft, and in multiplier mode a
# bigger shaft means a DEEPER assumed fixity, hence a softer member.)
FAST = ("longitudinal", "confinement", "fc", "shaft_diameter")


def _cfg(**kw):
    base = dict(optimize=True, variable=FAST)
    base.update(kw)
    return GlobalConfig(**base)


# ---------------------------------------------------------------------------
# Geometry: the silo lengthens the free column
# ---------------------------------------------------------------------------
def test_no_silo_is_bit_identical_to_before():
    """silo = 0 must leave every derived length exactly as it was."""
    g = Geometry(Hcol=264.0, D_shaft=84.0)
    assert g.silo == 0.0
    assert g.H_free == g.Hcol
    assert g.effective_length(3.0) == 264.0 + 3.0 * 84.0
    assert g.tip_flexibility(1e9, 3e9, 3.0) == Geometry(
        Hcol=264.0, D_shaft=84.0, silo=0.0).tip_flexibility(1e9, 3e9, 3.0)


def test_silo_adds_to_the_free_length():
    g = Geometry(Hcol=216.0, D_shaft=84.0, silo=48.0)
    assert g.H_free == 264.0
    assert g.fixity_depth(3.0) == 3.0 * 84.0      # measured below top of shaft
    assert g.effective_length(3.0) == 264.0 + 252.0


def test_silo_softens_and_lengthens_the_period():
    EI_c, EI_s = 2.0e9, 6.0e9
    k = [Geometry(Hcol=216.0, D_shaft=84.0, silo=h).lateral_stiffness(
        EI_c, EI_s, 3.0) for h in (0.0, 24.0, 48.0, 120.0)]
    assert k == sorted(k, reverse=True)            # strictly softening
    assert k[-1] < k[0]


# ---------------------------------------------------------------------------
# The checks themselves
# ---------------------------------------------------------------------------
def _bent(name, k, m=1.0, frame="F1", order=0, m_trans=None,
          deck_link="integral", k_fixed=None):
    """A bent with stiffness ``k`` per bound and mass ``m`` (both directions
    unless ``m_trans`` differs)."""
    return BentStiffness(name=name, frame=frame, order=order, Hcol=240.0,
                         silo=0.0, k=tuple(k), mass_long=m,
                         mass_trans=m if m_trans is None else m_trans,
                         deck_link=deck_link, bound_labels=("3D",) * len(k),
                         k_fixed=tuple(k_fixed or ()))


def _long(checks, name=None):
    """Longitudinal checks only — the two directions are identical in most
    fixtures, so this keeps assertions about counts unambiguous."""
    return [c for c in checks if c.direction == LONGITUDINAL
            and (name is None or c.name == name)]


def test_ratio_is_min_over_max_so_order_does_not_matter():
    crit = BalanceCriteria()
    fwd = _long(balance_checks([_bent("A", [10.0], order=0),
                                _bent("B", [8.0], order=1)], crit),
                STIFFNESS_CHECK)
    rev = _long(balance_checks([_bent("B", [8.0], order=0),
                                _bent("A", [10.0], order=1)], crit),
                STIFFNESS_CHECK)
    assert fwd[0].ratio == pytest.approx(0.8)
    assert rev[0].ratio == pytest.approx(0.8)
    assert fwd[0].passed and rev[0].passed


def test_stiffness_check_fails_below_the_limit():
    checks = balance_checks([_bent("A", [10.0], order=0),
                             _bent("B", [5.0], order=1)], BalanceCriteria())
    stiff = _long(checks, STIFFNESS_CHECK)
    assert stiff[0].ratio == pytest.approx(0.5)
    assert not stiff[0].passed
    assert stiff[0].limit == 0.75


def test_period_ratio_is_the_square_root_of_the_kappa_ratio():
    """With kappa = k/m and T = 2*pi*sqrt(m/k), (Ti/Tj)^2 = kappa_j/kappa_i.

    The two rules now live at different levels — stiffness inside a frame,
    geometry between frames — so they never land on the same pair.  Two
    SINGLE-bent frames give the geometry check while the identity still holds
    against the bents' own kappa.
    """
    crit = BalanceCriteria(mass_normalized=True)
    a = _bent("A", [10.0], m=2.0, frame="F1", order=0)
    b = _bent("B", [6.0], m=2.0, frame="F2", order=1)
    checks = _long(balance_checks([a, b], crit))
    assert {c.name for c in checks} == {GEOMETRY_CHECK}   # no within-frame rule
    rt = checks[0].ratio
    ka = a.kappa(LONGITUDINAL, 0, True)
    kb = b.kappa(LONGITUDINAL, 0, True)
    assert rt == pytest.approx(math.sqrt(min(ka, kb) / max(ka, kb)))
    # ... so the 0.75 stiffness rule is stricter than the 0.70 period rule
    assert math.sqrt(0.75) > 0.70


# ---------------------------------------------------------------------------
# Frames: derived per direction from deck_link
# ---------------------------------------------------------------------------
def _en_pattern():
    """The EN bridge shape: simple spans, a continuous frame, simple spans."""
    bents = []
    for i, name in enumerate(["A6", "A7", "A8", "A9", "A10", "A11", "A12"]):
        if name in ("A7", "A11"):
            bents.append(_bent(name, [100.0], frame="C1", order=i,
                               deck_link="bearing"))
        elif name in ("A8", "A9", "A10"):
            bents.append(_bent(name, [100.0], frame="C1", order=i,
                               deck_link="integral"))
        else:
            bents.append(_bent(name, [100.0], frame=f"F{name}", order=i))
    return bents


def test_frames_differ_by_direction():
    """A bearing shear-keyed transversely joins the continuous frame there, and
    LEAVES the model longitudinally -- released, it carries no deck that way,
    only its own cap and column self weight, so it is not a frame at all."""
    bents = _en_pattern()
    lon = [f.names for f in frames_for(bents, LONGITUDINAL)]
    tra = [f.names for f in frames_for(bents, TRANSVERSE)]
    assert lon == [("A6",), ("A8", "A9", "A10"), ("A12",)]
    assert tra == [("A6",), ("A7", "A8", "A9", "A10", "A11"), ("A12",)]
    # ordered along the bridge either way
    assert [f.order for f in frames_for(bents, LONGITUDINAL)] == sorted(
        f.order for f in frames_for(bents, LONGITUDINAL))


def test_free_deck_link_drops_out_entirely():
    bents = [_bent("A", [10.0], frame="C1", order=0),
             _bent("B", [10.0], frame="C1", order=1, deck_link="free")]
    for d in DIRECTIONS:
        assert [f.names for f in frames_for(bents, d)] == [("A",)]


def test_frame_period_sums_stiffness_and_mass():
    bents = [_bent("A", [100.0], m=2.0, frame="C1", order=0),
             _bent("B", [300.0], m=4.0, frame="C1", order=1)]
    f = frames_for(bents, LONGITUDINAL)[0]
    assert f.continuous
    assert f.K(0) == pytest.approx(400.0)
    assert f.M() == pytest.approx(6.0)
    assert f.T(0) == pytest.approx(2 * math.pi * math.sqrt(6.0 / 400.0))


def test_single_bent_frame_period_equals_the_bents_own():
    b = _bent("A", [120.0], m=3.0, frame="F1", order=0)
    f = frames_for([b], LONGITUDINAL)[0]
    assert not f.continuous
    assert f.T(0) == pytest.approx(b.T(LONGITUDINAL, 0))


def test_no_stiffness_rule_between_simply_supported_bents():
    """The correction that started this: a run of simple spans is a run of
    single-bent frames, so §7.1.2 has nothing to compare."""
    bents = [_bent("A", [200.0], frame="F1", order=0),
             _bent("B", [50.0], frame="F2", order=1)]   # ratio 0.25, way off
    checks = balance_checks(bents, BalanceCriteria())
    assert all(c.name == GEOMETRY_CHECK for c in checks)
    assert not any(c.name in (STIFFNESS_CHECK, STIFFNESS_ANY_CHECK)
                   for c in checks)


def test_any_two_rule_catches_what_adjacent_misses():
    """A gentle stiffness taper down one frame: every NEIGHBOUR is inside 0.75,
    yet the two ends are outside 0.50 of each other.

    It takes four bents — with three, 0.75 x 0.75 = 0.5625 is still above the
    0.50 limit, so no three-bent frame can pass adjacent and fail any-two.
    """
    bents = [_bent("A", [100.0], frame="C1", order=0),
             _bent("B", [78.0], frame="C1", order=1),
             _bent("C", [61.0], frame="C1", order=2),
             _bent("D", [48.0], frame="C1", order=3)]
    checks = _long(balance_checks(bents, BalanceCriteria()))
    adj = [c for c in checks if c.name == STIFFNESS_CHECK]
    anyc = [c for c in checks if c.name == STIFFNESS_ANY_CHECK]
    assert all(c.passed for c in adj), [c.label for c in adj if not c.passed]
    # non-adjacent pairs only — an adjacent pair is also "any two", but its
    # stricter 0.75 limit means the 0.50 check could never govern there
    assert {c.pair for c in anyc} == {("A", "C"), ("A", "D"), ("B", "D")}
    failed = [c for c in anyc if not c.passed]
    assert [c.pair for c in failed] == [("A", "D")]
    assert failed[0].limit == 0.50


def test_direction_can_change_the_verdict():
    """Same k, different tributary mass, so the period rule can pass one way and
    fail the other."""
    bents = [_bent("A", [100.0], m=1.0, m_trans=1.0, frame="F1", order=0),
             _bent("B", [100.0], m=1.2, m_trans=4.0, frame="F2", order=1)]
    checks = balance_checks(bents, BalanceCriteria())
    lon = [c for c in checks if c.direction == LONGITUDINAL]
    tra = [c for c in checks if c.direction == TRANSVERSE]
    assert all(c.passed for c in lon)
    assert not all(c.passed for c in tra)


def test_mass_normalisation_changes_the_verdict():
    """Equal stiffness but unequal tributary mass fails the normalised form."""
    bents = [_bent("A", [10.0], m=1.0, order=0), _bent("B", [10.0], m=2.0, order=1)]
    off = _long(balance_checks(bents, BalanceCriteria(mass_normalized=False)),
                STIFFNESS_CHECK)
    on = _long(balance_checks(bents, BalanceCriteria(mass_normalized=True)),
               STIFFNESS_CHECK)
    assert off[0].passed
    assert not on[0].passed


def test_only_adjacent_pairs_within_a_frame_are_compared():
    bents = [_bent("A", [10.0], frame="F1", order=0),
             _bent("B", [9.0], frame="F1", order=1),
             _bent("C", [8.0], frame="F1", order=2),
             _bent("D", [1.0], frame="F2", order=3)]
    pairs = {(x.name, y.name) for x, y in adjacent_pairs(bents)}
    assert pairs == {("A", "B"), ("B", "C")}       # no A-C, no C-D across frames


def test_single_bent_frame_produces_no_checks():
    assert balance_checks([_bent("A", [10.0])], BalanceCriteria()) == []


def test_frame_exclusion_markers():
    assert in_frame("F1") and in_frame("Bridge 2")
    assert not in_frame("") and not in_frame("-") and not in_frame("  ")


def test_unusable_stiffness_fails_rather_than_crashing():
    bents = [_bent("A", [float("nan")], order=0), _bent("B", [8.0], order=1)]
    checks = _long(balance_checks(bents, BalanceCriteria()))
    assert checks and not checks[0].passed
    assert math.isnan(checks[0].ratio)


# ---------------------------------------------------------------------------
# Sizing a silo
# ---------------------------------------------------------------------------
def test_required_silo_hits_the_target_stiffness():
    g = Geometry(Hcol=216.0, D_shaft=84.0)
    EI_c, EI_s = 2.0e9, 6.0e9
    k0 = g.lateral_stiffness(EI_c, EI_s, 3.0)
    h = required_silo(g, EI_c, EI_s, 3.0, k_target=0.7 * k0, silo_max=600.0)
    assert h is not None and h > 0
    k = Geometry(Hcol=216.0, D_shaft=84.0, silo=h).lateral_stiffness(EI_c, EI_s, 3.0)
    assert k == pytest.approx(0.7 * k0, rel=1e-3)


def test_required_silo_returns_none_when_the_cap_binds():
    g = Geometry(Hcol=216.0, D_shaft=84.0)
    k0 = g.lateral_stiffness(2.0e9, 6.0e9, 3.0)
    assert required_silo(g, 2.0e9, 6.0e9, 3.0, k_target=0.01 * k0,
                         silo_max=12.0) is None


def test_required_silo_is_zero_when_already_soft_enough():
    g = Geometry(Hcol=216.0, D_shaft=84.0)
    k0 = g.lateral_stiffness(2.0e9, 6.0e9, 3.0)
    assert required_silo(g, 2.0e9, 6.0e9, 3.0, k_target=2.0 * k0) == 0.0


def test_quantise_silo_rounds_up_and_clamps():
    assert quantise_silo(13.0, 12.0, 240.0) == 24.0     # up to the next foot
    assert quantise_silo(24.0, 12.0, 240.0) == 24.0     # exact stays put
    assert quantise_silo(500.0, 12.0, 240.0) == 240.0   # clamped to the cap
    assert quantise_silo(1.0, 12.0, 240.0, floor=36.0) == 36.0


def test_quantise_silo_floor_beats_the_cap():
    """An entered silo deeper than the auto-silo cap must never be shrunk."""
    assert quantise_silo(1.0, 12.0, cap=24.0, floor=360.0) == 360.0
    assert quantise_silo(600.0, 12.0, cap=24.0, floor=360.0) == 360.0


# ---------------------------------------------------------------------------
# Soil profile below a silo
# ---------------------------------------------------------------------------
def _profile():
    return SoilProfile(layers=(
        SoilLayer.from_engineering(20.0, "matlock_soft_clay", 120.0,
                                   su_top_ksf=1.0, su_bot_ksf=1.5),
        SoilLayer.from_engineering(30.0, "api_sand", 125.0, phi_deg=36.0,
                                   k_pci=90.0),
    ))


def test_below_zero_is_the_same_profile():
    p = _profile()
    assert p.below(0.0) is p
    assert p.below(-5.0) is p


def test_below_shifts_the_reference_level_only():
    """A silo moves the pile head down; the ground itself is unchanged, so every
    p-y term must be read at the TRUE depth in intact soil."""
    p = _profile()
    h = 5.0 * 12.0
    q = p.below(h)
    assert q.layers == p.layers                # profile untouched
    assert q.surface_offset == pytest.approx(h)
    assert q.depth == pytest.approx(p.depth - h)
    assert q.total_depth == pytest.approx(p.total_depth)
    for z in (0.0, 12.0, 120.0, 400.0):
        assert q.sigma_v_eff(z) == pytest.approx(p.sigma_v_eff(z + h))
        assert q.p_ult(z, 84.0) == pytest.approx(p.p_ult(z + h, 84.0))
        assert q.p_of_y(z, 0.5, 84.0) == pytest.approx(p.p_of_y(z + h, 0.5, 84.0))
        assert (q.secant_modulus(z, 0.5, 84.0)
                == pytest.approx(p.secant_modulus(z + h, 0.5, 84.0)))
        assert q.layer_at(z)[0] is p.layer_at(z + h)[0]


def test_silo_does_not_discard_the_near_surface_wedge():
    """Regression: the first implementation stripped the strata and restarted z
    at the silo bottom, which threw away the J*z/D wedge term and cost ~1/3 of
    the resistance at the shaft head.  A casing narrower than the shaft leaves
    that soil intact, so resistance must be at least the intact-ground value."""
    p = _profile()
    h = 10.0 * 12.0
    q = p.below(h)
    # at the shaft head, the silo'd pile sees the full 10 ft of confinement
    assert q.p_ult(0.0, 118.0) == pytest.approx(p.p_ult(h, 118.0))
    assert q.p_ult(0.0, 118.0) > p.p_ult(0.0, 118.0)


def test_below_past_the_whole_profile_still_resolves():
    p = _profile()
    q = p.below(500.0 * 12.0)
    assert q.depth == 0.0                      # nothing left below the head
    assert q.layer_at(0.0)[0] is p.layers[-1]  # extends the bottom layer
    assert q.signature() != p.signature()


# ---------------------------------------------------------------------------
# Integration through the batch
# ---------------------------------------------------------------------------
@pytest.mark.slow
def _uneven_df(n=3):
    """A table that genuinely fails the FRAME-PERIOD rule.

    The default table is simply supported (one frame per bent), so the only
    rule in play is the adjacent-frame period ratio — and 18/22/26 ft passes it
    comfortably.  These heights do not.
    """
    df = default_dataframe(n)
    # far enough apart to fail the 0.70 period ratio, but still designable at
    # the fixed 48 in diameter these fast tests use
    df["Hcol_ft"] = [14.0, 22.0, 28.0, 20.0, 26.0][:n]
    return df


def test_default_table_is_unbalanced_before_any_silo():
    df = _uneven_df(3)                  # 18 / 22 / 26 ft
    out = run_batch_balanced(df, _cfg(optimize=False, balance_auto_silo=False))
    assert out.balance is not None
    assert not out.balance.passed
    assert all(r.silo == 0.0 for r in out.results)      # nothing was changed
    # every frame holds one bent, so no stiffness rule applies at all — the
    # frame-period rule is the only thing that can fail here
    assert not any(c.name in (STIFFNESS_CHECK, STIFFNESS_ANY_CHECK)
                   for c in out.balance.checks)
    assert any(c.name == GEOMETRY_CHECK and not c.passed
               for c in out.balance.checks)

@pytest.mark.slow

@pytest.mark.parametrize("code", CODES)
def test_auto_silo_balances_and_every_column_still_passes_seismic(code):
    df = _uneven_df(3)
    out = run_batch_balanced(df, _cfg(code=code, balance_auto_silo=True))
    assert out.balance.passed, [c.label for c in out.balance.failed]
    assert out.balance.converged
    # a silo was actually used, and the seismic suite still holds everywhere
    assert any(r.silo > 0 for r in out.results)
    for rr in out.results:
        assert rr.feasible, (rr.name,
                             [c.name for c in rr.assessment.checks if not c.passed])
    # the summary reports the design that was finally checked, at the length it
    # was checked at
    for rr in out.results:
        row = out.summary[out.summary["name"] == rr.name].iloc[0]
        assert row["silo_ft"] == pytest.approx(rr.silo / 12.0, abs=0.01)
        assert row["H_free_ft"] == pytest.approx(rr.assessment.H_free / 12.0,
                                                 abs=0.01)
        assert row["balanced"] == "PASS"
        assert row["bal_T_long"] is not None and row["bal_T_trans"] is not None


@pytest.mark.slow
def test_a_silo_forces_a_real_seismic_re_analysis():
    """The pier that gets a silo must be re-analysed, not just re-labelled:
    Lp, the demands and (in an optimise run) the reinforcement all move."""
    df = _uneven_df(3)
    stage1 = run_batch_balanced(df, _cfg(balance_auto_silo=False))
    final = run_batch_balanced(df, _cfg(balance_auto_silo=True))

    siloed = [(a, b) for a, b in zip(stage1.results, final.results) if b.silo > 0]
    assert siloed, "expected the default table to need a silo"
    for before, after in siloed:
        # Lp = 0.08*H_free + ... so it MUST grow with the silo
        assert after.assessment.Lp > before.assessment.Lp
        assert after.assessment.H_free == pytest.approx(
            before.assessment.H_free + after.silo)
        # the self-weight is taken over the longer column too
        assert after.assessment.W_self > before.assessment.W_self
        # and the checks were genuinely re-evaluated at the new length
        assert after.feasible

    # a pier that got no silo is left completely alone
    for before, after in zip(stage1.results, final.results):
        if after.silo == 0.0:
            assert after.design.long_label() == before.design.long_label()
            assert after.assessment.Lp == pytest.approx(before.assessment.Lp)


@pytest.mark.slow
def test_silo_actually_lengthens_the_analysed_column():
    df = default_dataframe(3)
    out = run_batch_balanced(df, _cfg(balance_auto_silo=True))
    for rr in out.results:
        a = rr.assessment
        assert a.silo == pytest.approx(rr.silo)
        assert a.H_free == pytest.approx(a.Hcol_entered + rr.silo)
        # Lp, Vo and Vp are all driven by the FREE length, not the entered one
        assert a.governing_bound.Le == pytest.approx(
            a.H_free + a.governing_bound.fixity_depth)

@pytest.mark.slow

@pytest.mark.parametrize("strategy", ["greedy", "min_silo"])
def test_silo_cap_reports_instead_of_looping(strategy):
    """Whichever strategy is driving, a binding cap must be reported with the
    cap value and what to do about it — never an infinite loop or a bare FAIL.

    These are simply supported bents, so the only rule in play is BETWEEN-frame
    geometry.  Pursued to the cap and still short, that is exactly the case the
    codes send to time-history rather than to a deeper silo, so it is referred
    there — reported prominently, but not counted as a hard failure.
    """
    df = default_dataframe(3)
    df.loc[0, "Hcol_ft"] = 12.0
    df.loc[2, "Hcol_ft"] = 40.0
    out = run_batch_balanced(df, _cfg(balance_auto_silo=True, max_silo_ft=1.0,
                                      balance_strategy=strategy))
    bal = out.balance
    assert not bal.converged                  # the search genuinely gave up
    assert bal.needs_tha                      # ... and said so
    assert all(c.name == GEOMETRY_CHECK for c in bal.needs_tha)
    assert all(c.status == "THA" for c in bal.needs_tha)
    assert not bal.failed                     # no HARD failure remains
    log = "\n".join(bal.log)
    assert "cap" in log                       # names the binding constraint
    assert "raise the cap" in log             # and what to do about it
    assert "TIME-HISTORY REQUIRED" in log
    assert all(r.silo <= 1.0 * 12.0 + 1e-9 for r in out.results)


@pytest.mark.slow
def test_entered_silo_is_a_floor_never_reduced():
    df = default_dataframe(3)
    df.loc[2, "silo_ft"] = 3.0                 # the most flexible pier already
    out = run_batch_balanced(df, _cfg(balance_auto_silo=True))
    c3 = next(r for r in out.results if r.name == "C3")
    assert c3.silo >= 3.0 * 12.0 - 1e-9


@pytest.mark.slow
def test_entered_silo_deeper_than_the_cap_is_honoured():
    """The cap limits what the TOOL adds; a typed-in silo is the engineer's."""
    df = default_dataframe(3)
    df.loc[0, "silo_ft"] = 8.0
    out = run_batch_balanced(df, _cfg(balance_auto_silo=True, max_silo_ft=2.0))
    c1 = next(r for r in out.results if r.name == "C1")
    assert c1.silo == pytest.approx(8.0 * 12.0)
    assert c1.assessment.H_free == pytest.approx(c1.assessment.Hcol_entered
                                                 + 8.0 * 12.0)


@pytest.mark.slow
def test_balance_progress_is_separate_from_the_row_progress_count():
    """``progress`` keeps its once-per-row contract; the balance stage reports
    through ``balance_progress`` instead of corrupting the done-count."""
    df = _uneven_df(3)
    calls, msgs = [], []
    run_batch_balanced(
        df, _cfg(balance_auto_silo=True),
        progress=lambda d, t, n, s: calls.append((d, t)),
        balance_progress=msgs.append)
    assert [d for d, _ in calls] == [1, 2, 3]      # exactly once per row
    assert all(t == 3 for _, t in calls)
    assert msgs and all("Balancing pass" in m for m in msgs)


# ---------------------------------------------------------------------------
# Minimum-silo search (exact DP over the buildable grid)
# ---------------------------------------------------------------------------
def test_silo_states_grid():
    from seismic_column.balance import silo_states

    assert silo_states(0.0, 60.0, 12.0) == (0.0, 12.0, 24.0, 36.0, 48.0, 60.0)
    # an entered floor off the grid is kept as-is, then the grid resumes above it
    s = silo_states(5.0, 40.0, 12.0)
    assert s[0] == 5.0 and s[1] == 12.0 and s[-1] <= 40.0
    assert silo_states(0.0, 60.0, 0.0) == (0.0,)          # degenerate step


def _dp(bents, k_table, criteria, states=None):
    """Run the DP against an explicit {name: {silo: k}} table.

    The predicate is supplied by the caller now, so this mirrors what
    ``_plan_silos_min`` does: neighbours must satisfy the pair rules.
    """
    from seismic_column.balance import dp_min_silo, pair_ok

    states = states or {n: tuple(sorted(v)) for n, v in k_table.items()}

    def feasible(bi, si, bj, sj):
        return pair_ok(k_table[bi.name][si], k_table[bj.name][sj],
                       bi.mass(LONGITUDINAL), bj.mass(LONGITUDINAL), criteria)

    return dp_min_silo(bents, states, feasible)


def test_dp_matches_brute_force():
    """The DP must return the true grid optimum, checked by enumeration."""
    import itertools

    from seismic_column.balance import pair_ok

    crit = BalanceCriteria(mass_normalized=True)
    names = ["P1", "P2", "P3", "P4"]
    masses = [1.00, 1.05, 1.00, 1.15]
    grid = (0.0, 12.0, 24.0, 36.0, 48.0)
    # a silo softens: k falls with depth
    k_table = {n: {s: k0 * (1.0 - 0.11 * s / 12.0) for s in grid}
               for n, k0 in zip(names, [230.0, 200.0, 220.0, 175.0])}
    bents = [_bent(n, [k_table[n][0.0]], m=m, order=i)
             for i, (n, m) in enumerate(zip(names, masses))]

    def brute_force():
        best = None
        for combo in itertools.product(grid, repeat=len(names)):
            if all(pair_ok(k_table[names[i]][combo[i]],
                           k_table[names[i + 1]][combo[i + 1]],
                           masses[i], masses[i + 1], crit)
                   for i in range(len(names) - 1)):
                if best is None or sum(combo) < best:
                    best = sum(combo)
        return best

    plan = _dp(bents, k_table, crit)
    best = brute_force()
    assert best is not None, "fixture is infeasible — pick easier numbers"
    assert plan.feasible
    assert plan.total == pytest.approx(best)
    # and the DP must agree when there is NO answer, not invent one
    hard = {n: {s: k for s, k in v.items()} for n, v in k_table.items()}
    hard["P4"] = {s: 40.0 for s in grid}          # far too soft for any silo
    bents_hard = [_bent(n, [hard[n][0.0]], m=m, order=i)
                  for i, (n, m) in enumerate(zip(names, masses))]
    assert not _dp(bents_hard, hard, crit).feasible


def test_dp_respects_the_floor_and_the_cap():
    from seismic_column.balance import dp_min_silo, pair_ok

    crit = BalanceCriteria(mass_normalized=True)
    grid_a, grid_b = (24.0, 36.0), (0.0, 12.0)       # P1 floored at 24 in
    k = {"P1": {24.0: 200.0, 36.0: 180.0}, "P2": {0.0: 190.0, 12.0: 170.0}}
    bents = [_bent("P1", [200.0], order=0), _bent("P2", [190.0], order=1)]
    plan = dp_min_silo(
        bents, {"P1": grid_a, "P2": grid_b},
        lambda bi, si, bj, sj: pair_ok(k[bi.name][si], k[bj.name][sj],
                                       bi.mass(LONGITUDINAL),
                                       bj.mass(LONGITUDINAL), crit))
    assert plan.feasible
    assert plan.silos["P1"] >= 24.0                   # never below the floor
    assert all(v in k[n] for n, v in plan.silos.items())   # only grid points


def test_dp_reports_an_impossible_pair():
    crit = BalanceCriteria(mass_normalized=True)
    # P2 is far softer than P1 and no silo brings P1 down far enough
    k_table = {"P1": {0.0: 500.0, 12.0: 480.0}, "P2": {0.0: 50.0, 12.0: 48.0}}
    bents = [_bent("P1", [500.0], order=0), _bent("P2", [50.0], order=1)]
    plan = _dp(bents, k_table, crit)
    assert not plan.feasible
    assert any("INFEASIBLE" in n and "P1-P2" in n for n in plan.notes)


def test_dp_treats_frames_independently():
    crit = BalanceCriteria(mass_normalized=True)
    k_table = {n: {0.0: k, 12.0: k * 0.85} for n, k in
               (("A", 200.0), ("B", 190.0), ("C", 90.0))}
    bents = [_bent("A", [200.0], frame="F1", order=0),
             _bent("B", [190.0], frame="F1", order=1),
             _bent("C", [90.0], frame="F2", order=2)]   # own frame: unconstrained
    plan = _dp(bents, k_table, crit)
    assert plan.feasible
    assert plan.silos["C"] == 0.0                       # nothing to balance against


@pytest.mark.slow
def test_min_silo_is_never_worse_than_greedy():
    """Both strategies converge to the same minimum — greedy is Gauss-Seidel on
    the same monotone fixed point, not a heuristic.  This pins that they agree,
    so a future change to either shows up here."""
    df = default_dataframe(3)
    a = run_batch_balanced(df, _cfg(balance_strategy="greedy"))
    b = run_batch_balanced(df, _cfg(balance_strategy="min_silo"))
    assert a.balance.passed and b.balance.passed
    tot_a = sum(r.silo for r in a.results)
    tot_b = sum(r.silo for r in b.results)
    assert tot_b <= tot_a + 1e-9
    assert all(r.feasible for r in b.results)


@pytest.mark.slow
def test_min_silo_result_is_on_the_grid_and_verified():
    df = default_dataframe(3)
    out = run_batch_balanced(df, _cfg(balance_strategy="min_silo",
                                      silo_step_ft=2.0))
    assert out.balance.passed
    for rr in out.results:
        ft = rr.silo / 12.0
        assert abs(ft / 2.0 - round(ft / 2.0)) < 1e-6, f"{rr.name} off-grid: {ft}"


@pytest.mark.slow
def test_identical_bounds_are_collapsed():
    """Equal stiff/soft brackets make evaluate_column run the same analysis
    twice; the balance layer must not double every check because of it."""
    df = default_dataframe(2)
    df["mult_lb"] = 4.0
    df["mult_ub"] = 4.0                       # both bounds identical
    out = run_batch_balanced(df, _cfg(optimize=False, balance_auto_silo=False))
    assert len(out.results[0].assessment.bounds) == 2      # analysis still runs 2
    assert len(out.balance.bents[0].k) == 1                # but only 1 is reported
    assert len(out.balance.bents[0].bound_labels) == 1
    # one pair x one bound x two check types
    assert len(out.balance.checks) == 2

    df2 = default_dataframe(2)                # distinct bounds are all kept
    out2 = run_batch_balanced(df2, _cfg(optimize=False, balance_auto_silo=False))
    assert len(out2.balance.bents[0].k) == 2
    assert len(out2.balance.checks) == 4


@pytest.mark.slow
def test_period_rule_alone_drives_a_silo():
    """Regression: with mass normalisation OFF the stiffness and period rules
    decouple, so the period rule can be the ONLY thing failing.  The silo
    planner used to watch stiffness alone and would report "no further silo
    change is available" while a fix was in reach (found on the A7-A8 pair of
    the EN project)."""
    df = default_dataframe(2)
    # identical sections and heights -> identical k, so the stiffness ratio is
    # 1.000 and passes; only the tributary mass differs, and T = 2*pi*sqrt(m/k)
    # carries mass even when normalisation is off.
    df["Hcol_ft"] = 22.0
    df["weight_long_kip"] = [900.0, 2050.0]
    df["weight_trans_kip"] = [900.0, 2050.0]
    cfg = _cfg(optimize=False, balance_mass_normalized=False,
               balance_auto_silo=False)
    before = run_batch_balanced(df, cfg).balance
    # each bent is its own frame, so only the period rule is in play — and it
    # carries mass through T = 2*pi*sqrt(m/k) even with normalisation off
    t_bad = [c for c in before.checks if c.name == GEOMETRY_CHECK]
    assert t_bad and not all(c.passed for c in t_bad),         "period should fail on unequal mass"

    after = run_batch_balanced(df, _cfg(optimize=False,
                                        balance_mass_normalized=False,
                                        balance_auto_silo=True))
    assert after.balance.passed, [c.label for c in after.balance.failed]
    # the LIGHTER pier has the shorter period, so it is the one softened
    silos = {r.name: r.silo for r in after.results}
    assert silos["C1"] > 0 and silos["C2"] == 0


def test_balance_check_off_runs_nothing():
    df = default_dataframe(3)
    out = run_batch_balanced(df, _cfg(optimize=False, balance_check=False))
    assert out.balance is None
    assert all(r.silo == 0.0 for r in out.results)
    assert "balanced" not in out.summary.columns


@pytest.mark.slow
def test_excluded_piers_are_left_out_of_the_pairs():
    df = default_dataframe(3)
    df.loc[1, "frame"] = ""                    # C2 opts out
    out = run_batch_balanced(df, _cfg(optimize=False, balance_auto_silo=False))
    # frames are single-bent here, so a frame key is the pier's own frame id
    pairs = {c.pair for c in out.balance.checks}
    assert pairs == {("F1", "F3")}             # C2 removed, F1-F3 now adjacent
    assert out.summary.loc[out.summary["name"] == "C2", "balanced"].iloc[0] == "-"


@pytest.mark.slow
def test_stiffness_stays_inside_a_frame_but_geometry_crosses_it():
    """Two continuous frames: the stiffness rule never reaches across the
    boundary, but the frame-period rule is explicitly a BETWEEN-frames rule."""
    df = default_dataframe(4)
    df["frame"] = ["A", "A", "B", "B"]
    # CONTINUOUS frames: two bents on bearings would be a simply supported
    # span, which is matched on period alone unless the switch says otherwise
    df["deck_link"] = "integral"
    out = run_batch_balanced(df, _cfg(optimize=False, balance_auto_silo=False))
    stiff = {c.pair for c in out.balance.checks
             if c.name in (STIFFNESS_CHECK, STIFFNESS_ANY_CHECK)}
    geom = {c.pair for c in out.balance.checks if c.name == GEOMETRY_CHECK}
    assert stiff == {("C1", "C2"), ("C3", "C4")}      # within frames only
    assert geom == {("A", "B")}                       # across the boundary


@pytest.mark.slow
def test_user_limit_may_only_be_stricter_than_the_code():
    df = default_dataframe(3)
    out = run_batch_balanced(df, _cfg(optimize=False, balance_auto_silo=False,
                                      balance_k_ratio_min=0.5))
    assert out.balance.criteria.k_ratio_min == 0.75          # code floor wins
    out2 = run_batch_balanced(df, _cfg(optimize=False, balance_auto_silo=False,
                                       balance_k_ratio_min=0.9))
    assert out2.balance.criteria.k_ratio_min == 0.9          # stricter honoured


# ---------------------------------------------------------------------------
# Schema / persistence
# ---------------------------------------------------------------------------
def test_legacy_table_migrates_to_simply_supported_frames():
    """A pre-deck_link table used one frame id for every row, which under the
    new rules would read as one giant continuous frame and start applying the
    any-two rule.  Simply supported spans are the safe reading, and the
    regrouping must be recorded rather than done silently."""
    df = default_dataframe(3).drop(columns=["deck_link", "weight_trans_kip"])
    df = df.rename(columns={"weight_long_kip": "weight_kip"})
    df["frame"] = "F1"                         # the old default: all one frame
    out = validate(df)
    assert list(out["frame"]) == ["F1", "F2", "F3"]        # one frame per bent
    # `pinned` (bearings, no moment) keeps the fixed-free stiffness a legacy
    # table had — defaulting to `integral` would silently make every bent
    # fixed-fixed longitudinally and 4x stiffer
    assert list(out["deck_link"]) == ["pinned"] * 3
    assert (out["weight_trans_kip"] == out["weight_long_kip"]).all()
    assert any("own frame" in m for m in out.attrs.get("migrations", []))


def test_deck_link_is_validated():
    df = default_dataframe(1)
    df.loc[0, "deck_link"] = "welded"
    with pytest.raises(ValueError, match="deck_link"):
        validate(df)


@pytest.mark.slow
def test_continuous_frame_end_to_end():
    """A continuous frame declared in the table produces within-frame stiffness
    checks, one frame period, and direction-dependent membership."""
    df = default_dataframe(5)
    df["frame"] = ["FA", "C1", "C1", "C1", "FB"]
    df["deck_link"] = ["integral", "bearing", "integral", "bearing", "integral"]
    out = run_batch_balanced(df, _cfg(optimize=False, balance_auto_silo=False))
    b = out.balance
    lon = [f.names for f in b.frames[LONGITUDINAL]]
    tra = [f.names for f in b.frames[TRANSVERSE]]
    # C2/C4 are released longitudinally: they carry no deck that way, so they
    # leave the longitudinal model entirely and C3 is alone in the frame there
    assert ("C3",) in lon
    assert not any({"C2", "C4"} & set(names) for names in lon)
    assert ("C2", "C3", "C4") in tra
    # the stiffness rule fires only transversely, where the frame is continuous
    assert not any(c.name in (STIFFNESS_CHECK, STIFFNESS_ANY_CHECK)
                   for c in b.checks if c.direction == LONGITUDINAL)
    assert any(c.name == STIFFNESS_CHECK
               for c in b.checks if c.direction == TRANSVERSE)
    # the exact search cannot handle a continuous frame and says so
    out2 = run_batch_balanced(df, _cfg(optimize=False, balance_auto_silo=True,
                                       balance_strategy="min_silo"))
    assert any("fell back to pairwise repair" in m for m in out2.balance.log)


def test_validate_backfills_frame_and_silo_for_older_tables():
    df = default_dataframe(2).drop(columns=["frame", "silo_ft"])
    out = validate(df)
    # nothing was said about how the decks sit, so the safe reading is simply
    # supported spans -- one frame per bent, not one continuous deck
    assert list(out["frame"]) == ["F1", "F2"]
    assert list(out["silo_ft"]) == [0.0, 0.0]


def test_validate_rejects_a_bad_silo():
    df = default_dataframe(1)
    df.loc[0, "silo_ft"] = -1.0
    with pytest.raises(ValueError, match="negative"):
        validate(df)
    df.loc[0, "silo_ft"] = 50.0
    with pytest.raises(ValueError, match="maximum"):
        validate(df, max_silo_ft=20.0)


def test_validate_preserves_a_blank_frame_cell():
    df = default_dataframe(2)
    df.loc[0, "frame"] = ""
    assert validate(df).loc[0, "frame"] == ""


def test_project_round_trip_keeps_frame_silo_and_balance_settings():
    df = default_dataframe(2)
    df.loc[0, "silo_ft"] = 4.0
    df.loc[1, "frame"] = "B2"
    cfg = _cfg(balance_check=True, balance_auto_silo=False,
               balance_mass_normalized=False, max_silo_ft=15.0,
               silo_step_ft=0.5)
    df2, cfg2 = project_from_json(project_to_json(df, cfg))
    assert list(df2["silo_ft"]) == [4.0, 0.0]
    assert list(df2["frame"]) == ["F1", "B2"]
    assert cfg2.balance_check and not cfg2.balance_auto_silo
    assert not cfg2.balance_mass_normalized
    assert cfg2.max_silo_ft == 15.0 and cfg2.silo_step_ft == 0.5


@pytest.mark.slow
def test_results_write_the_silo_back_into_the_table():
    from seismic_column.batch import results_to_dataframe
    df = default_dataframe(3)
    out = run_batch_balanced(df, _cfg(balance_auto_silo=True))
    back = results_to_dataframe(out.results, df)
    for rr in out.results:
        assert back.loc[back["name"] == rr.name, "silo_ft"].iloc[0] == \
            pytest.approx(rr.silo / 12.0)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_balance_report_is_complete_and_clean():
    from seismic_column.report import balance_report, column_report
    df = _uneven_df(3)
    out = run_batch_balanced(df, _cfg(balance_auto_silo=True))
    txt = balance_report(out.balance)
    for heading in ("# Balanced stiffness & balanced frame geometry",
                    "## Frames", "## Bents", "## Checks", "## Balancing log"):
        assert heading in txt
    assert "nan" not in txt and "None" not in txt
    # the per-column report must show the silo it was analysed with
    silo_rr = next(r for r in out.results if r.silo > 0)
    col_txt = column_report(silo_rr)
    assert "Column silo" in col_txt and "H_free" in col_txt
    assert "nan" not in col_txt


# ---------------------------------------------------------------------------
# End condition: integral bents are fixed-fixed LONGITUDINALLY
# ---------------------------------------------------------------------------
def test_fixed_fixed_reduces_to_the_code_closed_forms():
    """A prismatic member must give exactly SDC C7.1.2-1 and C7.1.2-2."""
    EI = 2.5e9
    g = Geometry(Hcol=200.0, D_shaft=80.0)
    L = g.H_free + g.fixity_depth(3.0)
    free = g.lateral_stiffness(EI, EI, 3.0, "free")
    fixed = g.lateral_stiffness(EI, EI, 3.0, "fixed")
    assert free == pytest.approx(3.0 * EI / L ** 3)     # C7.1.2-1
    assert fixed == pytest.approx(12.0 * EI / L ** 3)   # C7.1.2-2
    assert fixed / free == pytest.approx(4.0)


def test_stepped_member_sits_between_the_two_limits():
    """The column-on-shaft member is not prismatic, so the fixed-fixed gain is
    less than the prismatic factor of 4 — the step sits in a different place in
    the two moment diagrams."""
    g = Geometry(Hcol=200.0, D_shaft=80.0)
    free = g.lateral_stiffness(2.0e9, 8.0e9, 3.0, "free")
    fixed = g.lateral_stiffness(2.0e9, 8.0e9, 3.0, "fixed")
    assert 1.0 < fixed / free < 4.0


def test_fixed_fixed_still_softens_with_a_silo():
    """required_silo bisects on the stiffness curve, so it must stay monotone
    under the fixed-fixed end condition too."""
    g = Geometry(Hcol=216.0, D_shaft=84.0)
    ks = [Geometry(Hcol=216.0, D_shaft=84.0, silo=h).lateral_stiffness(
        2.0e9, 6.0e9, 3.0, "fixed") for h in (0.0, 24.0, 48.0, 120.0)]
    assert ks == sorted(ks, reverse=True)


def test_only_integral_bents_are_fixed_longitudinally():
    for link, lon in (("integral", "fixed"), ("pinned", "free"),
                      ("bearing", "free"), ("free", "free")):
        b = _bent("A", [100.0], deck_link=link)
        assert b.end_fixity(LONGITUDINAL) == lon, link
        assert b.end_fixity(TRANSVERSE) == "free", link   # never fixed trans.


def test_integral_bent_is_stiffer_longitudinally_than_transversely():
    b = _bent("A", [100.0], deck_link="integral", k_fixed=[285.0])
    assert b.stiffness(LONGITUDINAL, 0) == 285.0
    assert b.stiffness(TRANSVERSE, 0) == 100.0
    # ... and the period follows, for the same mass
    assert b.T(LONGITUDINAL, 0) < b.T(TRANSVERSE, 0)


def test_frame_K_uses_each_members_end_condition():
    """An integral member brings its fixed-fixed stiffness longitudinally; a
    bearing member in the same frame brings its fixed-free one."""
    bents = [_bent("A", [100.0], frame="C1", order=0,
                   deck_link="integral", k_fixed=[300.0]),
             _bent("B", [100.0], frame="C1", order=1,
                   deck_link="pinned")]
    lon = frames_for(bents, LONGITUDINAL)[0]
    tra = frames_for(bents, TRANSVERSE)[0]
    assert lon.K(0) == pytest.approx(400.0)     # 300 fixed-fixed + 100 fixed-free
    assert tra.K(0) == pytest.approx(200.0)     # both fixed-free
    assert "fixed-fixed" in lon.end_conditions
    assert lon.end_conditions != tra.end_conditions


# --- referring an impractical geometry pair to time-history ----------------
def _chk(name, ratio, passed, tha=False):
    from seismic_column.balance import BalanceCheck
    return BalanceCheck(name=name, pair=("A", "B"), bound="best", ratio=ratio,
                        limit=0.75, passed=passed, tha_required=tha)


def test_a_tha_referral_is_not_a_hard_failure_but_is_still_reported():
    from seismic_column.balance import (BalanceResult, GEOMETRY_CHECK,
                                        STIFFNESS_CHECK)
    r = BalanceResult(checks=[
        _chk(STIFFNESS_CHECK, 0.90, True),
        _chk(GEOMETRY_CHECK, 0.48, False, tha=True),
    ])
    assert r.passed                      # no HARD failure left
    assert r.failed == []
    assert len(r.needs_tha) == 1
    assert r.needs_tha[0].status == "THA"


def test_a_within_frame_stiffness_shortfall_stays_a_hard_failure():
    """Time-history does not excuse how a frame distributes its own demand."""
    from seismic_column.balance import (BalanceResult, GEOMETRY_CHECK,
                                        STIFFNESS_CHECK)
    r = BalanceResult(checks=[
        _chk(STIFFNESS_CHECK, 0.60, False),
        _chk(GEOMETRY_CHECK, 0.48, False, tha=True),
    ])
    assert not r.passed
    assert [c.name for c in r.failed] == [STIFFNESS_CHECK]


def test_status_distinguishes_the_three_outcomes():
    from seismic_column.balance import GEOMETRY_CHECK, STIFFNESS_CHECK
    assert _chk(STIFFNESS_CHECK, 0.9, True).status == "OK"
    assert _chk(STIFFNESS_CHECK, 0.6, False).status == "NG"
    assert _chk(GEOMETRY_CHECK, 0.6, False, tha=True).status == "THA"


# --- silo sized for a pier's OWN shear, not just for balance ---------------
def test_shear_silo_is_zero_when_the_column_shear_check_passes():
    from seismic_column.batch import shear_silo_required
    class _Ck:
        name, demand, capacity = "Column shear", 400.0, 900.0
    class _A:
        H_free = 240.0
        checks = [_Ck()]
    class _R:
        assessment = _A()
    assert shear_silo_required(_R()) == 0.0


def test_shear_silo_scales_with_the_demand_capacity_ratio():
    """Vo ~ 1/H_free, so H_required = H_free * (Vo/phiVn)."""
    from seismic_column.batch import shear_silo_required
    class _Ck:
        name, demand, capacity = "Column shear", 1800.0, 900.0   # D/C = 2
    class _A:
        H_free = 240.0
        checks = [_Ck()]
    class _R:
        assessment = _A()
    # doubling the length halves Vo, so it needs about another H_free
    assert shear_silo_required(_R()) == pytest.approx(240.0 * 1.10)


def test_a_missing_shear_check_is_not_an_error():
    from seismic_column.batch import shear_silo_required
    class _A:
        H_free = 240.0
        checks = []
    class _R:
        assessment = _A()
    assert shear_silo_required(_R()) == 0.0


@pytest.mark.slow
def test_a_short_stiff_pier_gets_a_silo_for_its_own_shear():
    """The balance search never finds this: a pier can be perfectly balanced
    with its neighbours and still fail its own column-shear check."""
    df = default_dataframe(3)
    df["Hcol_ft"] = [6.0, 7.0, 8.0]          # short enough that Vo = Mo/H bites
    out = run_batch_balanced(df, _cfg(optimize=False, balance_auto_silo=True))
    log = "\n".join(out.balance.log)
    if any(r.silo > 0 for r in out.results):
        assert "Column shear drove a silo" in log or out.balance.passed
    # whatever it did, no pier may be left failing its shear check
    for rr in out.results:
        ck = next((c for c in rr.assessment.checks
                   if c.name == "Column shear"), None)
        if ck is not None and ck.capacity > 0:
            assert ck.demand / ck.capacity <= 1.02, (rr.name, rr.silo / 12)


def test_the_shear_silo_stage_re_runs_a_pier_on_its_FRAME_basis(monkeypatch):
    """Regression: this stage runs AFTER the frame stage, so if it re-runs a
    pier bare it silently reverts it to a stand-alone cantilever on its own
    period -- throwing away the end condition and demand the frame stage just
    established, and changing the very shear it is sizing for."""
    from seismic_column import batch as B

    def _row(dc):
        class _Ck:
            name, demand, capacity = "Column shear", 900.0 * dc, 900.0
        class _A:
            H_free = 240.0
            checks = [_Ck()]
        class _R:
            name = "P1"
            silo = 0.0
            assessment = _A()
        return _R()

    basis, ef = ("K/W sentinel",), {"longitudinal": "fixed"}
    monkeypatch.setattr(B, "_frame_basis", lambda *a, **k: {"P1": (basis, ef)})
    seen = []

    def _spy(row, cfg, **kw):
        seen.append(kw)
        return _row(0.5)                      # passes, so the loop stops

    monkeypatch.setattr(B, "run_row", _spy)
    floors, _log = B._shear_silo_floors([_row(2.0)], {"P1": None}, {}, None)

    assert floors["P1"] > 0.0                 # it did size a silo
    assert len(seen) == 1
    assert seen[0]["demand_basis"] is basis
    assert seen[0]["end_fixity"] is ef


def test_the_silo_predictor_reproduces_the_real_bent_stiffness(monkeypatch):
    """Regression: ``k_at`` predicted ONE COLUMN while the rule it plans against
    -- and ``m_at`` -- are per BENT.  Two neighbours with different column
    counts then looked balanced to the planner while the real check failed, so
    it never softened anyone.  At a bent's own silo the predictor must return
    exactly the stiffness that was measured there."""
    from seismic_column import batch as B

    bent = BentStiffness(name="P1", frame="F1", order=0, Hcol=240.0, silo=0.0,
                         k=(12.0,), mass_long=1.0, mass_trans=1.0,
                         deck_link="bearing", bound_labels=("3D",),
                         n_columns=3)
    assert bent.end_fixity(LONGITUDINAL) == "free"      # so k_fixed is not used
    assert bent.stiffness(LONGITUDINAL, 0) == pytest.approx(36.0)

    class _D:
        D = 78.0
    class _Bnd:
        multiplier = 1.0
    class _A:
        EI_col = EI_shaft = 1.0e9
        bounds = [_Bnd()]
    class _RR:
        shaft = _D()
        assessment = _A()

    monkeypatch.setattr(B, "stiffness_at_silo",
                        lambda *a, **k: 4.0)            # any raw value
    k_at, _m_at, to_raw = B._silo_ctx([bent], {"P1": _RR()}, GlobalConfig(),
                                      {"P1": 0.0})
    got = k_at(bent, 0.0, 0, LONGITUDINAL)
    assert got == pytest.approx(bent.stiffness(LONGITUDINAL, 0))
    # and a target in those units converts back to what required_silo bisects on
    assert to_raw(bent, got) == pytest.approx(4.0)


# --- a bent under an expansion joint carries TWO decks ---------------------
def _joint_pattern():
    """Two continuous frames meeting at a bearing bent, as on structure B:
    F1 = B2 B3 [B4], F2 = [B4] B5 B6."""
    return [_bent("B2", [100.0], m=2.0, frame="F1", order=0,
                  deck_link="integral"),
            _bent("B3", [100.0], m=2.0, frame="F1", order=1,
                  deck_link="integral"),
            _bent("B4", [100.0], m=4.0, frame="F2", order=2,
                  deck_link="bearing"),
            _bent("B5", [100.0], m=2.0, frame="F2", order=3,
                  deck_link="integral"),
            _bent("B6", [100.0], m=2.0, frame="F2", order=4,
                  deck_link="integral")]


def test_a_joint_bent_belongs_to_both_frames_it_carries():
    frames = {f.key: f for f in frames_for(_joint_pattern(), TRANSVERSE)}
    assert frames["F1"].names == ("B2", "B3", "B4")
    assert frames["F2"].names == ("B4", "B5", "B6")
    assert frames["F1"].shared("B4") and frames["F2"].shared("B4")
    assert not frames["F1"].shared("B2")


def test_the_joint_bent_brings_full_stiffness_but_half_its_mass():
    """Each frame is analysed alone and a rigid deck leans on the WHOLE bent,
    so k is full in both; the mass is the half-span belonging to that deck."""
    frames = {f.key: f for f in frames_for(_joint_pattern(), TRANSVERSE)}
    assert frames["F1"].K(0) == pytest.approx(300.0)      # 100+100+100, B4 full
    assert frames["F1"].M() == pytest.approx(2.0 + 2.0 + 0.5 * 4.0)
    assert frames["F2"].K(0) == pytest.approx(300.0)
    assert frames["F2"].M() == pytest.approx(0.5 * 4.0 + 2.0 + 2.0)


def test_a_released_bearing_is_not_a_longitudinal_frame_at_all():
    """It carries no deck longitudinally -- only its own cap and column self
    weight -- so pairing it against a real frame's period is meaningless."""
    lon = [f.names for f in frames_for(_joint_pattern(), LONGITUDINAL)]
    assert lon == [("B2", "B3"), ("B5", "B6")]
    assert not any("B4" in names for names in lon)


def test_a_joint_bent_is_not_shared_into_a_simple_span_frame():
    """A run of simple spans is one frame per bent, and that bent's tributary
    already IS the frame mass.  Sharing in would count the span beyond the
    joint twice."""
    bents = [_bent("A6", [100.0], m=2.0, frame="FA6", order=0,
                   deck_link="pinned"),
             _bent("A7", [100.0], m=4.0, frame="C1", order=1,
                   deck_link="bearing"),
             _bent("A8", [100.0], m=2.0, frame="C1", order=2,
                   deck_link="integral")]
    frames = {f.key: f for f in frames_for(bents, TRANSVERSE)}
    assert frames["FA6"].names == ("A6",)
    assert frames["FA6"].M() == pytest.approx(2.0)        # untouched
    assert frames["C1"].names == ("A7", "A8")
    assert frames["C1"].M() == pytest.approx(6.0)         # A7 at FULL mass


def test_naming_two_frames_puts_the_bent_in_both():
    """A pier under an expansion joint carries a deck either side.  Declaring
    'F1, F2' says so explicitly, instead of leaving it to be inferred."""
    bents = [_bent("B2", [100.0], m=2.0, frame="F1", order=0,
                   deck_link="integral"),
             _bent("B3", [100.0], m=4.0, frame="F1, F2", order=1,
                   deck_link="bearing"),
             _bent("B4", [100.0], m=2.0, frame="F2", order=2,
                   deck_link="integral")]
    frames = {f.key: f for f in frames_for(bents, TRANSVERSE)}
    assert frames["F1"].names == ("B2", "B3")
    assert frames["F2"].names == ("B3", "B4")
    assert frames["F1"].shared("B3") and frames["F2"].shared("B3")
    # full stiffness in each, half the mass in each
    assert frames["F1"].K(0) == pytest.approx(200.0)
    assert frames["F1"].M() == pytest.approx(2.0 + 0.5 * 4.0)
    assert frames["F2"].M() == pytest.approx(0.5 * 4.0 + 2.0)


def test_an_explicit_pair_works_where_the_automatic_rule_declines():
    """The automatic rule will not share into a single-bent frame, because a
    run of simple spans is modelled one frame per bent.  Naming both frames
    overrides that -- it is a statement about the articulation."""
    def build(last_frame):
        return [_bent("B21", [100.0], m=2.0, frame="F9", order=0,
                      deck_link="integral"),
                _bent("B22", [100.0], m=4.0, frame=last_frame, order=1,
                      deck_link="bearing"),
                _bent("B23", [100.0], m=2.0, frame="F10", order=2,
                      deck_link="pinned")]
    auto = {f.key: f for f in frames_for(build("F9"), TRANSVERSE)}
    assert auto["F10"].names == ("B23",)              # declined, as designed
    told = {f.key: f for f in frames_for(build("F9, F10"), TRANSVERSE)}
    assert told["F10"].names == ("B22", "B23")
    assert told["F10"].M() == pytest.approx(0.5 * 4.0 + 2.0)


def test_a_blank_or_dashed_entry_in_a_list_is_ignored():
    from seismic_column.io_schema import frame_keys
    assert frame_keys("F1, F2") == ("F1", "F2")
    assert frame_keys(" F9 ; F10 ") == ("F9", "F10")
    assert frame_keys("F1,F1") == ("F1",)             # a repeat is not a share
    assert frame_keys("F2, -") == ("F2",)
    assert frame_keys("-") == () and frame_keys("") == ()


# --- a pier can meet its two decks DIFFERENTLY -----------------------------
def _a7_pattern():
    """The EN bust: C1 sits on free bearings at BOTH ends, but the pier at one
    end also PINS the simple span beside it, and the pier at the other end only
    rollers its neighbour.  One deck_link per bent cannot say that."""
    return [_bent("A6", [100.0], m=2.0, frame="SA6", order=0,
                  deck_link="bearing"),
            _bent("A7", [100.0], m=4.0, frame="SA6, C1", order=1,
                  deck_link="pinned, bearing"),
            _bent("A8", [100.0], m=2.0, frame="C1", order=2,
                  deck_link="integral"),
            _bent("A10", [100.0], m=2.0, frame="C1, SA11", order=3,
                  deck_link="bearing, bearing"),
            _bent("A11", [100.0], m=2.0, frame="SA11", order=4,
                  deck_link="pinned")]


def test_a_pier_pinning_one_span_carries_it_all_longitudinally():
    """A7 pins SA6 and only bears C1, so longitudinally ALL of its tributary
    belongs to SA6 and none to C1 -- it is not a longitudinal member of C1."""
    lon = {f.key: f for f in frames_for(_a7_pattern(), LONGITUDINAL)}
    assert lon["SA6"].names == ("A7",)          # A6 only rollers it
    assert lon["SA6"].M() == pytest.approx(4.0)  # the WHOLE tributary, not half
    assert lon["C1"].names == ("A8",)           # both ends are free bearings
    assert "SA11" in lon and lon["SA11"].names == ("A11",)


def test_the_same_pier_splits_its_mass_transversely():
    """Shear keys engage both decks, so transversely A7 is in both frames at
    half its tributary -- the direction changes the answer, not the data."""
    tra = {f.key: f for f in frames_for(_a7_pattern(), TRANSVERSE)}
    assert tra["SA6"].names == ("A6", "A7")
    assert tra["C1"].names == ("A7", "A8", "A10")
    assert tra["SA6"].M() == pytest.approx(2.0 + 0.5 * 4.0)
    assert tra["SA6"].shared("A7") and tra["C1"].shared("A7")


def test_a_pier_free_to_both_decks_is_no_longitudinal_frame():
    """A10 has a free bearing under C1 and a roller under the next span, so it
    restrains nothing longitudinally -- only its own self weight."""
    lon = {f.key: f for f in frames_for(_a7_pattern(), LONGITUDINAL)}
    assert not any("A10" in f.names for f in lon.values())
    tra = {f.key: f for f in frames_for(_a7_pattern(), TRANSVERSE)}
    assert tra["C1"].shared("A10") and tra["SA11"].shared("A10")


def test_deck_link_list_must_line_up_with_the_frame_list():
    from seismic_column.io_schema import validate
    df = default_dataframe(2)
    df["frame"] = ["F1, F2", "F2"]
    df["deck_link"] = ["pinned, bearing, integral", "pinned"]
    with pytest.raises(ValueError, match="one link per frame"):
        validate(df)


# --- the simple-span stiffness switch --------------------------------------
def _simple_run():
    """Two simply supported spans in series: each span is a frame carried by
    two piers on bearings, and the middle pier carries both."""
    return [_bent("P1", [100.0], m=2.0, frame="S1", order=0,
                  deck_link="pinned"),
            _bent("P2", [300.0], m=2.0, frame="S1, S2", order=1,
                  deck_link="bearing, pinned"),
            _bent("P3", [110.0], m=2.0, frame="S2", order=2,
                  deck_link="bearing")]


def test_simple_spans_are_matched_on_period_only_by_default():
    checks = balance_checks(_simple_run(), BalanceCriteria())
    assert not [c for c in checks
                if c.name in (STIFFNESS_CHECK, STIFFNESS_ANY_CHECK)]
    assert [c for c in checks if c.name == GEOMETRY_CHECK]


def test_the_switch_holds_the_two_piers_of_a_span_to_the_adjacent_limit():
    crit = BalanceCriteria(simple_span_stiffness=True)
    # TRANSVERSELY the span has two supports (both shear-keyed); longitudinally
    # it hangs off its pin alone, so there is no pair to compare there.
    checks = [c for c in balance_checks(_simple_run(), crit)
              if c.name == STIFFNESS_CHECK and c.direction == TRANSVERSE]
    assert [c.pair for c in checks] == [("P1", "P2"), ("P2", "P3")]
    assert not checks[0].passed                   # 100 vs 300 on equal mass
    assert all(c.limit == crit.k_ratio_min for c in checks)
    assert not [c for c in balance_checks(_simple_run(), crit)
                if c.name == STIFFNESS_ANY_CHECK]


def test_the_switch_leaves_continuous_frames_alone():
    """A continuous frame is checked either way; the switch only reaches piers
    that are alone in their frame."""
    bents = _joint_pattern()
    off = balance_checks(bents, BalanceCriteria())
    on = balance_checks(bents, BalanceCriteria(simple_span_stiffness=True))
    key = lambda cs: sorted((c.name, c.pair, c.direction) for c in cs)
    assert key(off) == key(on)


def test_the_switch_ignores_a_continuous_frame_with_one_active_member():
    """F1 is B2+B3+B4 but B4 is released longitudinally, so only two members
    resist there.  A continuous frame reduced to one member in a direction is
    still NOT a simply supported span and must not be caught by the rule."""
    crit = BalanceCriteria(simple_span_stiffness=True)
    frames = {f.key: f for f in frames_for(_joint_pattern(), LONGITUDINAL)}
    assert frames["F1"].names == ("B2", "B3")     # B4 released here ...
    assert not frames["F1"].simply_supported      # ... but B2/B3 are integral
    # so it is checked for stiffness with the switch either way
    on = _long(balance_checks(_joint_pattern(), crit), STIFFNESS_CHECK)
    off = _long(balance_checks(_joint_pattern(), BalanceCriteria()),
                STIFFNESS_CHECK)
    assert [c.pair for c in on] == [c.pair for c in off] != []


def test_engineer_vocabulary_is_accepted_for_deck_links():
    from seismic_column.io_schema import deck_link_word, deck_links
    assert deck_links("roller, pin", 2) == ("bearing", "pinned")
    assert deck_links("expansion", 2) == ("bearing", "bearing")
    assert deck_links("fixed", 1) == ("pinned",)
    assert deck_link_word("bearing") == "roller (expansion)"
    assert deck_link_word("pinned") == "pin (fixed bearing)"
