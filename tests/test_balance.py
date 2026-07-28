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
FAST = ("longitudinal", "confinement", "fc")


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
          deck_link="pinned", k_fixed=None):
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
    """A bearing released longitudinally but shear-keyed joins the continuous
    frame transversely, and stands alone longitudinally."""
    bents = _en_pattern()
    lon = [f.names for f in frames_for(bents, LONGITUDINAL)]
    tra = [f.names for f in frames_for(bents, TRANSVERSE)]
    assert lon == [("A6",), ("A7",), ("A8", "A9", "A10"), ("A11",), ("A12",)]
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
    cap value and what to do about it — never an infinite loop or a bare FAIL."""
    df = default_dataframe(3)
    df.loc[0, "Hcol_ft"] = 12.0
    df.loc[2, "Hcol_ft"] = 40.0
    out = run_batch_balanced(df, _cfg(balance_auto_silo=True, max_silo_ft=1.0,
                                      balance_strategy=strategy))
    assert not out.balance.passed
    assert not out.balance.converged
    log = "\n".join(out.balance.log)
    assert "cap" in log                       # names the binding constraint
    assert "raise the cap" in log             # and what to do about it
    assert "UNRESOLVED" in log
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
    # C2/C4 are released longitudinally, so C3 is alone in the frame there
    assert ("C3",) in lon and ("C2",) in lon and ("C4",) in lon
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
    assert list(out["frame"]) == ["F1", "F1"]     # one run of piers in series
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
