"""Balanced stiffness / balanced frame geometry between adjacent piers, and the
column silo that tunes them."""
import math

import pytest

from seismic_column.balance import (
    GEOMETRY_CHECK,
    STIFFNESS_CHECK,
    BalanceCriteria,
    BentStiffness,
    adjacent_pairs,
    balance_checks,
    quantise_silo,
    required_silo,
)
from seismic_column.batch import run_batch_balanced
from seismic_column.geometry import Geometry
from seismic_column.io_schema import (
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
def _bent(name, k, m=1.0, frame="F1", order=0, T=None):
    if T is None:
        T = tuple(2.0 * math.pi * math.sqrt(m / ki) for ki in k)
    return BentStiffness(name=name, frame=frame, order=order, Hcol=240.0,
                         silo=0.0, mass=m, k=tuple(k), T=tuple(T),
                         bound_labels=("3D",) * len(k))


def test_ratio_is_min_over_max_so_order_does_not_matter():
    crit = BalanceCriteria()
    a, b = _bent("A", [10.0], order=0), _bent("B", [8.0], order=1)
    fwd = balance_checks([a, b], crit)
    rev = balance_checks([_bent("B", [8.0], order=0), _bent("A", [10.0], order=1)],
                         crit)
    assert fwd[0].ratio == pytest.approx(0.8)
    assert rev[0].ratio == pytest.approx(0.8)
    assert fwd[0].passed and rev[0].passed


def test_stiffness_check_fails_below_the_limit():
    checks = balance_checks([_bent("A", [10.0], order=0),
                             _bent("B", [5.0], order=1)], BalanceCriteria())
    stiff = [c for c in checks if c.name == STIFFNESS_CHECK]
    assert stiff[0].ratio == pytest.approx(0.5)
    assert not stiff[0].passed
    assert stiff[0].limit == 0.75


def test_period_ratio_is_the_square_root_of_the_kappa_ratio():
    """With kappa = k/m and T = 2*pi*sqrt(m/k), (Ti/Tj)^2 = kappa_j/kappa_i."""
    checks = balance_checks([_bent("A", [10.0], m=2.0, order=0),
                             _bent("B", [6.0], m=2.0, order=1)],
                            BalanceCriteria(mass_normalized=True))
    rk = next(c.ratio for c in checks if c.name == STIFFNESS_CHECK)
    rt = next(c.ratio for c in checks if c.name == GEOMETRY_CHECK)
    assert rt == pytest.approx(math.sqrt(rk))
    # ... so the 0.75 stiffness rule is stricter than the 0.70 period rule
    assert math.sqrt(0.75) > 0.70


def test_mass_normalisation_changes_the_verdict():
    """Equal stiffness but unequal tributary mass fails the normalised form."""
    bents = [_bent("A", [10.0], m=1.0, order=0), _bent("B", [10.0], m=2.0, order=1)]
    assert balance_checks(bents, BalanceCriteria(mass_normalized=False))[0].passed
    assert not balance_checks(bents, BalanceCriteria(mass_normalized=True))[0].passed


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
    checks = balance_checks(bents, BalanceCriteria())
    assert not checks[0].passed
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
def test_default_table_is_unbalanced_before_any_silo():
    df = default_dataframe(3)                  # 18 / 22 / 26 ft
    out = run_batch_balanced(df, _cfg(optimize=False, balance_auto_silo=False))
    assert out.balance is not None
    assert not out.balance.passed
    assert all(r.silo == 0.0 for r in out.results)      # nothing was changed
    assert any(c.name == STIFFNESS_CHECK and not c.passed
               for c in out.balance.checks)


@pytest.mark.parametrize("code", CODES)
def test_auto_silo_balances_and_every_column_still_passes_seismic(code):
    df = default_dataframe(3)
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


def test_a_silo_forces_a_real_seismic_re_analysis():
    """The pier that gets a silo must be re-analysed, not just re-labelled:
    Lp, the demands and (in an optimise run) the reinforcement all move."""
    df = default_dataframe(3)
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


def test_entered_silo_is_a_floor_never_reduced():
    df = default_dataframe(3)
    df.loc[2, "silo_ft"] = 3.0                 # the most flexible pier already
    out = run_batch_balanced(df, _cfg(balance_auto_silo=True))
    c3 = next(r for r in out.results if r.name == "C3")
    assert c3.silo >= 3.0 * 12.0 - 1e-9


def test_entered_silo_deeper_than_the_cap_is_honoured():
    """The cap limits what the TOOL adds; a typed-in silo is the engineer's."""
    df = default_dataframe(3)
    df.loc[0, "silo_ft"] = 8.0
    out = run_batch_balanced(df, _cfg(balance_auto_silo=True, max_silo_ft=2.0))
    c1 = next(r for r in out.results if r.name == "C1")
    assert c1.silo == pytest.approx(8.0 * 12.0)
    assert c1.assessment.H_free == pytest.approx(c1.assessment.Hcol_entered
                                                 + 8.0 * 12.0)


def test_balance_progress_is_separate_from_the_row_progress_count():
    """``progress`` keeps its once-per-row contract; the balance stage reports
    through ``balance_progress`` instead of corrupting the done-count."""
    df = default_dataframe(3)
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
    """Run the DP against an explicit {name: {silo: k}} table."""
    from seismic_column.balance import dp_min_silo

    states = states or {n: tuple(sorted(v)) for n, v in k_table.items()}
    return dp_min_silo(
        bents, states,
        k_of=lambda b, s, bound: k_table[b.name][s],
        m_of=lambda b, s: b.mass,
        criteria=criteria)


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
    from seismic_column.balance import dp_min_silo

    crit = BalanceCriteria(mass_normalized=True)
    grid_a, grid_b = (24.0, 36.0), (0.0, 12.0)       # P1 floored at 24 in
    k = {"P1": {24.0: 200.0, 36.0: 180.0}, "P2": {0.0: 190.0, 12.0: 170.0}}
    bents = [_bent("P1", [200.0], order=0), _bent("P2", [190.0], order=1)]
    plan = dp_min_silo(bents, {"P1": grid_a, "P2": grid_b},
                       k_of=lambda b, s, i: k[b.name][s],
                       m_of=lambda b, s: b.mass, criteria=crit)
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


def test_min_silo_result_is_on_the_grid_and_verified():
    df = default_dataframe(3)
    out = run_batch_balanced(df, _cfg(balance_strategy="min_silo",
                                      silo_step_ft=2.0))
    assert out.balance.passed
    for rr in out.results:
        ft = rr.silo / 12.0
        assert abs(ft / 2.0 - round(ft / 2.0)) < 1e-6, f"{rr.name} off-grid: {ft}"


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
    df["weight_kip"] = [900.0, 2050.0]
    cfg = _cfg(optimize=False, balance_mass_normalized=False,
               balance_auto_silo=False)
    before = run_batch_balanced(df, cfg).balance
    k_ok = [c for c in before.checks if c.name == STIFFNESS_CHECK]
    t_bad = [c for c in before.checks if c.name == GEOMETRY_CHECK]
    assert all(c.passed for c in k_ok), "stiffness should pass on equal sections"
    assert not all(c.passed for c in t_bad), "period should fail on unequal mass"

    after = run_batch_balanced(df, _cfg(optimize=False,
                                        balance_mass_normalized=False,
                                        balance_auto_silo=True))
    assert after.balance.passed, [c.label for c in after.balance.failed]
    # the LIGHTER pier has the shorter period, so it is the one softened
    silos = {r.name: r.silo for r in after.results}
    assert silos["C1"] > 0 and silos["C2"] == 0


def test_mass_ratio_window_bounds_the_two_rules():
    """The two rules constrain the same k ratio from opposite sides."""
    from seismic_column.balance import joint_feasible, mass_ratio_window

    crit = BalanceCriteria(mass_normalized=False)          # 0.75 / 0.70
    lo, hi = mass_ratio_window(crit)
    assert lo == pytest.approx(0.75 * 0.70 ** 2)           # 0.3675
    assert hi == pytest.approx(1.0 / (0.75 * 0.70 ** 2))   # 2.721

    ok, mu, (x_lo, x_hi) = joint_feasible(
        _bent("A", [100.0], m=1.0, order=0),
        _bent("B", [100.0], m=1.9, order=1), crit)
    assert ok and x_lo <= x_hi                              # mu 0.53, tight
    bad, mu2, _ = joint_feasible(
        _bent("A", [100.0], m=1.0, order=0),
        _bent("B", [100.0], m=4.0, order=1), crit)[0:3]
    assert not bad                                          # mu 0.25 < 0.3675

    # mass normalisation ON makes the period rule automatic, so no window
    assert mass_ratio_window(BalanceCriteria(mass_normalized=True))[1] == float("inf")


def test_infeasible_mass_disparity_is_reported_not_ground_against():
    """A pair outside the mass window can't be fixed by any silo — say so."""
    df = default_dataframe(2)
    df["Hcol_ft"] = 22.0
    df["weight_kip"] = [700.0, 3200.0]            # ratio well past 2.72
    out = run_batch_balanced(df, _cfg(optimize=False,
                                      balance_mass_normalized=False,
                                      balance_auto_silo=True))
    assert not out.balance.passed
    assert any("INFEASIBLE" in line for line in out.balance.log)
    assert any("mass normalisation" in line for line in out.balance.log)


def test_balance_check_off_runs_nothing():
    df = default_dataframe(3)
    out = run_batch_balanced(df, _cfg(optimize=False, balance_check=False))
    assert out.balance is None
    assert all(r.silo == 0.0 for r in out.results)
    assert "balanced" not in out.summary.columns


def test_excluded_piers_are_left_out_of_the_pairs():
    df = default_dataframe(3)
    df.loc[1, "frame"] = ""                    # C2 opts out
    out = run_batch_balanced(df, _cfg(optimize=False, balance_auto_silo=False))
    pairs = {c.pair for c in out.balance.checks}
    assert pairs == {("C1", "C3")}             # C2 removed, C1-C3 now adjacent
    assert out.summary.loc[out.summary["name"] == "C2", "balanced"].iloc[0] == "-"


def test_separate_frames_are_never_compared():
    df = default_dataframe(4)
    df["frame"] = ["A", "A", "B", "B"]
    out = run_batch_balanced(df, _cfg(optimize=False, balance_auto_silo=False))
    assert {c.pair for c in out.balance.checks} == {("C1", "C2"), ("C3", "C4")}


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
def test_balance_report_is_complete_and_clean():
    from seismic_column.report import balance_report, column_report
    df = default_dataframe(3)
    out = run_batch_balanced(df, _cfg(balance_auto_silo=True))
    txt = balance_report(out.balance)
    for heading in ("# Balanced stiffness & balanced frame geometry",
                    "## Bents", "## Adjacent pairs", "## Balancing log"):
        assert heading in txt
    assert "nan" not in txt and "None" not in txt
    # the per-column report must show the silo it was analysed with
    silo_rr = next(r for r in out.results if r.silo > 0)
    col_txt = column_report(silo_rr)
    assert "Column silo" in col_txt and "H_free" in col_txt
    assert "nan" not in col_txt
