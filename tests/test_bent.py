"""Multi-column bents: the transverse push/pull couple and what follows from it."""
from __future__ import annotations

import pytest

from seismic_column.balance import BentStiffness, bent_stiffness
from seismic_column.bent import couple_axials, evaluate_bent, offsets
from seismic_column.demand import DesignSpectrum
from seismic_column.geometry import Geometry
from seismic_column.io_schema import LONGITUDINAL, TRANSVERSE
from seismic_column.section import CircularSection


def _kw(**over):
    kw = dict(
        column=CircularSection(D=48, fc=4, cover=2, n_bars=24, long_bar_no=11,
                               spiral_bar_no=6, spiral_spacing=3),
        shaft=CircularSection(D=84, fc=4, cover=3, n_bars=40, long_bar_no=11,
                              spiral_bar_no=6, spiral_spacing=3.5),
        geometry=Geometry(Hcol=18 * 12, D_shaft=84),
        spectrum=DesignSpectrum(Sds=1.0, Sd1=0.6),
        weight=1200.0, weight_trans=1200.0)
    kw.update(over)
    return kw


# --- the couple is statics, so assert the statics -------------------------
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_the_couple_carries_no_net_axial_and_reproduces_the_moment(n):
    s, M = 240.0, 1.0e6
    xs, dPs = offsets(n, s), couple_axials(n, s, M)
    assert sum(dPs) == pytest.approx(0.0, abs=1e-6)          # a couple, not a load
    assert sum(d * x for d, x in zip(dPs, xs)) == pytest.approx(M)


def test_offsets_are_centred_on_the_bent():
    assert offsets(2, 100.0) == [-50.0, 50.0]
    assert offsets(3, 100.0) == [-100.0, 0.0, 100.0]
    assert sum(offsets(4, 90.0)) == pytest.approx(0.0)


def test_a_single_column_has_no_couple_to_distribute():
    assert couple_axials(1, 0.0, 1.0e6) == [0.0]


def test_closer_columns_take_a_bigger_couple():
    """The spacing IS the lever arm — halving it doubles the axial."""
    wide = couple_axials(2, 480.0, 1.0e6)
    tight = couple_axials(2, 240.0, 1.0e6)
    assert tight[1] == pytest.approx(2.0 * wide[1])


# --- n = 1 must be a pass-through ------------------------------------------
def test_single_column_bent_is_a_pass_through():
    b = evaluate_bent(1, 0.0, 700.0, **_kw())
    assert not b.multi
    assert len(b.positions) == 1
    assert b.delta_P == 0.0
    assert b.positions[0].axial == 700.0
    # and it carries the column's own checks untouched
    assert b.checks == b.assessment.checks


def test_single_column_bent_matches_evaluate_column_exactly():
    from seismic_column.sdc_capacity import evaluate_column
    direct = evaluate_column(axial=700.0, **_kw())
    viab = evaluate_bent(1, 0.0, 700.0, **_kw()).assessment
    for d in (LONGITUDINAL, TRANSVERSE):
        a, b = direct.directions[d], viab.directions[d]
        assert a.governing_bound.delta_c == pytest.approx(b.governing_bound.delta_c)
        assert (a.governing_bound.demand.disp_demand
                == pytest.approx(b.governing_bound.demand.disp_demand))
        assert a.Vo == pytest.approx(b.Vo)


# --- what the couple does to the columns -----------------------------------
def test_the_windward_column_is_unloaded_and_the_leeward_one_compressed():
    b = evaluate_bent(2, 30 * 12.0, 700.0, **_kw())
    wind, lee = b.positions
    assert wind.x < 0 < lee.x
    assert wind.delta_P < 0 < lee.delta_P
    assert wind.axial < 700.0 < lee.axial
    assert wind.delta_P == pytest.approx(-lee.delta_P)


def test_axial_changes_the_moment_capacity_of_the_same_section():
    """Same column, different axial -- that is the whole point."""
    b = evaluate_bent(2, 30 * 12.0, 700.0, **_kw())
    wind, lee = b.positions
    assert lee.assessment.mc_col.Mp > wind.assessment.mc_col.Mp
    # and so the overstrength shear each hands its shaft differs
    assert (lee.assessment.directions[TRANSVERSE].Vo
            > wind.assessment.directions[TRANSVERSE].Vo)


def test_a_short_lever_arm_can_put_the_windward_column_into_tension():
    wide = evaluate_bent(2, 40 * 12.0, 700.0, **_kw())
    tight = evaluate_bent(2, 8 * 12.0, 700.0, **_kw())
    assert not wide.positions[0].net_tension
    assert tight.positions[0].net_tension
    assert any("NET TENSION" in m for m in tight.log)


def test_the_transverse_head_is_fixed_by_the_cap():
    """A multi-column bent is a portal frame, not two cantilevers."""
    b = evaluate_bent(2, 30 * 12.0, 700.0, **_kw())
    for p in b.positions:
        assert p.assessment.directions[TRANSVERSE].end_fixity == "fixed"
        # ... but longitudinally there is no cap action and no couple
        assert p.assessment.directions[LONGITUDINAL].end_fixity == "free"


def test_the_push_pull_settles():
    b = evaluate_bent(2, 30 * 12.0, 700.0, **_kw())
    assert b.converged
    assert b.iterations <= 5


def test_the_couple_equals_the_sum_of_the_column_overstrength_moments():
    """Statics: cut at the top of shaft, take moments about the centreline.

    ``V_bent*H = sum(dP*x) + sum(Mo)`` and ``V_bent = sum(2*Mo/H)``, so the
    couple carries exactly ``sum(Mo)`` — the base moments take the rest.
    Charging it the whole of ``V*Le`` overstates it several times over.
    """
    b = evaluate_bent(2, 30 * 12.0, 700.0, **_kw())
    xs = [p.x for p in b.positions]
    # the reported couple must reconcile with the reported axials EXACTLY
    assert (sum(p.delta_P * x for p, x in zip(b.positions, xs))
            == pytest.approx(b.M_overturn))
    # and match sum(Mo) of the final runs to within convergence: a fixed point
    # always carries one step of lag, and it is parked here deliberately
    assert b.M_overturn == pytest.approx(
        sum(p.assessment.Mo for p in b.positions), rel=0.01)


# --- the balance layer sees a BENT, not a column ---------------------------
def _bs(n):
    return BentStiffness(name="B", frame="F", order=0, Hcol=240.0, silo=0.0,
                         k=(100.0,), mass_long=4.0, mass_trans=4.0,
                         deck_link="pinned", n_columns=n, k_fixed=(400.0,),
                         bound_labels=("best",))


def test_a_multi_column_bent_is_fixed_fixed_transversely():
    assert _bs(1).end_fixity(TRANSVERSE) == "free"
    assert _bs(2).end_fixity(TRANSVERSE) == "fixed"
    # longitudinally the cap does nothing; deck_link still decides
    assert _bs(2).end_fixity(LONGITUDINAL) == "free"


def test_bent_stiffness_adds_the_columns_up():
    one, two = _bs(1), _bs(2)
    assert two.stiffness(LONGITUDINAL, 0) == pytest.approx(
        2.0 * one.stiffness(LONGITUDINAL, 0))
    # transversely BOTH effects apply: twice the columns AND the fixed head
    assert two.stiffness(TRANSVERSE, 0) == pytest.approx(2.0 * 400.0)
    assert one.stiffness(TRANSVERSE, 0) == pytest.approx(100.0)


# --- wide bents, and the per-bent axial ------------------------------------
def test_the_bent_axial_is_shared_between_the_columns():
    """The table gives the BENT reaction, as it does for the tributary weight."""
    one = evaluate_bent(1, 0.0, 1400.0, **_kw())
    two = evaluate_bent(2, 30 * 12.0, 1400.0, **_kw())
    assert one.positions[0].axial == pytest.approx(1400.0)
    # 1400 on the bent, shared two ways, then +/- the couple
    assert sum(p.delta_P for p in two.positions) == pytest.approx(0.0)
    assert (sum(p.axial for p in two.positions)
            == pytest.approx(1400.0))


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_wide_bents_work_and_only_the_extremes_are_analysed(n):
    b = evaluate_bent(n, 20 * 12.0, 700.0 * n, **_kw())
    assert len(b.positions) == n
    # the extremes carry the largest swing, either side of zero
    assert b.positions[0].delta_P < 0 < b.positions[-1].delta_P
    assert all(abs(p.delta_P) <= abs(b.positions[-1].delta_P) + 1e-9
               for p in b.positions)
    # interior columns reuse the dead-load analysis, so no extra p-y solves
    if n > 2:
        mids = {id(p.assessment) for p in b.positions[1:-1]}
        assert len(mids) == 1
        assert id(b.positions[0].assessment) not in mids
    # and one of the extremes governs
    assert b.governing.index in (0, n - 1)


def test_a_wide_bent_says_how_the_interior_columns_were_treated():
    b = evaluate_bent(4, 15 * 12.0, 2800.0, **_kw())
    assert any("interior" in m for m in b.log)


# --- a PINNED cap removes the frame action entirely ------------------------
def test_a_pinned_cap_develops_no_push_pull_couple():
    """Statics: V*H = sum(Mo) and the base moments are sum(Mo), so nothing left.

    Physically each column becomes an independent cantilever — there is no cap
    moment to transfer, so no couple between the columns.
    """
    b = evaluate_bent(2, 30 * 12.0, 1400.0, cap_fixity="pinned", **_kw())
    assert b.multi
    assert b.delta_P == 0.0
    assert all(p.delta_P == 0.0 for p in b.positions)
    assert all(p.axial == pytest.approx(700.0) for p in b.positions)
    assert any("PINNED" in m for m in b.log)


def test_a_pinned_cap_stays_fixed_free_transversely_and_halves_Vo():
    fixed = evaluate_bent(2, 30 * 12.0, 1400.0, cap_fixity="fixed", **_kw())
    pinned = evaluate_bent(2, 30 * 12.0, 1400.0, cap_fixity="pinned", **_kw())
    assert pinned.positions[0].assessment.directions[TRANSVERSE].end_fixity == "free"
    assert fixed.positions[0].assessment.directions[TRANSVERSE].end_fixity == "fixed"
    # one hinge instead of two, at the same axial -> half the overstrength shear
    pin_Vo = pinned.positions[0].assessment.directions[TRANSVERSE].Vo
    mid_Vo = evaluate_bent(1, 0.0, 700.0, **_kw()).assessment \
        .directions[TRANSVERSE].Vo
    assert pin_Vo == pytest.approx(mid_Vo)


def test_a_pinned_cap_is_softer_in_the_balance_layer():
    from seismic_column.balance import BentStiffness
    def bs(cap):
        return BentStiffness(name="B", frame="F", order=0, Hcol=240.0, silo=0.0,
                             k=(100.0,), mass_long=4.0, mass_trans=4.0,
                             deck_link="pinned", n_columns=2, k_fixed=(400.0,),
                             cap_fixity=cap, bound_labels=("best",))
    assert bs("fixed").end_fixity(TRANSVERSE) == "fixed"
    assert bs("pinned").end_fixity(TRANSVERSE) == "free"
    # two columns either way, but the pinned cap keeps the cantilever stiffness
    assert bs("pinned").stiffness(TRANSVERSE, 0) == pytest.approx(200.0)
    assert bs("fixed").stiffness(TRANSVERSE, 0) == pytest.approx(800.0)


# --- production hardening: schema, validation, wide bents ------------------
def test_a_table_without_the_multicolumn_columns_still_loads():
    """Every project written before multi-column bents existed."""
    from seismic_column.io_schema import default_dataframe, validate
    df = default_dataframe(3).drop(columns=["n_columns", "col_spacing_ft",
                                            "cap_fixity"])
    v = validate(df)
    assert (v["n_columns"] == 1).all()          # single-column, as it meant
    assert (v["col_spacing_ft"] == 0.0).all()
    assert (v["cap_fixity"] == "fixed").all()   # ignored while n_columns == 1


def test_a_multicolumn_bent_without_a_spacing_is_rejected():
    """The spacing IS the lever arm; without it there is no couple to find."""
    from seismic_column.io_schema import default_dataframe, validate
    df = default_dataframe(2)
    df.loc[0, "n_columns"] = 2                  # no col_spacing_ft
    with pytest.raises(ValueError, match="col_spacing_ft"):
        validate(df)


def test_zero_or_negative_column_count_is_rejected():
    from seismic_column.io_schema import default_dataframe, validate
    df = default_dataframe(2)
    df.loc[0, "n_columns"] = 0
    with pytest.raises(ValueError, match="n_columns"):
        validate(df)


def test_an_unknown_cap_fixity_is_rejected():
    from seismic_column.io_schema import default_dataframe, validate
    df = default_dataframe(2)
    df.loc[0, ["n_columns", "col_spacing_ft", "cap_fixity"]] = [2, 20.0, "welded"]
    with pytest.raises(ValueError, match="cap_fixity"):
        validate(df)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 8])
def test_wide_bents_keep_the_statics_exact(n):
    """Multi-cell bents: the couple must balance at any width."""
    b = evaluate_bent(n, 18 * 12.0, 700.0 * n, **_kw())
    xs = [p.x for p in b.positions]
    assert sum(p.delta_P for p in b.positions) == pytest.approx(0.0, abs=1e-6)
    assert (sum(p.delta_P * x for p, x in zip(b.positions, xs))
            == pytest.approx(b.M_overturn))
    # the axial swing falls as the bent widens -- same overturning, bigger Sx^2
    assert b.delta_P > 0
    # cost stays bounded however wide it gets
    assert len({id(p.assessment) for p in b.positions}) <= 3


def test_a_wider_bent_takes_a_smaller_axial_swing():
    narrow = evaluate_bent(2, 18 * 12.0, 1400.0, **_kw())
    wide = evaluate_bent(6, 18 * 12.0, 4200.0, **_kw())
    assert wide.delta_P < narrow.delta_P


@pytest.mark.parametrize("n", [3, 5, 7])
def test_an_odd_bent_has_a_centre_column_with_no_swing(n):
    b = evaluate_bent(n, 20 * 12.0, 700.0 * n, **_kw())
    mid = b.positions[n // 2]
    assert mid.x == pytest.approx(0.0)
    assert mid.delta_P == pytest.approx(0.0)


def test_the_governing_position_is_always_an_extreme():
    for n in (2, 3, 4, 5, 6):
        b = evaluate_bent(n, 18 * 12.0, 700.0 * n, **_kw())
        assert b.governing.index in (0, n - 1)
