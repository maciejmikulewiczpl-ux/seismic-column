"""Two-direction envelope: the merge, and that capacity stays direction-free."""
from __future__ import annotations

import pytest

from seismic_column.demand import DesignSpectrum
from seismic_column.geometry import Geometry
from seismic_column.io_schema import LONGITUDINAL, TRANSVERSE
from seismic_column.sdc_capacity import (Check, DirectionalResult,
                                         _envelope_checks, evaluate_column)
from seismic_column.section import CircularSection


def _assess(weight, weight_trans):
    col = CircularSection(D=48, fc=4, cover=2, n_bars=24, long_bar_no=11,
                          spiral_bar_no=6, spiral_spacing=3)
    shaft = CircularSection(D=84, fc=4, cover=3, n_bars=40, long_bar_no=11,
                            spiral_bar_no=6, spiral_spacing=3.5)
    return evaluate_column(col, shaft, Geometry(Hcol=18 * 12, D_shaft=84),
                           DesignSpectrum(Sds=1.0, Sd1=0.6), axial=700,
                           weight=weight, weight_trans=weight_trans)


def _dres(direction, checks):
    return DirectionalResult(direction, 0.0, 0.0, [], None, checks)


def _pair(lon_checks, trn_checks):
    return {LONGITUDINAL: _dres(LONGITUDINAL, lon_checks),
            TRANSVERSE: _dres(TRANSVERSE, trn_checks)}


def test_one_direction_passes_straight_through():
    """A single-direction run must be byte-for-byte what it always was."""
    cs = [Check("A", 1.0, 2.0, True, "note"), Check("B", 3.0, 2.0, False)]
    assert _envelope_checks({LONGITUDINAL: _dres(LONGITUDINAL, cs)}) == cs


def test_envelope_keeps_the_worse_ratio_per_check():
    out = {c.name: c for c in _envelope_checks(_pair(
        [Check("A", 1.0, 2.0, True), Check("B", 1.8, 2.0, True)],
        [Check("A", 1.5, 2.0, True), Check("B", 1.2, 2.0, True)]))}
    assert out["A"].demand == 1.5           # transverse worse
    assert out["B"].demand == 1.8           # longitudinal worse


def test_a_failing_check_beats_a_passing_one():
    """Even if the ratio comparison is degenerate — a zero capacity gives inf."""
    out = {c.name: c for c in _envelope_checks(_pair(
        [Check("A", 1.0, 0.0, True)],       # ratio inf but flagged passing
        [Check("A", 1.0, 2.0, False)]))}
    assert not out["A"].passed


def test_direction_independent_checks_collapse_silently():
    """Identical in both ⇒ one entry, and no '[... governs]' noise."""
    out = _envelope_checks(_pair(
        [Check("Shaft flexure", 1.0, 2.0, True, "capacity protected")],
        [Check("Shaft flexure", 1.0, 2.0, True, "capacity protected")]))
    assert len(out) == 1
    assert "governs" not in out[0].note
    assert out[0].note == "capacity protected"


def test_the_governing_direction_is_named_when_they_differ():
    out = _envelope_checks(_pair(
        [Check("Displacement capacity", 1.0, 2.0, True)],
        [Check("Displacement capacity", 1.5, 2.0, True)]))
    assert "[transverse governs]" in out[0].note


def test_ties_resolve_to_longitudinal():
    out = _envelope_checks(_pair(
        [Check("A", 1.0, 2.0, True, "lon")],
        [Check("A", 1.0, 2.0, True, "trn")]))
    assert out[0].note == "lon"


def test_every_check_survives_the_merge():
    out = _envelope_checks(_pair(
        [Check(n, 1.0, 2.0, True) for n in ("A", "B", "C")],
        [Check(n, 1.5, 2.0, True) for n in ("A", "B", "C")]))
    assert sorted(c.name for c in out) == ["A", "B", "C"]


# --- integration: the whole assessment, both directions ---------------------
def test_transverse_changes_only_the_demand_side():
    """Capacity is axisymmetric and mass-free, so only Dd may move."""
    heavy = _assess(weight=1000.0, weight_trans=1400.0)
    assert heavy.two_direction
    lon = heavy.directions[LONGITUDINAL].governing_bound
    trn = heavy.directions[TRANSVERSE].governing_bound
    # capacity identical
    assert lon.delta_y == pytest.approx(trn.delta_y)
    assert lon.delta_c == pytest.approx(trn.delta_c)
    assert lon.fixity_depth == pytest.approx(trn.fixity_depth)
    assert lon.stiffness == pytest.approx(trn.stiffness)
    # demand not
    assert trn.demand.period > lon.demand.period
    assert trn.demand.disp_demand > lon.demand.disp_demand


def test_no_transverse_weight_means_one_direction():
    a = _assess(weight=1000.0, weight_trans=None)
    assert not a.two_direction
    assert list(a.directions) == [LONGITUDINAL]


# --- head moment in the pile solver ----------------------------------------
def test_head_moment_defaults_to_the_free_head_solve():
    """M_head=0 must reproduce the cantilever solve exactly, not approximately."""
    import numpy as np

    from seismic_column.pile_solver import solve_lateral
    from seismic_column.soil import SoilProfile, SoilLayer

    soil = SoilProfile((SoilLayer(thickness=900.0, py_model="elastic_subgrade",
                                  gamma_eff=0.0, k_py=0.5),))
    args = (300.0, 600.0, 2.0e9, 6.0e9, 84.0, 500.0, soil)
    base = solve_lateral(*args, 400.0)
    zero = solve_lateral(*args, 400.0, M_head=0.0)
    assert np.array_equal(base.moment, zero.moment)
    assert np.array_equal(base.y, zero.y)


def test_head_moment_sign_puts_Mp_at_the_interface():
    """The fixed-fixed head condition: V = 2Mp/H with M = +Mp gives Mp there.

    The solver's DOF-1 convention is the opposite of the obvious hand
    derivation — the wrong sign lands 3*Mp at the interface, which is exactly
    the error :func:`frame_seismic._shaft_demand` guards against.
    """
    from seismic_column.pile_solver import solve_lateral
    from seismic_column.soil import SoilProfile, SoilLayer

    soil = SoilProfile((SoilLayer(thickness=900.0, py_model="elastic_subgrade",
                                  gamma_eff=0.0, k_py=0.5),))
    H, Mp = 300.0, 3.0e5
    args = (H, 600.0, 2.0e9, 6.0e9, 84.0, 500.0, soil)
    right = solve_lateral(*args, 2 * Mp / H, M_head=+Mp)
    wrong = solve_lateral(*args, 2 * Mp / H, M_head=-Mp)
    m_right = abs(right.moment[right.ground_index])
    m_wrong = abs(wrong.moment[wrong.ground_index])
    assert m_right == pytest.approx(Mp, rel=0.10)
    assert m_wrong > 2.0 * Mp                  # the 3x error the guard catches
