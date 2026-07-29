"""Frame demand as the design basis.

The invariant that makes this safe to apply everywhere: for a SINGLE-BENT frame
the frame's (K, W) is that bent's own, so the frame basis reproduces the
stand-alone cantilever exactly.  Only a bent that shares a frame moves.
"""
from __future__ import annotations

import pytest

from seismic_column.demand import DesignSpectrum
from seismic_column.geometry import Geometry
from seismic_column.io_schema import LONGITUDINAL, TRANSVERSE
from seismic_column.sdc_capacity import evaluate_column
from seismic_column.section import CircularSection


def _assess(weight=1200.0, weight_trans=1200.0, basis=None, fixity=None):
    col = CircularSection(D=48, fc=4, cover=2, n_bars=24, long_bar_no=11,
                          spiral_bar_no=6, spiral_spacing=3)
    shaft = CircularSection(D=84, fc=4, cover=3, n_bars=40, long_bar_no=11,
                            spiral_bar_no=6, spiral_spacing=3.5)
    return evaluate_column(col, shaft, Geometry(Hcol=18 * 12, D_shaft=84),
                           DesignSpectrum(Sds=1.0, Sd1=0.6), axial=700,
                           weight=weight, weight_trans=weight_trans,
                           demand_basis=basis, end_fixity=fixity)


def _own(a, direction=LONGITUDINAL):
    """The (k, W) this bent would use on its own, keyed by bound label."""
    d = a.directions[direction]
    return {b.soil_label or f"{b.multiplier:.2g}·D_shaft":
            (b.stiffness, d.weight_mass) for b in d.bounds}


# --- the invariant ----------------------------------------------------------
def test_a_single_bent_frame_basis_reproduces_the_stand_alone_result():
    """Supplying a bent's OWN (k, W) must change nothing at all."""
    plain = _assess()
    basis = {LONGITUDINAL: _own(plain, LONGITUDINAL),
             TRANSVERSE: _own(plain, TRANSVERSE)}
    framed = _assess(basis=basis)
    for d in (LONGITUDINAL, TRANSVERSE):
        for p, f in zip(plain.directions[d].bounds, framed.directions[d].bounds):
            assert f.demand.period == pytest.approx(p.demand.period)
            assert f.demand.disp_demand == pytest.approx(p.demand.disp_demand)
            assert f.delta_y == pytest.approx(p.delta_y)
            assert f.delta_c == pytest.approx(p.delta_c)
            assert f.mu_demand == pytest.approx(p.mu_demand)


def test_no_basis_is_the_stand_alone_behaviour():
    a = _assess(basis=None)
    assert a.directions[LONGITUDINAL].bounds[0].demand.disp_demand > 0


# --- a stiffer frame cuts the demand ---------------------------------------
def test_a_stiffer_frame_shortens_the_period_and_cuts_the_demand():
    plain = _assess()
    own = _own(plain)
    # three identical members acting together: 3x the stiffness, 3x the mass
    frame = {lbl: (3.0 * k, 3.0 * w) for lbl, (k, w) in own.items()}
    a = _assess(basis={LONGITUDINAL: frame})
    p0 = plain.directions[LONGITUDINAL].bounds[0]
    f0 = a.directions[LONGITUDINAL].bounds[0]
    # k and m scale together, so T is unchanged -- the point is that the frame
    # basis is what drives it, not the bent's own numbers
    assert f0.demand.period == pytest.approx(p0.demand.period)
    # now stiffen WITHOUT adding mass: the period must drop
    stiffer = {lbl: (3.0 * k, w) for lbl, (k, w) in own.items()}
    b = _assess(basis={LONGITUDINAL: stiffer})
    s0 = b.directions[LONGITUDINAL].bounds[0]
    assert s0.demand.period < p0.demand.period
    assert s0.demand.disp_demand < p0.demand.disp_demand


def test_the_basis_only_touches_the_direction_it_is_given_for():
    plain = _assess()
    own = _own(plain)
    stiffer = {lbl: (5.0 * k, w) for lbl, (k, w) in own.items()}
    a = _assess(basis={LONGITUDINAL: stiffer})
    assert (a.directions[LONGITUDINAL].bounds[0].demand.period
            < plain.directions[LONGITUDINAL].bounds[0].demand.period)
    assert (a.directions[TRANSVERSE].bounds[0].demand.period
            == pytest.approx(plain.directions[TRANSVERSE].bounds[0].demand.period))


def test_an_unknown_bound_label_falls_back_to_the_bent_itself():
    """A basis that does not cover a bound must not silently mis-index."""
    plain = _assess()
    a = _assess(basis={LONGITUDINAL: {"not-a-real-bound": (9999.0, 1.0)}})
    for p, f in zip(plain.directions[LONGITUDINAL].bounds,
                    a.directions[LONGITUDINAL].bounds):
        assert f.demand.disp_demand == pytest.approx(p.demand.disp_demand)


# --- the end condition ------------------------------------------------------
def test_fixed_end_condition_uses_the_two_hinge_mechanism():
    """Fixed-fixed takes more shear to form a mechanism, over a shorter lever."""
    free = _assess()
    fixed = _assess(fixity={LONGITUDINAL: "fixed"})
    f0 = free.directions[LONGITUDINAL].bounds[0]
    x0 = fixed.directions[LONGITUDINAL].bounds[0]
    assert x0.delta_y != pytest.approx(f0.delta_y)
    assert x0.delta_c < f0.delta_c          # shorter hinge spacing
    # transverse was not named, so it stays fixed-free
    assert (fixed.directions[TRANSVERSE].bounds[0].delta_c
            == pytest.approx(free.directions[TRANSVERSE].bounds[0].delta_c))


def test_capacity_is_unchanged_when_no_end_condition_is_given():
    plain = _assess()
    named = _assess(fixity={LONGITUDINAL: "free", TRANSVERSE: "free"})
    for d in (LONGITUDINAL, TRANSVERSE):
        assert (named.directions[d].bounds[0].delta_c
                == pytest.approx(plain.directions[d].bounds[0].delta_c))


# --- the overstrength shear follows the mechanism, not the cantilever -------
def _named(checks, name):
    return next(c for c in checks if c.name == name)


def test_fixed_end_doubles_the_overstrength_member_shear():
    """Both the COLUMN and the SHAFT carry it -- it is one force, not two."""
    free = _assess()
    fixed = _assess(fixity={LONGITUDINAL: "fixed"})
    f_lon = free.directions[LONGITUDINAL]
    x_lon = fixed.directions[LONGITUDINAL]
    assert x_lon.Vo == pytest.approx(2.0 * f_lon.Vo)
    # the column shear check reads the same Vo the shaft checks do
    assert (_named(x_lon.checks, "Column shear").demand
            == pytest.approx(x_lon.Vo))
    # and the direction that was NOT named keeps the cantilever value
    assert (fixed.directions[TRANSVERSE].Vo
            == pytest.approx(free.directions[TRANSVERSE].Vo))


def test_the_shaft_checks_are_unchanged_when_both_ends_are_free():
    plain = _assess()
    named = _assess(fixity={LONGITUDINAL: "free", TRANSVERSE: "free"})
    for d in (LONGITUDINAL, TRANSVERSE):
        for n in ("Column shear", "Shaft shear (capacity protection)"):
            assert (_named(named.directions[d].checks, n).demand
                    == pytest.approx(_named(plain.directions[d].checks, n).demand))
