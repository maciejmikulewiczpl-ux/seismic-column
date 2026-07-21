"""Short-period magnification Rd — parametric and tabular spectra."""
import pytest

from seismic_column.demand import (
    DesignSpectrum,
    SpectrumSpec,
    TabularSpectrum,
    displacement_demand,
    magnified_demand,
    short_period_magnification,
)


def test_rd_unity_for_long_period():
    # T well above T* = 1.25*Ts -> no magnification
    assert short_period_magnification(T=2.0, Ts=0.6, mu_d=3.0) == pytest.approx(1.0)


def test_rd_unity_when_elastic():
    # mu_d -> 1 kills the magnification regardless of period
    assert short_period_magnification(T=0.3, Ts=0.6, mu_d=1.0) == pytest.approx(1.0)


def test_rd_formula_value():
    # (1 - 1/3)*(0.75/0.3) + 1/3 = 0.6667*2.5 + 0.3333 = 2.0
    assert short_period_magnification(T=0.3, Ts=0.6, mu_d=3.0) == pytest.approx(2.0)


def test_tabular_ts_matches_smooth_1_over_T():
    # a fine 1/T curve (Sa = 0.6/T on the velocity branch) should give Ts≈Sd1/Sds
    periods = tuple(round(0.1 * i, 2) for i in range(1, 61))  # 0.1..6.0 s
    accels = tuple(min(1.0, 0.6 / max(t, 0.6)) for t in periods)  # plateau 1.0, then 0.6/T
    ts = TabularSpectrum(periods, accels)
    # max(Sa)=1.0, max(T*Sa) over 1-5s = 0.6 -> Ts = 0.6
    assert ts.Ts == pytest.approx(0.6, abs=0.02)


def test_tabular_coarse_curve_reads_higher():
    """A coarse curve with linear interpolation bulges above 1/T, so Ts is a bit
    larger (more conservative) than the smooth parametric equivalent."""
    ts = TabularSpectrum((0.0, 0.2, 0.5, 1.0, 2.0, 4.0),
                         (0.40, 1.00, 1.00, 0.60, 0.30, 0.15))
    assert ts.Ts > 0.60           # bulge between 1 and 2 s
    assert ts.Ts == pytest.approx(0.675, abs=0.01)


def test_tabular_magnification_applies():
    """magnified_demand must now magnify a tabular spectrum (previously it
    returned the demand unchanged because Ts was None)."""
    ts = TabularSpectrum((0.0, 0.2, 0.5, 1.0, 2.0, 4.0),
                         (1.2, 3.0, 3.0, 1.8, 0.9, 0.45))
    d = displacement_demand(ts, stiffness=500.0, weight=500.0)  # stiff -> short T
    dm = magnified_demand(d, ts, delta_y=0.7)
    if dm.period < 1.25 * ts.Ts:      # genuinely short-period
        assert dm.Rd > 1.0
        assert dm.disp_demand == pytest.approx(dm.Rd * dm.disp_elastic)
        assert dm.Ts == pytest.approx(ts.Ts)


def test_magnified_demand_stores_derivation_fields():
    ds = DesignSpectrum(Sds=3.0, Sd1=1.8)  # Ts = 0.6
    d = displacement_demand(ds, stiffness=600.0, weight=500.0)
    dm = magnified_demand(d, ds, delta_y=0.6)
    if dm.Rd > 1.0:
        assert dm.Ts == pytest.approx(0.6)
        assert dm.mu_for_Rd == pytest.approx(dm.disp_demand / 0.6, rel=1e-6)


def test_tabular_curve_below_1s_gives_no_corner():
    # a curve that ends before 1 s can't define SD1 -> Ts = 0
    ts = TabularSpectrum((0.0, 0.3, 0.6), (1.0, 0.8, 0.5))
    assert ts.Ts == 0.0
