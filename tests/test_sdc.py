from seismic_column.demand import (
    DesignSpectrum,
    SpectrumSpec,
    TabularSpectrum,
    displacement_demand,
)
from seismic_column.geometry import Geometry
from seismic_column.sdc_capacity import (
    evaluate_column,
    min_transverse_ratio,
    plastic_hinge_length,
)
from seismic_column.section import CircularSection


def test_plastic_hinge_length_lower_bound():
    fye, dbl = 68.0, 1.27
    lp = plastic_hinge_length(Hcol=10.0, fye=fye, dbl=dbl)  # short -> min governs
    assert abs(lp - 0.3 * fye * dbl) < 1e-9
    lp2 = plastic_hinge_length(Hcol=400.0, fye=fye, dbl=dbl)
    assert lp2 > 0.3 * fye * dbl


def test_min_transverse_ratio_positive():
    assert min_transverse_ratio(4.0, 68.0, 1809.6, 1385.4) > 0


def test_spectrum_shape():
    spec = DesignSpectrum(Sds=1.0, Sd1=0.6)
    assert abs(spec.Sa(spec.T0) - 1.0) < 1e-9   # plateau start
    assert abs(spec.Sa(spec.Ts) - 1.0) < 1e-9   # plateau end
    assert spec.Sa(2.0) < spec.Sa(1.0)          # descending branch
    assert abs(spec.Sa(spec.Ts + 1e-9) - spec.Sd1 / spec.Ts) < 1e-6


def test_displacement_demand_positive():
    spec = DesignSpectrum(Sds=1.0, Sd1=0.6)
    d = displacement_demand(spec, stiffness=50.0, weight=800.0)
    assert d.period > 0 and d.disp_demand > 0


def test_evaluate_column_returns_full_checks():
    col = CircularSection(D=48, fc=4, cover=2, n_bars=24, long_bar_no=11,
                          spiral_bar_no=6, spiral_spacing=3)
    shaft = CircularSection(D=84, fc=4, cover=3, n_bars=40, long_bar_no=11,
                            spiral_bar_no=6, spiral_spacing=3.5)
    geom = Geometry(Hcol=18 * 12, D_shaft=84)
    spec = DesignSpectrum(Sds=1.0, Sd1=0.6)
    a = evaluate_column(col, shaft, geom, spec, axial=700, weight=700)
    names = {c.name for c in a.checks}
    assert "Displacement capacity" in names
    assert "Shaft flexure (capacity protection)" in names
    assert "Shaft shear (capacity protection)" in names
    assert len(a.bounds) == 2
    assert a.governing_bound in a.bounds


def test_tabular_spectrum_interpolation():
    ts = TabularSpectrum(periods=(0.0, 1.0, 2.0), accels=(1.0, 0.5, 0.25))
    assert abs(ts.Sa(0.5) - 0.75) < 1e-9      # midpoint interpolation
    assert ts.Sa(-1.0) == 1.0                 # clamp low
    assert ts.Sa(5.0) == 0.25                 # clamp high
    built = SpectrumSpec(kind="tabular", periods=(2.0, 0.0, 1.0),
                         accels=(0.25, 1.0, 0.5)).build()
    assert isinstance(built, TabularSpectrum)
    assert built.periods == (0.0, 1.0, 2.0)   # sorted on build


def test_low_level_earthquake_elastic_check():
    col = CircularSection(D=48, fc=4, cover=2, n_bars=24, long_bar_no=11,
                          spiral_bar_no=6, spiral_spacing=3)
    shaft = CircularSection(D=84, fc=4, cover=3, n_bars=40, long_bar_no=11,
                            spiral_bar_no=6, spiral_spacing=3.5)
    geom = Geometry(Hcol=18 * 12, D_shaft=84)
    design = DesignSpectrum(Sds=1.0, Sd1=0.6)
    small = DesignSpectrum(Sds=0.05, Sd1=0.03)   # tiny LLE -> elastic
    big = DesignSpectrum(Sds=3.0, Sd1=2.0)        # huge LLE -> yields
    a_ok = evaluate_column(col, shaft, geom, design, 700, 700,
                           lle_spectrum=small, lle_mu_limit=1.0)
    a_bad = evaluate_column(col, shaft, geom, design, 700, 700,
                            lle_spectrum=big, lle_mu_limit=1.0)
    lle_ok = next(c for c in a_ok.checks if "Low-level" in c.name)
    lle_bad = next(c for c in a_bad.checks if "Low-level" in c.name)
    assert lle_ok.passed
    assert not lle_bad.passed
