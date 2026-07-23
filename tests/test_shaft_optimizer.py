"""Optimiser sizing of the capacity-protected Type II shaft."""
import pytest

from seismic_column.io_schema import GlobalConfig, default_dataframe
from seismic_column.batch import run_batch


def _run(df, code="SDC 2.1", variable=("longitudinal", "confinement", "fc")):
    # these tests exercise SHAFT sizing at a fixed column, so exclude the
    # (slow) column-diameter search unless a test asks for it.
    cfg = GlobalConfig(code=code, optimize=True, variable=variable)
    _, results = run_batch(df, cfg)
    return results[0]


def test_optimiser_increases_shaft_longitudinal_when_flexure_short():
    """A weak shaft under a strong column must be re-reinforced by the optimiser
    until shaft flexure passes."""
    df = default_dataframe(1)
    df.loc[0, "Dcol_in"] = 60
    df.loc[0, "n_bars"] = 28
    df.loc[0, "long_bar_no"] = 11
    df.loc[0, "D_shaft_in"] = 84
    df.loc[0, "shaft_n_bars"] = 12       # deliberately light starting shaft
    df.loc[0, "shaft_long_bar_no"] = 8
    df.loc[0, "axial_kip"] = 1000
    df.loc[0, "weight_kip"] = 1000
    rr = _run(df)
    sf = next(c for c in rr.assessment.checks
              if c.name == "Shaft flexure (capacity protection)")
    assert sf.passed
    # the optimiser must have increased the shaft steel above the light start
    assert rr.shaft.section().rho_l > 0.008


def test_shaft_can_reach_high_ratio_before_giving_up():
    """When flexure demands heavy steel, the shaft longitudinal ladder escalates
    to #18 / bundling rather than capping at a single #14 layer (~1.7%)."""
    from seismic_column.optimizer import (
        ColumnDesign, OptimizeSpec, _size_shaft_longitudinal, _plastic_moment)
    shaft = ColumnDesign(D=72, fc=5, cover=6, n_bars=20, long_bar_no=8,
                         spiral_bar_no=6, spiral_spacing=4, fce_floor=5.0)
    spec = OptimizeSpec()
    # a very high flexural demand a single #14 layer (~1.7%) cannot meet
    m_demand = 1.5 * _plastic_moment(
        ColumnDesign(D=72, fc=5, cover=6, n_bars=44, long_bar_no=14,
                     spiral_bar_no=6, spiral_spacing=4, fce_floor=5.0), 3000.0)
    sized = _size_shaft_longitudinal(shaft, m_demand, 3000.0, spec)
    # it escalated past the conventional single-#14 ceiling (to #18 or bundling)
    assert sized.long_bar_no == 18 or sized.long_bundle > 1
    assert sized.section().rho_l > 0.02


def test_default_shaft_stays_conventional_single_layer():
    """The escalation must not turn an easy default into bundled/huge bars."""
    rr = _run(default_dataframe(1))
    assert rr.shaft.long_bundle == 1                 # no needless bundling
    assert rr.shaft.section().rho_l <= 0.04


def test_shaft_longitudinal_respects_max_ratio():
    rr = _run(default_dataframe(1))
    assert rr.shaft.section().rho_l <= 0.04 + 1e-9


def test_shaft_grows_with_column_to_keep_oversize():
    """When the optimiser grows the column, the shaft is enlarged so it stays
    >= column + min oversize (default 24 in) — no 6 in oversize results."""
    from seismic_column.optimizer import (
        ColumnDesign, OptimizeSpec, optimize_column, required_shaft_diameter)
    from seismic_column.geometry import Geometry
    from seismic_column.demand import DesignSpectrum
    from seismic_column.provisions import get_provisions

    sizes = tuple(range(36, 181, 6))
    assert required_shaft_diameter(42, 24, sizes) == 66   # tracks DOWN to col+24
    assert required_shaft_diameter(54, 24, sizes) == 78   # = col + 24
    assert required_shaft_diameter(48, 24, sizes) == 72

    col = ColumnDesign(D=36, fc=5, cover=2, n_bars=16, long_bar_no=9,
                       spiral_bar_no=5, spiral_spacing=4)
    shaft = ColumnDesign(D=60, fc=5, cover=6, n_bars=24, long_bar_no=11,
                         spiral_bar_no=6, spiral_spacing=4)
    res = optimize_column(
        col, shaft, Geometry(Hcol=30 * 12, D_shaft=60),
        DesignSpectrum(Sds=1.2, Sd1=0.7), 1200, 1200,
        spec=OptimizeSpec(variable={"longitudinal", "confinement", "diameter", "fc"},
                          priority=("longitudinal", "confinement", "diameter", "fc")),
        provisions=get_provisions("AASHTO SGS 3rd Ed."))
    # whatever column the optimiser lands on, the shaft is >= 24 in larger
    assert res.shaft.D - res.design.D >= 24


def test_min_oversize_respects_code_floor_under_caltrans():
    """A user oversize below the Caltrans 24 in Type II floor is raised to 24."""
    from seismic_column.optimizer import ColumnDesign, OptimizeSpec, optimize_column
    from seismic_column.geometry import Geometry
    from seismic_column.demand import DesignSpectrum
    from seismic_column.provisions import get_provisions
    col = ColumnDesign(D=36, fc=5, cover=2, n_bars=20, long_bar_no=10,
                       spiral_bar_no=6, spiral_spacing=4, fce_floor=5.0)
    shaft = ColumnDesign(D=60, fc=5, cover=6, n_bars=24, long_bar_no=11,
                         spiral_bar_no=6, spiral_spacing=4, fce_floor=5.0)
    res = optimize_column(
        col, shaft, Geometry(Hcol=24 * 12, D_shaft=60),
        DesignSpectrum(Sds=1.0, Sd1=0.6), 900, 900,
        spec=OptimizeSpec(variable={"diameter"}, priority=("diameter",),
                          min_shaft_oversize=6.0),          # user asks for only 6"
        provisions=get_provisions("SDC 2.1"))
    assert res.shaft.D - res.design.D >= 24                 # code floor wins


def test_objectives_trade_diameter_for_steel():
    """The objectives span the diameter/steel trade-off: min_diameter gives the
    smallest column (heavier steel), min_steel a larger column at the ~1% floor,
    and target_steel a column sized to ~the requested ratio.  Small diameter
    ladder keeps the test fast."""
    from seismic_column.optimizer import ColumnDesign, OptimizeSpec, optimize_column
    from seismic_column.geometry import Geometry
    from seismic_column.demand import DesignSpectrum
    from seismic_column.provisions import get_provisions
    col = ColumnDesign(D=36, fc=5, cover=2, n_bars=12, long_bar_no=9,
                       spiral_bar_no=6, spiral_spacing=4, fce_floor=5.0)
    shaft = ColumnDesign(D=60, fc=5, cover=6, n_bars=20, long_bar_no=11,
                         spiral_bar_no=6, spiral_spacing=4, fce_floor=5.0)
    geom = Geometry(Hcol=22 * 12, D_shaft=60)
    spec = DesignSpectrum(Sds=1.0, Sd1=0.6)

    def run(obj, **kw):
        return optimize_column(
            col, shaft, geom, spec, 900, 900,
            spec=OptimizeSpec(
                variable={"longitudinal", "confinement", "diameter", "fc"},
                objective=obj, diameters=(36, 42, 48, 54, 60), **kw),
            provisions=get_provisions("SDC 2.1"))

    md, ms = run("min_diameter"), run("min_steel")
    assert md.feasible and ms.feasible
    # least-steel uses a larger (or equal) column but less (or equal) steel
    assert ms.design.D >= md.design.D
    assert ms.design.rho_l() <= md.design.rho_l() + 1e-9
    # min_steel lands at/near the 1% floor
    assert ms.design.rho_l() <= 0.0102 or ms.design.D == 60
    # shaft stays >= column + 24 in in every case
    assert md.shaft.D - md.design.D >= 24 and ms.shaft.D - ms.design.D >= 24

    # target_steel holds the cage at ~the requested ratio and finds the smallest
    # working column; steel sits near (>=) the target, between the two extremes
    ts = run("target_steel", target_rho=0.015)
    assert ts.feasible
    assert 0.015 - 1e-6 <= ts.design.rho_l() <= 0.02      # ~1.5%, not min, not max
    assert md.design.D <= ts.design.D <= ms.design.D


def test_fixed_diameter_objective_minimises_steel_at_entered_size():
    """'fixed_diameter' keeps the entered column diameter and returns the
    smallest feasible longitudinal ratio there (no diameter search)."""
    from seismic_column.optimizer import ColumnDesign, OptimizeSpec, optimize_column
    from seismic_column.geometry import Geometry
    from seismic_column.demand import DesignSpectrum
    from seismic_column.provisions import get_provisions
    # start with a heavy cage at a chosen 54 in column; optimiser should keep 54
    # and drop the steel to the minimum that still passes.
    col = ColumnDesign(D=54, fc=5, cover=2, n_bars=40, long_bar_no=11,
                       spiral_bar_no=6, spiral_spacing=4, fce_floor=5.0)
    shaft = ColumnDesign(D=84, fc=5, cover=6, n_bars=24, long_bar_no=11,
                         spiral_bar_no=6, spiral_spacing=4, fce_floor=5.0)
    res = optimize_column(
        col, shaft, Geometry(Hcol=22 * 12, D_shaft=84),
        DesignSpectrum(Sds=1.0, Sd1=0.6), 900, 900,
        spec=OptimizeSpec(
            variable={"longitudinal", "confinement", "diameter", "fc"},
            objective="fixed_diameter"),
        provisions=get_provisions("SDC 2.1"))
    assert res.feasible
    assert res.design.D == 54                         # diameter untouched
    assert res.design.rho_l() < col.rho_l()           # steel reduced to minimum
    assert res.design.rho_l() >= 0.01 - 1e-9          # not below the floor
