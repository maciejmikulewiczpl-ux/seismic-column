import pandas as pd

from seismic_column.batch import run_batch
from seismic_column.io_schema import (
    GlobalConfig,
    default_dataframe,
    project_from_json,
    project_to_json,
    read_table,
    validate,
    write_table,
)
from seismic_column.demand import SpectrumSpec
from seismic_column.optimizer import ColumnDesign, OptimizeSpec, optimize_column
from seismic_column.geometry import Geometry
from seismic_column.demand import DesignSpectrum
from seismic_column.report import column_report
from seismic_column.sdc_capacity import evaluate_column


def test_default_dataframe_valid():
    df = validate(default_dataframe(3))
    assert len(df) == 3
    assert "Hcol_ft" in df.columns


def test_roundtrip_csv(tmp_path):
    df = default_dataframe(2)
    p = tmp_path / "cols.csv"
    write_table(df, p)
    df2 = read_table(p)
    assert len(df2) == 2


def test_run_batch_optimize():
    df = default_dataframe(2)
    # fix the column diameter (exclude the slow diameter search) — this test is
    # about the batch optimise loop producing feasible designs, not sizing.
    cfg = GlobalConfig(optimize=True,
                       variable=("longitudinal", "confinement", "fc"))
    summary, results = run_batch(df, cfg)
    assert len(summary) == 2
    assert len(results) == 2
    for r in results:
        assert r.feasible  # starter batch should be solvable


def test_optimizer_reaches_feasible():
    start = ColumnDesign(D=48, fc=4, cover=2, n_bars=12, long_bar_no=8,
                         spiral_bar_no=4, spiral_spacing=5)
    shaft = ColumnDesign(D=84, fc=4, cover=3, n_bars=36, long_bar_no=11,
                         spiral_bar_no=6, spiral_spacing=4)
    geom = Geometry(Hcol=20 * 12, D_shaft=84)
    spec = DesignSpectrum(Sds=1.0, Sd1=0.6)
    # this case needs the column to grow — small diameter ladder + min_diameter
    # keeps the search fast while exercising the diameter objective.
    res = optimize_column(start, shaft, geom, spec, axial=800, weight=800,
                          spec=OptimizeSpec(
                              variable={"longitudinal", "confinement",
                                        "diameter", "fc"},
                              objective="min_diameter",
                              diameters=(48, 54, 60, 66)))
    assert res.feasible
    assert res.assessment.passed


def test_report_renders():
    df = default_dataframe(1)
    _, results = run_batch(df, GlobalConfig(optimize=False))   # render, don't optimise
    md = column_report(results[0])
    assert "Seismic Column Report" in md
    assert "SDC checks" in md


def test_project_roundtrip():
    df = default_dataframe(2)
    cfg = GlobalConfig(
        design_spectrum=SpectrumSpec(kind="tabular", periods=(0.0, 1.0, 3.0),
                                     accels=(1.0, 0.6, 0.2)),
        lle_spectrum=SpectrumSpec(kind="parametric", Sds=0.4, Sd1=0.24),
        lle_mu_limit=1.0, optimize=False,
    )
    text = project_to_json(df, cfg)
    df2, cfg2 = project_from_json(text)
    assert len(df2) == 2
    assert cfg2.design_spectrum.kind == "tabular"
    assert cfg2.design_spectrum.periods == (0.0, 1.0, 3.0)
    assert cfg2.lle_spectrum is not None
    assert abs(cfg2.lle_spectrum.Sds - 0.4) < 1e-9
    assert cfg2.optimize is False


def test_run_batch_progress_callback():
    """The optional progress callback fires once per column with a growing
    done-count reaching (total, total) and a status label."""
    calls = []
    run_batch(default_dataframe(3), GlobalConfig(optimize=False),
              progress=lambda d, t, name, status: calls.append((d, t, name, status)))
    assert len(calls) == 3
    assert [c[0] for c in calls] == [1, 2, 3]          # monotonic done-count
    assert all(c[1] == 3 for c in calls)                # total
    assert all(c[3] in ("PASS", "FAIL", "ERROR") for c in calls)


def test_run_batch_without_callback_unchanged():
    summary, results = run_batch(default_dataframe(2), GlobalConfig(optimize=False))
    assert len(summary) == 2 and len(results) == 2


def test_fractional_spacing_writeback_when_input_is_integer():
    """Regression: a batch table with all-integer spiral spacings makes pandas
    infer int64; writing a fractional optimiser result (3.5 in pitch) back used
    to raise "Invalid value '3.5' for dtype 'int64'".  validate() must pin the
    non-integer design columns to float so results_to_dataframe stays safe."""
    from seismic_column.batch import RowResult, results_to_dataframe

    df = default_dataframe(1)
    df.loc[0, ["spiral_spacing_in", "shaft_spiral_spacing_in", "fc_ksi"]] = [4, 4, 5]
    v = validate(df)
    assert v["spiral_spacing_in"].dtype == "float64"
    assert v["shaft_spiral_spacing_in"].dtype == "float64"

    cd = ColumnDesign(D=48, fc=5, cover=2, n_bars=24, long_bar_no=11,
                      spiral_bar_no=6, spiral_spacing=3.5)
    sd = ColumnDesign(D=84, fc=5, cover=6, n_bars=40, long_bar_no=11,
                      spiral_bar_no=6, spiral_spacing=3.5)
    a = evaluate_column(cd.section(), sd.section(),
                        Geometry(Hcol=20 * 12, D_shaft=84),
                        DesignSpectrum(Sds=1.0, Sd1=0.6), 800, 800)
    rr = RowResult(str(v.loc[0, "name"]), cd, sd, a, a.passed, True, [])
    out = results_to_dataframe([rr], v)          # must not raise
    assert out.loc[0, "spiral_spacing_in"] == 3.5
    assert out.loc[0, "shaft_spiral_spacing_in"] == 3.5


def test_blank_reinforcement_allowed_when_optimising_but_not_checking():
    """Optimise runs may leave the rebar/f'c blank (the optimiser sizes them);
    check runs still require them."""
    import numpy as np
    from seismic_column.io_schema import OPTIMIZED_COLUMNS, validate
    df = default_dataframe(1)
    for col in OPTIMIZED_COLUMNS:
        df.loc[0, col] = np.nan
    # optimise: blanks filled, runs, produces a feasible design
    _, results = run_batch(df, GlobalConfig(optimize=True,
                                            variable=("longitudinal", "confinement", "fc")))
    assert results and results[0].feasible
    # check: blank reinforcement is an error
    import pytest
    with pytest.raises(ValueError, match="missing values"):
        run_batch(df, GlobalConfig(optimize=False))
    # a required (non-reinforcement) blank still errors even when optimising
    df2 = default_dataframe(1)
    df2.loc[0, "weight_kip"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        run_batch(df2, GlobalConfig(optimize=True))
