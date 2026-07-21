"""Project save/open round-trip, including optimised designs and bundles."""
import pandas as pd
import pytest

from seismic_column.batch import _row_to_inputs, results_to_dataframe, run_batch
from seismic_column.io_schema import (
    COLUMNS,
    GlobalConfig,
    default_dataframe,
    load_project,
    project_from_json,
    save_project,
    validate,
)
from seismic_column.materials import bar_area


def test_bundle_columns_present_in_schema():
    for col in ("long_bundle", "spiral_bundle",
                "shaft_long_bundle", "shaft_spiral_bundle"):
        assert col in COLUMNS


def test_bundles_flow_into_section():
    df = validate(default_dataframe(1))
    df.loc[0, "long_bundle"] = 2
    df.loc[0, "spiral_bundle"] = 2
    df.loc[0, "shaft_spiral_bundle"] = 2
    col, shaft = _row_to_inputs(df.iloc[0], GlobalConfig())[1:3]
    assert col.long_bundle == 2 and col.spiral_bundle == 2
    assert shaft.spiral_bundle == 2
    # Ast doubles for a 2-bar longitudinal bundle.
    expected = col.n_bars * 2 * bar_area(col.long_bar_no)
    assert col.section().Ast == pytest.approx(expected)


def test_results_to_dataframe_captures_optimised_design():
    cfg = GlobalConfig(code="SDC 2.1", optimize=True)
    df0 = default_dataframe(3)
    _, results = run_batch(df0, cfg)
    df_opt = results_to_dataframe(results, df0)
    assert list(df_opt.columns) == list(COLUMNS)
    # every optimised row's geometry matches its result design
    by_name = {r.name: r for r in results}
    for _, row in df_opt.iterrows():
        rr = by_name[row["name"]]
        assert int(row["n_bars"]) == rr.design.n_bars
        assert int(row["long_bar_no"]) == rr.design.long_bar_no
        assert float(row["spiral_spacing_in"]) == pytest.approx(rr.design.spiral_spacing)
        assert int(row["shaft_spiral_bar_no"]) == rr.shaft.spiral_bar_no
        assert int(row["spiral_bundle"]) == rr.design.spiral_bundle


def test_project_roundtrip_preserves_optimised_designs(tmp_path):
    cfg = GlobalConfig(code="SDC 2.1", optimize=True, allow_bundling=True)
    df0 = default_dataframe(3)
    _, results = run_batch(df0, cfg)
    df_opt = results_to_dataframe(results, df0)

    path = tmp_path / "proj.json"
    save_project(path, df_opt, cfg)
    df_re, cfg_re = load_project(path)

    assert df_re.reset_index(drop=True).equals(df_opt.reset_index(drop=True))
    assert cfg_re.code == cfg.code
    # the reopened optimised designs still pass as-entered checks
    _, res2 = run_batch(df_re, GlobalConfig(code="SDC 2.1", optimize=False))
    assert all(r.feasible for r in res2)


def test_old_project_without_bundle_columns_loads():
    """A project JSON predating the bundle columns must still open (default 1)."""
    old = {
        "version": 1, "config": {},
        "columns": [{
            "name": "C1", "Hcol_ft": 22, "D_shaft_in": 84, "weight_kip": 800,
            "axial_kip": 800, "Dcol_in": 48, "fc_ksi": 4, "cover_in": 2,
            "n_bars": 16, "long_bar_no": 9, "spiral_bar_no": 5,
            "spiral_spacing_in": 4, "mult_lb": 3, "mult_ub": 6,
            "shaft_fc_ksi": 4, "shaft_cover_in": 3, "shaft_n_bars": 36,
            "shaft_long_bar_no": 11, "shaft_spiral_bar_no": 6,
            "shaft_spiral_spacing_in": 4,
        }],
    }
    import json
    df, _ = project_from_json(json.dumps(old))
    assert int(df.loc[0, "long_bundle"]) == 1
    assert int(df.loc[0, "spiral_bundle"]) == 1
    assert int(df.loc[0, "shaft_spiral_bundle"]) == 1
