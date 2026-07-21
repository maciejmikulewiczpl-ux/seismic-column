"""Optimiser sizing of the capacity-protected Type II shaft."""
import pytest

from seismic_column.io_schema import GlobalConfig, default_dataframe
from seismic_column.batch import run_batch


def _run(df, code="SDC 2.1"):
    _, results = run_batch(df, GlobalConfig(code=code, optimize=True))
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
    """When flexure demands heavy steel, the shaft ladder escalates to #18 /
    bundling rather than capping at a single #14 layer (~1.7%)."""
    df = default_dataframe(1)
    df.loc[0, "Dcol_in"] = 72
    df.loc[0, "n_bars"] = 40
    df.loc[0, "long_bar_no"] = 14
    df.loc[0, "D_shaft_in"] = 78          # only 6 in oversize (AASHTO allows)
    df.loc[0, "axial_kip"] = 2000
    df.loc[0, "weight_kip"] = 2000
    df.loc[0, "Hcol_ft"] = 14
    rr = _run(df, code="AASHTO SGS 3rd Ed.")
    sf = next(c for c in rr.assessment.checks
              if c.name == "Shaft flexure (capacity protection)")
    # it should now pass by pushing shaft steel well past the old ~1.7% ceiling
    assert sf.passed
    assert rr.shaft.section().rho_l > 0.02


def test_default_shaft_stays_conventional_single_layer():
    """The escalation must not turn an easy default into bundled/huge bars."""
    rr = _run(default_dataframe(1))
    assert rr.shaft.long_bundle == 1                 # no needless bundling
    assert rr.shaft.section().rho_l <= 0.04


def test_shaft_longitudinal_respects_max_ratio():
    rr = _run(default_dataframe(1))
    assert rr.shaft.section().rho_l <= 0.04 + 1e-9
