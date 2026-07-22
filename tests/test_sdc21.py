"""Rigorous tests for the Caltrans SDC 2.1 provisions added in the audit.

Covers the pieces that are easy to get wrong and that the spec pins down
numerically: the Table 5.3.8.2-1 min-transverse lookup and its boundaries, the
concrete-shear fs cap and net-tension rule (both codes), the axial-load ratio,
the detailing spacing rules, and a smoke test of every code / optimise path.
"""
import math

import pytest

from seismic_column.demand import DesignSpectrum
from seismic_column.geometry import Geometry
from seismic_column.io_schema import GlobalConfig, default_dataframe, validate
from seismic_column.provisions import AASHTO_SGS_3, SDC_2_1, get_provisions
from seismic_column.section import CircularSection
from seismic_column.batch import run_batch
from seismic_column.sdc_capacity import (
    axial_load_ratio,
    caltrans_min_transverse_ratio,
    concrete_shear_stress,
    evaluate_column,
    max_longitudinal_spacing_caltrans,
    max_transverse_spacing_caltrans,
)


# ---------------------------------------------------------------------------
# Table 5.3.8.2-1 — minimum transverse reinforcement lookup
# ---------------------------------------------------------------------------
def _table(D_ft, L_over_Dc, rho_l_pct, rho_dl_pct):
    """Helper: drive caltrans_min_transverse_ratio via engineered inputs.

    Chooses P and Ag so that ρdl and ρl come out at the requested percentages.
    """
    D = D_ft * 12.0
    Hcol = L_over_Dc * D
    Ag = math.pi * D ** 2 / 4.0
    fc = 4.0
    rho_l = rho_l_pct / 100.0
    # rho_dl = P / (min(fc,5)*Ag)  ->  P = rho_dl * fc * Ag
    P = (rho_dl_pct / 100.0) * min(fc, 5.0) * Ag
    return caltrans_min_transverse_ratio(D, Hcol, rho_l, P, fc, Ag)


@pytest.mark.parametrize("D_ft,rho_dl_pct,expected", [
    (4.0, 5.0, 0.006),    # 3–6 ft, ρdl ≤ 10
    (4.0, 10.0, 0.006),   # boundary: exactly 10% stays in the low band
    (4.0, 12.0, 0.007),   # 3–6 ft, 10 < ρdl ≤ 15
    (6.0, 9.0, 0.006),    # Dc = 6 ft is still the small-diameter band
    (8.0, 5.0, 0.007),    # 6–11 ft, ρdl ≤ 10
    (8.0, 13.0, 0.008),   # 6–11 ft, 10 < ρdl ≤ 15
    (11.0, 5.0, 0.007),   # upper diameter boundary
])
def test_table_5382_values(D_ft, rho_dl_pct, expected):
    rho_l_pct = 2.0  # within both ρl limits (2.3 / 2.15)
    rho_s_min, in_table, _ = _table(D_ft, 5.0, rho_l_pct, rho_dl_pct)
    assert in_table
    assert rho_s_min == pytest.approx(expected)


def test_table_5382_aspect_ratio_out_of_range():
    _, in_table, note = _table(4.0, 9.0, 2.0, 5.0)   # L/Dc = 9 > 8
    assert not in_table and "L/Dc" in note


def test_table_5382_diameter_out_of_range():
    for D_ft in (2.5, 12.0):
        _, in_table, note = _table(D_ft, 5.0, 2.0, 5.0)
        assert not in_table and "Dc" in note


def test_table_5382_rho_l_out_of_range():
    # ρl = 2.5% exceeds the 2.3% limit for the 3–6 ft band.
    _, in_table, note = _table(4.0, 5.0, 2.5, 5.0)
    assert not in_table and "ρl" in note


def test_table_5382_axial_out_of_range():
    _, in_table, note = _table(4.0, 5.0, 2.0, 16.0)   # ρdl = 16% > 15%
    assert not in_table and "ρdl" in note


def test_out_of_table_check_is_not_certified():
    """A section outside the table must FAIL the check, never silently pass."""
    # Very tall column: L/Dc > 8 pushes it out of the table.
    col = CircularSection(D=48, fc=4, cover=2, n_bars=20, long_bar_no=10,
                          spiral_bar_no=8, spiral_spacing=2)  # huge ρs
    shaft = CircularSection(D=84, fc=4, cover=3, n_bars=40, long_bar_no=11,
                            spiral_bar_no=6, spiral_spacing=3.5)
    geom = Geometry(Hcol=40 * 12, D_shaft=84)   # L/Dc = 10
    spec = DesignSpectrum(Sds=1.0, Sd1=0.6)
    a = evaluate_column(col, shaft, geom, spec, axial=500, weight=500,
                        provisions=SDC_2_1)
    ck = next(c for c in a.checks if c.name == "Transverse steel ratio (min)")
    assert not ck.passed
    assert "PSDC" in ck.note


# ---------------------------------------------------------------------------
# Axial load ratio ρdl = Pdl / (f'c·Ag), f'c capped at 5 ksi (§5.3.3)
# ---------------------------------------------------------------------------
def test_axial_load_ratio_fc_cap():
    Ag = 1000.0
    # f'c = 4 ksi: no cap.
    assert axial_load_ratio(400.0, 4.0, Ag) == pytest.approx(0.1)
    # f'c = 8 ksi: capped at 5 ksi in the denominator.
    assert axial_load_ratio(400.0, 8.0, Ag) == pytest.approx(400.0 / (5.0 * Ag))


# ---------------------------------------------------------------------------
# Concrete shear — fs cap and net-tension rule now apply to BOTH codes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["caltrans", "aashto"])
def test_net_tension_gives_zero_vc(model):
    vc = concrete_shear_stress(fc=4.0, rho_s=0.01, fyh=68.0, P=-50.0,
                               Ag=1800.0, mu_d=1.0, inside_hinge=True,
                               model=model)
    assert vc == 0.0


def test_fs_cap_actually_reduces_alpha():
    """The fs ≤ 0.35 cap must bite for heavy confinement.

    rho_s*fyh = 0.02*68 = 1.36 ksi (>> 0.35).  At mu_d = 4.0, capping fs at
    0.35 gives alpha' = 0.35/0.15 + 3.67 - 4.0 = 2.003, whereas the uncapped
    fs would clamp alpha' to 3.0.  So the capped result must be well below the
    uncapped-clamped result, and neither vc cap binds at alpha' = 2.0.
    """
    kw = dict(fc=4.0, rho_s=0.02, fyh=68.0, P=800.0, Ag=1800.0, mu_d=4.0,
              inside_hinge=True)
    # Hand calc: F2 = 1 + 800000/(2000*1800) = 1.2222, sqrt(4000) = 63.246 psi.
    vc_ct = concrete_shear_stress(model="caltrans", **kw)
    vc_aa = concrete_shear_stress(model="aashto", **kw)
    assert vc_ct == pytest.approx(0.15487, rel=1e-3)   # capped -> alpha 2.003
    assert vc_aa == pytest.approx(0.15672, rel=1e-3)
    # Uncapped fs would clamp alpha to 3.0 -> Caltrans vc = 0.2319; confirm we
    # are NOT getting that (i.e. the cap really changed the answer).
    assert vc_ct < 0.20
    # With the cap active and both below their vc caps, the two codes agree.
    assert vc_ct == pytest.approx(vc_aa, rel=0.03)


def test_vc_caps_differ_between_codes():
    """Where alpha' hits 3.0, AASHTO's 0.11√f'c cap is stricter than
    Caltrans' 4√f'c cap — the models are NOT identical there."""
    kw = dict(fc=4.0, rho_s=0.02, fyh=68.0, P=800.0, Ag=1800.0, mu_d=1.5,
              inside_hinge=True)
    vc_ct = concrete_shear_stress(model="caltrans", **kw)
    vc_aa = concrete_shear_stress(model="aashto", **kw)
    assert vc_ct == pytest.approx(0.23190, rel=1e-3)   # 4√f'c not yet binding
    assert vc_aa == pytest.approx(0.22, rel=1e-3)       # 0.11√f'c binds
    assert vc_aa < vc_ct


# ---------------------------------------------------------------------------
# Detailing spacing rules (§8.4.1.1 / §8.4.2)
# ---------------------------------------------------------------------------
def test_caltrans_tie_spacing():
    # dbl = 1.0 in (#8): min(6*1.0, 8) = 6.0
    s, gov = max_transverse_spacing_caltrans(1.0)
    assert s == pytest.approx(6.0) and "6·dbl" in gov
    # dbl = 1.693 (#14): min(10.16, 8) = 8.0
    s, gov = max_transverse_spacing_caltrans(1.693)
    assert s == pytest.approx(8.0) and "8" in gov


def test_caltrans_longitudinal_spacing_diameter_threshold():
    assert max_longitudinal_spacing_caltrans(60.0)[0] == pytest.approx(10.0)   # 5 ft
    assert max_longitudinal_spacing_caltrans(60.1)[0] == pytest.approx(12.0)   # >5 ft


# ---------------------------------------------------------------------------
# Provisions wiring
# ---------------------------------------------------------------------------
def test_sdc20_key_alias_resolves_to_21():
    assert get_provisions("SDC 2.0") is SDC_2_1
    assert get_provisions("SDC 2.1") is SDC_2_1


def test_caltrans_has_no_shaft_confinement_fraction():
    # §5.3.8.4/5.3.8.5 give no §8.8.12-style rule for a Type II shaft.
    assert SDC_2_1.shaft_confinement_fraction is None
    assert AASHTO_SGS_3.shaft_confinement_fraction == 0.5


def test_caltrans_fce_floor_only():
    assert SDC_2_1.fce_floor == 5.0
    assert AASHTO_SGS_3.fce_floor is None


# ---------------------------------------------------------------------------
# Type II oversize validation (Caltrans needs 24 in, AASHTO only "larger")
# ---------------------------------------------------------------------------
def test_oversize_validation_by_code():
    df = default_dataframe(1)
    df.loc[0, "Dcol_in"] = 70.0   # shaft 84 -> only 14 in oversize
    validate(df, 0.0)             # AASHTO: accepts
    with pytest.raises(ValueError, match="24 in larger"):
        validate(df, 24.0)        # Caltrans: rejects


# ---------------------------------------------------------------------------
# End-to-end smoke test: every code / optimise combination must run clean
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("code", ["SDC 2.1", "AASHTO SGS 3rd Ed."])
@pytest.mark.parametrize("optimize", [False, True])
def test_batch_runs_without_error(code, optimize):
    # exclude the (slow) column-diameter search — this smoke test checks the
    # batch runs clean and reaches feasible, not the diameter objective.
    cfg = GlobalConfig(code=code, optimize=optimize,
                       variable=("longitudinal", "confinement", "fc"))
    summary, results = run_batch(default_dataframe(3), cfg)
    assert len(results) == 3
    assert not any(str(s).startswith("ERROR") for s in summary["status"])
    if optimize:
        # the greedy search should reach a feasible design for the defaults
        assert all(r.feasible for r in results)
