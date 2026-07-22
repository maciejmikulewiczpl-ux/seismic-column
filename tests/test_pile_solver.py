"""FE lateral-pile solver: closed-form validation, coupling, Df_eq inversion."""
import numpy as np
import pytest

from seismic_column.geometry import Geometry
from seismic_column.pile_solver import equivalent_fixity_depth, solve_lateral
from seismic_column.soil import SoilLayer, SoilProfile


def _elastic(ks, L, D=48.0):
    return SoilProfile((SoilLayer(thickness=L, py_model="elastic_subgrade",
                                  gamma_eff=0.0, k_py=ks),))


def _clay(D=84.0):
    return SoilProfile((SoilLayer.from_engineering(
        80, "matlock_soft_clay", 120, su_top_ksf=1.5, eps50=0.01,
        submerged=True),))


def _sand():
    return SoilProfile((SoilLayer.from_engineering(
        80, "api_sand", 120, phi_deg=36, k_pci=90, submerged=True),))


# ---------------------------------------------------------------------------
# validation against the beam-on-elastic-foundation closed form
# ---------------------------------------------------------------------------
def test_boef_head_deflection_matches_closed_form():
    """Semi-infinite pile, free end, point load H: y0 = 2*H*beta/k."""
    EI, ks, D, H = 3.0e9, 2.0, 48.0, 50.0
    beta = (ks / (4 * EI)) ** 0.25
    L = 12.0 / beta                         # long -> semi-infinite
    sol = solve_lateral(0.0, L, EI, EI, D, 0.0, _elastic(ks, L), H,
                        max_nodes=401)
    y0 = 2 * H * beta / ks
    assert sol.converged
    assert sol.head_deflection == pytest.approx(y0, rel=0.01)


def test_boef_max_moment_matches_closed_form():
    EI, ks, D, H = 3.0e9, 2.0, 48.0, 50.0
    beta = (ks / (4 * EI)) ** 0.25
    L = 12.0 / beta
    sol = solve_lateral(0.0, L, EI, EI, D, 0.0, _elastic(ks, L), H,
                        max_nodes=401)
    m_max = 0.3224 * H / beta               # BoEF free-end point load
    assert sol.max_inground_moment == pytest.approx(m_max, rel=0.03)


# ---------------------------------------------------------------------------
# the column overturning moment loads the shaft (the physics point)
# ---------------------------------------------------------------------------
def test_column_moment_deflects_the_pile():
    """A taller column delivers more moment (V*Hcol) to the shaft head, so both
    the head deflection AND the equivalent shaft fixity depth grow with Hcol —
    a pure-shear model would leave Df_eq unchanged."""
    prof = _clay()
    depths, defl = [], []
    for Hcol_ft in (0.5, 15.0, 30.0):
        s = solve_lateral(Hcol_ft * 12, 80 * 12, 2.5e9, 5.0e9, 84.0, 800.0,
                          prof, 150.0)
        depths.append(s.Df_eq)
        defl.append(s.head_deflection)
    assert defl[0] < defl[1] < defl[2]      # more moment -> more deflection
    assert depths[0] < depths[2]            # shaft fixity deepens with moment


# ---------------------------------------------------------------------------
# Df_eq inversion round-trips the two-segment cantilever
# ---------------------------------------------------------------------------
def test_df_eq_round_trips_geometry():
    EI_col, EI_shaft, D, Hcol = 2.5e9, 5.0e9, 84.0, 20 * 12
    sol = solve_lateral(Hcol, 80 * 12, EI_col, EI_shaft, D, 800.0, _clay(),
                        150.0)
    g = Geometry(Hcol=Hcol, D_shaft=D)
    f_cantilever = g.tip_flexibility(EI_col, EI_shaft, sol.Df_eq / D)
    assert f_cantilever == pytest.approx(sol.head_flexibility, rel=1e-4)


def test_df_eq_zero_when_soil_stiffer_than_column():
    # column flexure alone exceeds a tiny f_soil -> Df_eq clamps to 0
    f_col = (240.0 ** 3) / (3.0 * 2.5e9)
    assert equivalent_fixity_depth(0.5 * f_col, 240.0, 2.5e9, 5.0e9) == 0.0


def test_df_eq_monotonic_in_flexibility():
    f_col = (240.0 ** 3) / (3.0 * 2.5e9)
    d1 = equivalent_fixity_depth(2 * f_col, 240.0, 2.5e9, 5.0e9)
    d2 = equivalent_fixity_depth(5 * f_col, 240.0, 2.5e9, 5.0e9)
    assert 0 < d1 < d2


# ---------------------------------------------------------------------------
# physical sensitivity + convergence
# ---------------------------------------------------------------------------
def test_stiffer_soil_reduces_deflection_and_fixity():
    stiff = SoilProfile((SoilLayer.from_engineering(
        80, "api_sand", 130, phi_deg=40, k_pci=200, submerged=True),))
    soft = SoilProfile((SoilLayer.from_engineering(
        80, "matlock_soft_clay", 110, su_top_ksf=0.5, eps50=0.02,
        submerged=True),))
    a = solve_lateral(20 * 12, 80 * 12, 2.5e9, 5.0e9, 84.0, 800.0, stiff, 150.0)
    b = solve_lateral(20 * 12, 80 * 12, 2.5e9, 5.0e9, 84.0, 800.0, soft, 150.0)
    assert a.head_deflection < b.head_deflection
    assert a.Df_eq < b.Df_eq


@pytest.mark.parametrize("prof", [_clay(), _sand()])
def test_nonlinear_converges(prof):
    sol = solve_lateral(20 * 12, 80 * 12, 2.5e9, 5.0e9, 84.0, 800.0, prof,
                        150.0)
    assert sol.converged
    assert sol.head_deflection > 0
    assert sol.max_inground_moment > 0
    # tip of a long pile is essentially undeflected
    assert abs(sol.y[-1]) < 0.01 * sol.head_deflection


def test_pdelta_softens_response():
    """Compressive axial load reduces lateral stiffness (P-Δ)."""
    prof = _sand()
    no_p = solve_lateral(20 * 12, 80 * 12, 2.5e9, 5.0e9, 84.0, 0.0, prof, 150.0)
    with_p = solve_lateral(20 * 12, 80 * 12, 2.5e9, 5.0e9, 84.0, 1500.0, prof,
                           150.0)
    assert with_p.head_deflection > no_p.head_deflection


def test_diagram_endpoints_are_exact_at_zero_axial():
    """With a node forced at the ground line and no axial load, the interface
    moment equals V·Hcol and the shear at the top equals V_head exactly."""
    prof = _sand()
    V, Hcol = 150.0, 20 * 12.0
    s = solve_lateral(Hcol, 80 * 12, 2.5e9, 5.0e9, 84.0, 0.0, prof, V)
    gi = s.ground_index
    assert s.x[gi] == pytest.approx(Hcol)                 # node exactly at ground
    assert s.moment[gi] / 12.0 == pytest.approx(V * Hcol / 12.0, rel=1e-3)
    assert s.shear[0] == pytest.approx(V, rel=1e-3)
    assert abs(s.moment[0]) < 1e-3 * V * Hcol             # free top, M≈0
    assert s.max_inground_shear > 0


def test_pdelta_amplifies_interface_moment():
    """Axial load through the lateral deflection amplifies the interface moment
    above the first-order V·Hcol."""
    prof = _sand()
    V, Hcol = 150.0, 20 * 12.0
    m0 = solve_lateral(Hcol, 80 * 12, 2.5e9, 5.0e9, 84.0, 0.0, prof, V)
    mp = solve_lateral(Hcol, 80 * 12, 2.5e9, 5.0e9, 84.0, 2000.0, prof, V)
    gi = m0.ground_index
    assert mp.moment[gi] > m0.moment[gi] > 0


def test_soft_soil_high_force_flagged_unstable_not_rigid():
    """A very high pile-head force into soft clay must be flagged unstable — and
    must NOT be credited with a stiff (Df_eq -> 0) base, which would be
    dangerously unconservative."""
    soft = SoilProfile((SoilLayer.from_engineering(
        120, "matlock_soft_clay", 115, su_top_ksf=0.4, eps50=0.02,
        submerged=True),))
    # 10-ft column class demand: huge head force + axial
    sol = solve_lateral(25 * 12, 120 * 12, 8.0e9, 2.0e10, 156.0, 6000.0, soft,
                        2500.0)
    assert not sol.stable
    assert sol.Df_eq >= 50.0 * 156.0 - 1e-6          # flexible sentinel, not 0
    assert sol.Df_eq > 0.0                            # never a rigid base


def test_singular_solve_degrades_to_unstable(monkeypatch):
    """A singular/failed linear solve must NOT crash — it degrades to the
    unstable flexible sentinel (like the physicality guard), so a batch row
    reports a FAILED stability check instead of 'ERROR: singular matrix'."""
    import seismic_column.pile_solver as ps

    def _raise(*_a, **_k):
        raise np.linalg.LinAlgError("singular matrix")

    monkeypatch.setattr(ps, "solve_banded", _raise)
    sol = solve_lateral(198.0, 60 * 12, 3e10, 3e10, 124.0, 1360.0,
                        _sand(), 500.0)
    assert not sol.stable and not sol.converged
    assert sol.Df_eq == pytest.approx(50.0 * 124.0)   # flexible sentinel
    assert np.all(np.isfinite(sol.y))                 # no nan/inf leaked out


def test_nonfinite_solve_degrades_to_unstable(monkeypatch):
    """If LAPACK returns inf/nan (near-singular) instead of raising, the solver
    still degrades gracefully rather than propagating non-finite results."""
    import seismic_column.pile_solver as ps

    def _nan_shaped(l_and_u, ab, F):
        return np.full_like(F, np.inf)   # correctly-shaped non-finite result

    monkeypatch.setattr(ps, "solve_banded", _nan_shaped)
    sol = solve_lateral(198.0, 60 * 12, 3e10, 3e10, 124.0, 1360.0,
                        _sand(), 500.0)
    assert not sol.stable
    assert np.all(np.isfinite(sol.y))


def test_unstable_solve_bails_early(monkeypatch):
    """A P-Δ-buckling (soft-soil, huge load) solve must bail out well before the
    max_iter cap instead of grinding every secant pass — the perf guard that
    keeps a big soil optimise from looking like a crash."""
    import seismic_column.pile_solver as ps
    calls = {"n": 0}
    orig = ps.solve_banded

    def counted(l_and_u, ab, F):
        calls["n"] += 1
        return orig(l_and_u, ab, F)

    monkeypatch.setattr(ps, "solve_banded", counted)
    # very soft soil + very large head force on a slender pile -> runs away
    soft = SoilProfile((SoilLayer.from_engineering(
        80, "api_sand", 110, phi_deg=20, k_pci=5, submerged=True),),
        stiffness_factor=0.1)
    sol = solve_lateral(240.0, 40 * 12, 5e8, 5e8, 60.0, 8000.0, soft, 6000.0,
                        max_iter=100)
    assert not sol.stable
    assert calls["n"] < 60          # bailed well before the 100-pass cap
