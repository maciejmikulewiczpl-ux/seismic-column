"""p-y curves, effective stress, and the Davisson closed-form (soil.py)."""
import math

import pytest

from seismic_column.soil import (
    KSF_TO_KSI,
    PCF_TO_KCI,
    SoilLayer,
    SoilProfile,
    davisson_fixity_depth,
)

D = 84.0  # shaft diameter, in


def _clay(su_ksf=1.5, eps50=0.01, model="matlock_soft_clay"):
    return SoilProfile((SoilLayer.from_engineering(
        60, model, 120, su_top_ksf=su_ksf, eps50=eps50, submerged=True),))


def _sand(phi=35, k_pci=60):
    return SoilProfile((SoilLayer.from_engineering(
        60, "api_sand", 120, phi_deg=phi, k_pci=k_pci, submerged=True),))


# ---------------------------------------------------------------------------
# unit conversion
# ---------------------------------------------------------------------------
def test_from_engineering_units():
    lyr = SoilLayer.from_engineering(10, "matlock_soft_clay", 120,
                                     su_top_ksf=2.0, submerged=False)
    assert lyr.thickness == pytest.approx(120.0)          # 10 ft -> 120 in
    assert lyr.gamma_eff == pytest.approx(120 * PCF_TO_KCI)
    assert lyr.su_top == pytest.approx(2.0 * KSF_TO_KSI)


def test_submerged_uses_buoyant_weight():
    dry = SoilLayer.from_engineering(10, "api_sand", 120, phi_deg=35)
    wet = SoilLayer.from_engineering(10, "api_sand", 120, phi_deg=35,
                                     submerged=True)
    assert wet.gamma_eff == pytest.approx((120 - 62.4) * PCF_TO_KCI)
    assert wet.gamma_eff < dry.gamma_eff


def test_rejects_unknown_model():
    with pytest.raises(ValueError):
        SoilLayer(thickness=120, py_model="nope", gamma_eff=1e-5)


# ---------------------------------------------------------------------------
# effective stress + strength interpolation
# ---------------------------------------------------------------------------
def test_effective_stress_layered():
    a = SoilLayer.from_engineering(10, "api_sand", 110, phi_deg=32)
    b = SoilLayer.from_engineering(10, "api_sand", 125, phi_deg=36,
                                   submerged=True)
    prof = SoilProfile((a, b))
    # at 120 in (10 ft, base of layer a): sigma = gamma_a * 120
    assert prof.sigma_v_eff(120) == pytest.approx(a.gamma_eff * 120)
    # at 240 in: add layer b over its 120 in
    assert prof.sigma_v_eff(240) == pytest.approx(
        a.gamma_eff * 120 + b.gamma_eff * 120)
    # strictly increasing
    assert prof.sigma_v_eff(240) > prof.sigma_v_eff(120)


def test_su_interpolates_within_layer():
    lyr = SoilLayer.from_engineering(10, "matlock_soft_clay", 120,
                                     su_top_ksf=1.0, su_bot_ksf=3.0)
    assert lyr.su_at(0.0) == pytest.approx(1.0 * KSF_TO_KSI)
    assert lyr.su_at(120.0) == pytest.approx(3.0 * KSF_TO_KSI)
    assert lyr.su_at(60.0) == pytest.approx(2.0 * KSF_TO_KSI)   # midpoint


# ---------------------------------------------------------------------------
# p-y curve shape (all models)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("prof", [_clay(), _clay(model="welch_stiff_clay"),
                                  _sand()])
def test_py_monotonic_and_bounded(prof):
    pu = prof.p_ult(120, D)
    ys = [0.001, 0.01, 0.1, 0.5, 1.0, 3.0, 10.0]
    ps = [prof.p_of_y(120, y, D) for y in ys]
    assert all(p2 >= p1 - 1e-9 for p1, p2 in zip(ps, ps[1:]))   # non-decreasing
    assert all(0.0 <= p <= pu + 1e-9 for p in ps)               # bounded by pu
    assert ps[0] > 0.0


def test_secant_modulus_softens_with_deflection():
    prof = _sand()
    Es = [prof.secant_modulus(120, y, D) for y in (0.05, 0.5, 2.0)]
    assert Es[0] > Es[1] > Es[2]                                # softening


def test_api_sand_initial_modulus_is_k_z():
    prof = _sand(k_pci=60)
    Es0 = prof._initial_modulus(120, D)
    assert Es0 == pytest.approx(60 * 1e-3 * 120, rel=1e-9)      # k*z


def test_api_sand_caps_at_A_pu():
    prof = _sand()
    pu = prof.p_ult(120, D)
    p_big = prof.p_of_y(120, 50.0, D)
    assert p_big == pytest.approx(0.9 * pu, rel=0.02)           # A=0.9 cyclic


def test_matlock_cyclic_cap():
    prof = _clay()
    pu = prof.p_ult(120, D)
    # deep behaviour (z >= XR) caps at 0.72 pu; here shallow -> below that
    p_big = prof.p_of_y(120, 50.0, D)
    assert p_big <= 0.72 * pu + 1e-9


def test_deeper_layer_has_higher_pu_sand():
    prof = _sand()
    assert prof.p_ult(240, D) > prof.p_ult(120, D)             # grows with depth


# ---------------------------------------------------------------------------
# Davisson closed-form
# ---------------------------------------------------------------------------
def test_davisson_positive_and_branches():
    EI = 3.5e9
    lf_clay = davisson_fixity_depth(EI, _clay(), D)
    lf_sand = davisson_fixity_depth(EI, _sand(), D)
    assert lf_clay > 0 and lf_sand > 0
    # stiffer pile -> deeper fixity in both
    assert davisson_fixity_depth(2 * EI, _sand(), D) > lf_sand


def test_davisson_softer_soil_deeper_fixity():
    EI = 3.5e9
    soft = davisson_fixity_depth(EI, _clay(su_ksf=0.5), D)
    stiff = davisson_fixity_depth(EI, _clay(su_ksf=3.0), D)
    assert soft > stiff                                        # softer -> deeper
