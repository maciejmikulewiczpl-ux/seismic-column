"""Design-code provisions: parameter sets and clause references.

Two codes are supported:

* **Caltrans SDC 2.0 (2019)**
* **AASHTO Guide Specifications for LRFD Seismic Bridge Design, 3rd Ed. (2011)**

The seismic *methodology* is common to both documents — Mander confined
concrete, fibre moment-curvature with the Caltrans elasto-plastic (equal-area)
idealisation, the analytical plastic-hinge length, displacement-capacity from
the plastic mechanism, the ESA displacement demand, and the concrete/steel
shear model.  The items that *can* differ between the two codes (longitudinal
and transverse reinforcement limits, displacement-ductility limits, the flexural
overstrength factor, the confinement coefficients, and the P-Delta / minimum-
strength factors) together with the clause references are collected here, so a
single code selection switches them consistently across the whole tool.

IMPORTANT: To the best of current knowledge the numeric limits below coincide
for these two documents; the principal visible difference on switching codes is
the set of clause references cited in the report.  The AASHTO SGS article
numbers reflect the 3rd edition and should be confirmed against your copy — and
because every code-specific value lives in this one module, any confirmed
difference can be encoded here and will propagate everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CodeProvisions:
    key: str
    name: str
    # ---- numeric limits / factors ----
    rho_l_min: float            # min longitudinal steel ratio
    rho_l_max: float            # max longitudinal steel ratio
    mu_d_limit_single: float    # displacement ductility demand limit, single column
    overstrength_factor: float  # Mo = factor * Mp (A706)
    conf_c1: float              # coeff on (Ag/Ac - 1) in rho_s,min
    conf_c2: float              # coeff on f'c/fyh in rho_s,min
    rho_s_min_floor: float      # constant floor on rho_s (e.g. AASHTO 0.005)
    vs_max_coeff: float         # max shear reinf: Vs <= coeff*sqrt(f'c)*Ae
    pdelta_factor: float        # P*Delta <= factor * Mp
    min_strength_factor: float  # Vp >= factor * P_dead
    # ---- clause references ----
    ref_confined: str
    ref_flexure: str
    ref_plastic_hinge: str
    ref_displacement: str
    ref_demand: str
    ref_shear: str
    ref_overstrength: str
    ref_longitudinal: str
    ref_transverse: str
    ref_max_shear: str
    ref_ductility: str
    ref_pdelta: str
    ref_min_strength: str


SDC_2_0 = CodeProvisions(
    key="SDC 2.0",
    name="Caltrans SDC 2.0 (2019)",
    rho_l_min=0.01, rho_l_max=0.04, mu_d_limit_single=5.0,
    overstrength_factor=1.2, conf_c1=0.45, conf_c2=0.12,
    rho_s_min_floor=0.0, vs_max_coeff=0.25,
    pdelta_factor=0.25, min_strength_factor=0.10,
    ref_confined="Caltrans SDC 2.0 §3.2.6; Mander et al. (1988)",
    ref_flexure="Caltrans SDC 2.0 §3.3 (moment-curvature)",
    ref_plastic_hinge="Caltrans SDC 2.0 §4.3.1 (analytical plastic-hinge length)",
    ref_displacement="Caltrans SDC 2.0 §4.2 (displacement capacity)",
    ref_demand="Caltrans SDC 2.0 §2.1 / §5.2 (ESA)",
    ref_shear="Caltrans SDC 2.0 §3.6.2 (concrete) & §3.6.3 (transverse)",
    ref_overstrength="Caltrans SDC 2.0 §4.3.1 (Mo = 1.2·Mp, A706)",
    ref_longitudinal="Caltrans SDC 2.0 §3.7.1 (0.01 ≤ ρl ≤ 0.04)",
    ref_transverse="Caltrans SDC 2.0 §3.8.1 (confinement); ACI 318 spiral",
    ref_max_shear="Caltrans SDC 2.0 §3.6.2 (max shear reinf.: Vs ≤ 0.25√f'c·Ae)",
    ref_ductility="Caltrans SDC 2.0 §4.3.2 (μD ≤ 5, single-column bent)",
    ref_pdelta="Caltrans SDC 2.0 §4.11.1 (P-Δ)",
    ref_min_strength="Caltrans SDC 2.0 §4.8 (minimum lateral strength)",
)

AASHTO_SGS_3 = CodeProvisions(
    key="AASHTO SGS 3rd Ed.",
    name="AASHTO Guide Spec. for LRFD Seismic Bridge Design, 3rd Ed. (2011)",
    rho_l_min=0.01, rho_l_max=0.04, mu_d_limit_single=5.0,
    overstrength_factor=1.2, conf_c1=0.0, conf_c2=0.0,
    rho_s_min_floor=0.005, vs_max_coeff=0.25,
    pdelta_factor=0.25, min_strength_factor=0.10,
    ref_confined="AASHTO SGS 3rd Ed. §8.4.4; Mander et al. (1988)",
    ref_flexure="AASHTO SGS 3rd Ed. §8.4 / §4.11.5 (moment-curvature)",
    ref_plastic_hinge="AASHTO SGS 3rd Ed. §4.11.6 (analytical plastic-hinge length)",
    ref_displacement="AASHTO SGS 3rd Ed. §4.8.1 (displacement capacity)",
    ref_demand="AASHTO SGS 3rd Ed. §4.3–4.4 (ESA / ESM)",
    ref_shear="AASHTO SGS 3rd Ed. §8.6.1–8.6.2 (shear)",
    ref_overstrength="AASHTO SGS 3rd Ed. §8.5 (Mpo = λmo·Mp, λmo = 1.2 A706)",
    ref_longitudinal="AASHTO SGS 3rd Ed. §8.8.1 (0.01 ≤ ρl ≤ 0.04)",
    ref_transverse="AASHTO SGS 3rd Ed. §8.6.5 (ρs ≥ 0.005); confinement §8.8.2 separate",
    ref_max_shear="AASHTO SGS 3rd Ed. §8.6.4 (Vs ≤ 0.25√f'c·Ae)",
    ref_ductility="AASHTO SGS 3rd Ed. §4.9 (μD limits)",
    ref_pdelta="AASHTO SGS 3rd Ed. §4.11.5 (P-Δ)",
    ref_min_strength="AASHTO SGS 3rd Ed. §8.7.1 (minimum lateral strength)",
)

PROVISIONS: dict[str, CodeProvisions] = {
    SDC_2_0.key: SDC_2_0,
    AASHTO_SGS_3.key: AASHTO_SGS_3,
}


def get_provisions(key: str) -> CodeProvisions:
    """Return the provisions for ``key`` (defaults to SDC 2.0)."""
    return PROVISIONS.get(key, SDC_2_0)
