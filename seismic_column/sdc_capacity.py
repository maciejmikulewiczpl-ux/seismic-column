"""Caltrans SDC 2.0 capacity calculations and design checks (ESA).

This module ties together the moment-curvature response, the Type II shaft
geometry and the seismic demand to evaluate a single column, for one or more
assumed points of fixity, and returns the full set of SDC checks:

* displacement capacity ``Dc`` vs demand ``Dd``
* displacement ductility demand ``mu_d`` vs limit
* longitudinal / transverse reinforcement limits
* shear capacity (SDC 3.6) vs overstrength shear
* P-Delta
* minimum lateral strength
* Type II shaft capacity protection (flexure and shear)

Unless stated otherwise, forces are kip, lengths in, stresses ksi.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .demand import DemandResult, DesignSpectrum, displacement_demand
from .geometry import Geometry
from .materials import bar_diameter
from .moment_curvature import MomentCurvature, moment_curvature
from .provisions import SDC_2_0, CodeProvisions
from .section import CircularSection

OVERSTRENGTH_FACTOR = 1.2   # SDC A706 overstrength magnification, Mo = 1.2 Mp
PHI_SHEAR = 0.90            # shear resistance factor
MU_D_LIMIT_SINGLE = 5.0    # single-column bent displacement ductility limit
UNIT_WEIGHT_DEFAULT = 0.150  # reinforced concrete unit weight, kip/ft^3 (kcf)


def column_self_weight(Ag: float, Hcol: float, unit_weight_kcf: float) -> float:
    """Self-weight of the column shaft length above the hinge, kip.

    Ag in in^2, Hcol in in, unit weight in kcf (kip/ft^3 -> /1728 for kip/in^3).
    """
    return unit_weight_kcf / 1728.0 * Ag * Hcol


@dataclass
class Check:
    name: str
    demand: float
    capacity: float
    passed: bool
    note: str = ""

    @property
    def ratio(self) -> float:
        if self.capacity == 0:
            return float("inf")
        return self.demand / self.capacity


# ---------------------------------------------------------------------------
# Component capacity helpers
# ---------------------------------------------------------------------------
def plastic_hinge_length(Hcol: float, fye: float, dbl: float) -> float:
    """Analytical plastic hinge length Lp (SDC 4.x), in.

    Lp = 0.08*L + 0.15*fye*dbl >= 0.3*fye*dbl   (fye ksi, dbl in, L in)
    """
    return max(0.08 * Hcol + 0.15 * fye * dbl, 0.3 * fye * dbl)


def concrete_shear_stress(fc: float, rho_s: float, fyh: float, P: float,
                          Ag: float, mu_d: float, inside_hinge: bool) -> float:
    """Concrete shear stress vc (ksi) per SDC 3.6.2."""
    fc_psi = fc * 1000.0
    if inside_hinge:
        f1 = rho_s * fyh / 0.15 + 3.67 - mu_d
        f1 = min(max(f1, 0.3), 3.0)
    else:
        f1 = 3.0
    P_lb = max(P, 0.0) * 1000.0
    f2 = min(1.0 + P_lb / (2000.0 * Ag), 1.5)
    vc_psi = f1 * f2 * math.sqrt(fc_psi)
    vc_psi = min(vc_psi, 4.0 * math.sqrt(fc_psi))
    return vc_psi / 1000.0


def shear_capacity(section: CircularSection, P: float, mu_d: float,
                   inside_hinge: bool, vs_max_coeff: float | None = None,
                   ) -> tuple[float, float, float]:
    """Return (phi*Vn, Vc, Vs) in kip for a circular section (SDC 3.6 / AASHTO 8.6).

    If ``vs_max_coeff`` is given, Vs is capped at ``coeff*sqrt(f'c)*Ae`` (max
    shear reinforcement, e.g. AASHTO 8.6.4 with coeff = 0.25, f'c in ksi).
    """
    Ag = section.Ag
    Ae = 0.8 * Ag
    vc = concrete_shear_stress(section.fc, section.rho_s, section.fyh, P, Ag,
                               mu_d, inside_hinge)
    Vc = vc * Ae
    # transverse (spiral) shear: Vs = (pi/2) * Asp * fyh * D' / s
    Asp = section.transverse_area()
    Dp = section.ds
    Vs = (math.pi / 2.0) * Asp * section.fyh * Dp / section.spiral_spacing
    if vs_max_coeff is not None:
        Vs = min(Vs, vs_max_coeff * math.sqrt(section.fc) * Ae)
    Vn = Vc + Vs
    return PHI_SHEAR * Vn, Vc, Vs


def min_transverse_ratio(fc: float, fyh: float, Ag: float, Acore: float,
                         c1: float = 0.45, c2: float = 0.12,
                         floor: float = 0.0) -> float:
    """Minimum volumetric transverse steel ratio.

    max of the ACI/Caltrans confinement terms and a constant floor (e.g. the
    AASHTO 8.6.5 minimum rho_s >= 0.005).
    """
    return max(c1 * (Ag / Acore - 1.0) * fc / fyh, c2 * fc / fyh, floor)


# ---------------------------------------------------------------------------
# Per-fixity-bound result
# ---------------------------------------------------------------------------
@dataclass
class BoundResult:
    multiplier: float
    fixity_depth: float
    Le: float
    stiffness: float
    demand: DemandResult
    delta_y: float
    delta_p: float
    delta_c: float
    mu_capacity: float
    mu_demand: float
    shaft_moment_interface: float   # Mo at top of shaft
    shaft_moment_fixity: float      # Mo amplified to point of fixity (informational)
    lle_demand: DemandResult | None = None   # low-level earthquake demand
    mu_lle: float | None = None              # LLE displacement ductility (elastic if <=1)


@dataclass
class ColumnAssessment:
    mc_col: MomentCurvature
    mc_shaft: MomentCurvature
    Lp: float
    EI_col: float
    EI_shaft: float
    Ieff_col: float
    Ieff_shaft: float
    Ig_col: float
    Ig_shaft: float
    Mo: float
    bounds: list[BoundResult]
    checks: list[Check]
    governing_bound: BoundResult
    W_self: float = 0.0        # column self-weight above hinge, kip
    P_used: float = 0.0        # axial used for capacity (entered + self-weight), kip
    weight_mass: float = 0.0   # seismic weight used for mass (entered + participation), kip
    axial_entered: float = 0.0
    weight_entered: float = 0.0
    provisions: CodeProvisions = SDC_2_0

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def evaluate_column(
    column: CircularSection,
    shaft: CircularSection,
    geometry: Geometry,
    spectrum: DesignSpectrum,
    axial: float,
    weight: float,
    fixity_multipliers: tuple[float, ...] = (3.0, 6.0),
    mu_d_limit: float = MU_D_LIMIT_SINGLE,
    rho_l_min: float = 0.01,
    rho_l_max: float = 0.04,
    shaft_axial: float | None = None,
    shaft_moment_basis: str = "interface",
    lle_spectrum=None,
    lle_mu_limit: float = 1.0,
    concrete_unit_weight: float = UNIT_WEIGHT_DEFAULT,
    self_weight_mass_factor: float = 1.0 / 3.0,
    self_weight_in_axial: bool = True,
    provisions: CodeProvisions = SDC_2_0,
) -> ColumnAssessment:
    """Full SDC 2.0 ESA assessment of a column on a Type II shaft.

    Runs the analysis for every fixity multiplier and returns an envelope
    (worst case) set of checks.

    ``shaft_moment_basis`` selects the flexural capacity-protection demand on
    the shaft: ``"interface"`` (default) uses the column overstrength moment Mo
    at the top of the shaft (standard SDC capacity-protection demand);
    ``"fixity"`` amplifies Mo linearly to the assumed point of fixity
    (conservative, requires no soil model).

    ``lle_spectrum`` (optional) is a lower-level / functional-evaluation
    earthquake spectrum.  When supplied, the column is checked to remain
    essentially elastic under it (displacement ductility ``mu_LLE <=
    lle_mu_limit``, default 1.0).

    Column self-weight (of the column length above the hinge) is added to the
    axial load used for capacity (if ``self_weight_in_axial``) and a fraction
    ``self_weight_mass_factor`` of it is added to the seismic weight that drives
    the mass / period / displacement demand (both design and low-level EQ).
    """
    if shaft_moment_basis not in ("interface", "fixity"):
        raise ValueError("shaft_moment_basis must be 'interface' or 'fixity'")

    # --- column self-weight participation ---
    W_self = column_self_weight(column.Ag, geometry.Hcol, concrete_unit_weight)
    P_used = axial + (W_self if self_weight_in_axial else 0.0)
    weight_mass = weight + self_weight_mass_factor * W_self
    if shaft_axial is None:
        shaft_axial = P_used

    mc_col = moment_curvature(column, P_used)
    mc_shaft = moment_curvature(shaft, shaft_axial)

    Ec_col = column.confined.Ec
    Ec_shaft = shaft.confined.Ec
    EI_col = mc_col.EI_eff
    EI_shaft = mc_shaft.EI_eff
    Ig_col = column.gross_inertia()
    Ig_shaft = shaft.gross_inertia()
    Ieff_col = EI_col / Ec_col
    Ieff_shaft = EI_shaft / Ec_shaft

    dbl = bar_diameter(column.long_bar_no)
    Lp = plastic_hinge_length(geometry.Hcol, column.fye, dbl)
    Mo = provisions.overstrength_factor * mc_col.Mp

    bounds: list[BoundResult] = []
    for mult in fixity_multipliers:
        k = geometry.lateral_stiffness(EI_col, EI_shaft, mult)
        demand = displacement_demand(spectrum, k, weight_mass)

        # displacement capacity: hinge at top of shaft, arm = Hcol
        F_y = mc_col.Mp / geometry.Hcol
        delta_y = F_y * geometry.tip_flexibility(EI_col, EI_shaft, mult)
        theta_p = Lp * (mc_col.phi_u - mc_col.phi_y)
        delta_p = theta_p * (geometry.Hcol - Lp / 2.0)
        delta_c = delta_y + delta_p
        mu_capacity = delta_c / delta_y if delta_y > 0 else float("nan")
        mu_demand = demand.disp_demand / delta_y if delta_y > 0 else float("nan")

        Df = geometry.fixity_depth(mult)
        shaft_moment_fixity = Mo * (geometry.Hcol + Df) / geometry.Hcol

        lle_demand = None
        mu_lle = None
        if lle_spectrum is not None:
            lle_demand = displacement_demand(lle_spectrum, k, weight_mass)
            mu_lle = lle_demand.disp_demand / delta_y if delta_y > 0 else float("nan")

        bounds.append(BoundResult(
            multiplier=mult, fixity_depth=Df, Le=geometry.effective_length(mult),
            stiffness=k, demand=demand, delta_y=delta_y, delta_p=delta_p,
            delta_c=delta_c, mu_capacity=mu_capacity, mu_demand=mu_demand,
            shaft_moment_interface=Mo, shaft_moment_fixity=shaft_moment_fixity,
            lle_demand=lle_demand, mu_lle=mu_lle,
        ))

    # governing bound = largest displacement-demand/capacity ratio
    governing = max(bounds, key=lambda b: b.demand.disp_demand / b.delta_c)

    checks = _build_checks(
        column, shaft, geometry, mc_col, mc_shaft, Lp, Mo, P_used, shaft_axial,
        bounds, governing, mu_d_limit, rho_l_min, rho_l_max, shaft_moment_basis,
        lle_spectrum is not None, lle_mu_limit, provisions,
    )

    return ColumnAssessment(
        mc_col=mc_col, mc_shaft=mc_shaft, Lp=Lp, EI_col=EI_col, EI_shaft=EI_shaft,
        Ieff_col=Ieff_col, Ieff_shaft=Ieff_shaft, Ig_col=Ig_col, Ig_shaft=Ig_shaft,
        Mo=Mo, bounds=bounds, checks=checks, governing_bound=governing,
        W_self=W_self, P_used=P_used, weight_mass=weight_mass,
        axial_entered=axial, weight_entered=weight, provisions=provisions,
    )


def _build_checks(
    column, shaft, geometry, mc_col, mc_shaft, Lp, Mo, axial, shaft_axial,
    bounds, governing, mu_d_limit, rho_l_min, rho_l_max, shaft_moment_basis,
    has_lle=False, lle_mu_limit=1.0, provisions=SDC_2_0,
) -> list[Check]:
    checks: list[Check] = []

    # 1. displacement capacity vs demand (governing bound)
    checks.append(Check(
        "Displacement capacity", governing.demand.disp_demand, governing.delta_c,
        governing.delta_c >= governing.demand.disp_demand,
        f"Dd={governing.demand.disp_demand:.2f} in, Dc={governing.delta_c:.2f} in "
        f"(mult={governing.multiplier:g})",
    ))

    # 2. displacement ductility demand limit
    mu_d = max(b.mu_demand for b in bounds)
    checks.append(Check(
        "Displacement ductility demand", mu_d, mu_d_limit, mu_d <= mu_d_limit,
        f"mu_d={mu_d:.2f} <= {mu_d_limit:g}",
    ))

    # 3. longitudinal reinforcement limits (column)
    checks.append(Check(
        "Longitudinal steel ratio (min)", rho_l_min, column.rho_l,
        column.rho_l >= rho_l_min, f"rho_l={column.rho_l:.4f}",
    ))
    checks.append(Check(
        "Longitudinal steel ratio (max)", column.rho_l, rho_l_max,
        column.rho_l <= rho_l_max, f"rho_l={column.rho_l:.4f}",
    ))

    # 4. transverse reinforcement minimum (column)
    rho_s_min = min_transverse_ratio(column.fc, column.fyh, column.Ag, column.Acore,
                                     provisions.conf_c1, provisions.conf_c2,
                                     provisions.rho_s_min_floor)
    checks.append(Check(
        "Transverse steel ratio (min)", rho_s_min, column.rho_s,
        column.rho_s >= rho_s_min, f"rho_s={column.rho_s:.4f} >= {rho_s_min:.4f}",
    ))

    # 5. column shear (inside plastic hinge, governing mu_d)
    phiVn, Vc, Vs = shear_capacity(column, axial, mu_d, inside_hinge=True,
                                   vs_max_coeff=provisions.vs_max_coeff)
    Vo = Mo / geometry.Hcol
    checks.append(Check(
        "Column shear", Vo, phiVn, phiVn >= Vo,
        f"Vo={Vo:.1f} kip, phiVn={phiVn:.1f} kip (Vc={Vc:.1f}, Vs={Vs:.1f})",
    ))

    # 6. P-Delta: ignore if P*Dd <= factor*Mp
    Dd = governing.demand.disp_demand
    pdelta = axial * Dd
    pd_cap = provisions.pdelta_factor * mc_col.Mp
    checks.append(Check(
        "P-Delta", pdelta, pd_cap, pdelta <= pd_cap,
        f"P*Dd={pdelta:.0f} <= {provisions.pdelta_factor:g}*Mp={pd_cap:.0f} kip-in",
    ))

    # 7. minimum lateral strength: Vp = Mp/Hcol >= factor*P
    Vp = mc_col.Mp / geometry.Hcol
    min_v = provisions.min_strength_factor * axial
    checks.append(Check(
        "Minimum lateral strength", min_v, Vp, Vp >= min_v,
        f"Vp={Vp:.1f} kip >= {provisions.min_strength_factor:g}*P={min_v:.1f} kip",
    ))

    # 8. Type II shaft flexural capacity protection: Mne_shaft >= Mo demand
    if shaft_moment_basis == "fixity":
        shaft_moment_demand = max(b.shaft_moment_fixity for b in bounds)
    else:
        shaft_moment_demand = max(b.shaft_moment_interface for b in bounds)
    checks.append(Check(
        "Shaft flexure (capacity protection)", shaft_moment_demand, mc_shaft.Mp,
        mc_shaft.Mp >= shaft_moment_demand,
        f"Mo_demand={shaft_moment_demand:.0f} ({shaft_moment_basis}), "
        f"Mn_shaft={mc_shaft.Mp:.0f} kip-in",
    ))

    # 9. Type II shaft shear capacity protection: phiVn_shaft >= Vo
    phiVn_s, Vc_s, Vs_s = shear_capacity(shaft, shaft_axial, mu_d=1.0,
                                         inside_hinge=False,
                                         vs_max_coeff=provisions.vs_max_coeff)
    checks.append(Check(
        "Shaft shear (capacity protection)", Vo, phiVn_s, phiVn_s >= Vo,
        f"Vo={Vo:.1f} kip, phiVn_shaft={phiVn_s:.1f} kip",
    ))

    # 10. Low-level earthquake: structure must remain essentially elastic
    if has_lle:
        mu_lle = max(b.mu_lle for b in bounds)
        checks.append(Check(
            "Low-level EQ (essentially elastic)", mu_lle, lle_mu_limit,
            mu_lle <= lle_mu_limit,
            f"mu_LLE={mu_lle:.2f} <= {lle_mu_limit:g} "
            f"(Dd_LLE={max(b.lle_demand.disp_demand for b in bounds):.2f} in)",
        ))

    return checks
