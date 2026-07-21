"""Caltrans SDC 2.1 capacity calculations and design checks (ESA).

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

from .demand import (
    DemandResult,
    DesignSpectrum,
    displacement_demand,
    magnified_demand,
)
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


@dataclass
class ShearBreakdown:
    """Every intermediate quantity behind ``phiVn``, for reporting.

    Stresses are ksi, areas in^2, forces kip.  ``alpha`` is the concrete shear
    stress adjustment factor — Caltrans calls it F1, AASHTO calls it alpha'.
    """
    model: str
    alpha: float            # F1 / alpha', after clamping to [0.3, 3.0]
    alpha_raw: float        # before clamping
    fs: float               # rho_s*fyh used in alpha (ksi; AASHTO caps at 0.35)
    fs_raw: float           # rho_s*fyh before the cap
    axial_factor: float     # F2 / (1 + Pc/(2Ag)), after any cap
    axial_factor_raw: float
    vc_uncapped: float      # ksi, before the code's vc cap
    vc_cap: float           # ksi, governing cap on vc
    cap_label: str          # which cap governs, for the report
    vc: float               # ksi
    Ae: float
    Vc: float
    Vs_uncapped: float
    Vs_cap: float
    Vs: float
    Vn: float
    phiVn: float


def _alpha_prime(rho_s: float, fyh: float, mu_d: float, inside_hinge: bool,
                 fs_limit: float | None) -> tuple[float, float, float, float]:
    """Return (alpha, alpha_raw, fs, fs_raw) for the shear stress adjustment.

    ``fs_limit`` caps rho_s*fyh (0.35 ksi in both codes: SDC 5.3.7.2-5 /
    SGS 8.6.2-6); pass ``None`` to leave it uncapped.  Outside the plastic hinge
    alpha' = 3 (SDC 5.3.7.2-4 / SGS 8.6.1).
    """
    fs_raw = rho_s * fyh
    fs = min(fs_raw, fs_limit) if fs_limit is not None else fs_raw
    if not inside_hinge:
        return 3.0, 3.0, fs, fs_raw
    alpha_raw = fs / 0.15 + 3.67 - mu_d
    return min(max(alpha_raw, 0.3), 3.0), alpha_raw, fs, fs_raw


def concrete_shear_stress(fc: float, rho_s: float, fyh: float, P: float,
                          Ag: float, mu_d: float, inside_hinge: bool,
                          model: str = "caltrans") -> float:
    """Concrete shear stress vc (ksi); see :func:`shear_breakdown`."""
    return _concrete_shear(fc, rho_s, fyh, P, Ag, mu_d, inside_hinge, model)[0]


def _concrete_shear(fc: float, rho_s: float, fyh: float, P: float, Ag: float,
                    mu_d: float, inside_hinge: bool, model: str) -> tuple:
    """Return (vc, alpha, alpha_raw, fs, fs_raw, f2, f2_raw, vc_unc, cap, label).

    All stresses ksi.  ``model`` is ``"caltrans"`` (SDC 2.1 §5.3.7.2) or
    ``"aashto"`` (SGS §8.6.2).

    Both models share the ``fs = ρs·fyh ≤ 0.35 ksi`` limit and the net-tension
    rule ``vc = 0`` — SDC 2.1 §5.3.7.2-5 states both explicitly, matching
    SGS §8.6.2-6 and §8.6.2-4.  They differ only in units and the vc cap.
    """
    if model not in ("aashto", "caltrans"):
        raise ValueError(f"unknown shear model {model!r}")

    # fs = ρs·fyh capped at 0.35 ksi in BOTH codes (SDC 5.3.7.2-5 / SGS 8.6.2-6).
    alpha, alpha_raw, fs, fs_raw = _alpha_prime(rho_s, fyh, mu_d, inside_hinge,
                                                0.35)

    if model == "aashto":
        # SGS 8.6.2-4: no concrete contribution unless Pc is compressive.
        if P <= 0.0:
            return (0.0, alpha, alpha_raw, fs, fs_raw, 0.0, 0.0, 0.0, 0.0,
                    "Pc not compressive → vc = 0 (8.6.2-4)")
        root = math.sqrt(fc)                        # f'c in ksi
        f2 = 1.0 + P / (2.0 * Ag)                   # 8.6.2-3, P kip / Ag in^2
        # 0.032, NOT the 0.0032 printed in the 3rd Ed.  The September 2024
        # errata (LRFDSEIS-3-Errata, p. 8-10) corrects "0.0032" to "0.032";
        # 0.0032 under-predicts vc by 10x and would make the 0.047*alpha' cap
        # unreachable.  0.032 ~ sqrt(1000)/1000, the psi->ksi conversion of the
        # equivalent Caltrans SDC §5.3.7.2 expression.
        vc_unc = 0.032 * alpha * f2 * root
        cap_abs = 0.11 * root
        cap_alpha = 0.047 * alpha * root
        cap = min(cap_abs, cap_alpha)
        label = "0.11√f'c" if cap_abs <= cap_alpha else "0.047α'√f'c"
        return (min(vc_unc, cap), alpha, alpha_raw, fs, fs_raw, f2, f2,
                vc_unc, cap, label)

    # Caltrans SDC 2.1 §5.3.7.2, evaluated in psi.
    # 5.3.7.2: "For members with net axial load in tension, vc = 0."
    if P <= 0.0:
        return (0.0, alpha, alpha_raw, fs, fs_raw, 0.0, 0.0, 0.0, 0.0,
                "net axial tension → vc = 0 (5.3.7.2)")
    root_psi = math.sqrt(fc * 1000.0)
    f2_raw = 1.0 + P * 1000.0 / (2000.0 * Ag)       # Pc lb / Ag in^2
    f2 = min(f2_raw, 1.5)
    vc_unc = alpha * f2 * root_psi / 1000.0
    cap = 4.0 * root_psi / 1000.0
    return (min(vc_unc, cap), alpha, alpha_raw, fs, fs_raw, f2, f2_raw,
            vc_unc, cap, "4√f'c")


def shear_breakdown(section: CircularSection, P: float, mu_d: float,
                    inside_hinge: bool,
                    provisions: CodeProvisions = SDC_2_0) -> ShearBreakdown:
    """Full shear derivation for a circular section (SDC 3.6 / SGS 8.6).

    Vs follows SGS 8.6.3-1 for circular hoops/spirals (n = 1 core section) and
    is capped at ``vs_max_coeff*sqrt(f'c)*Ae`` (max shear reinforcement,
    SGS 8.6.4 with coeff = 0.25, f'c in ksi).
    """
    Ag = section.Ag
    Ae = 0.8 * Ag
    (vc, alpha, alpha_raw, fs, fs_raw, f2, f2_raw, vc_unc, vc_cap,
     cap_label) = _concrete_shear(section.fc, section.rho_s, section.fyh, P, Ag,
                                  mu_d, inside_hinge, provisions.shear_model)
    Vc = vc * Ae
    # transverse (spiral) shear: Vs = (pi/2) * Asp * fyh * D' / s
    Asp = section.transverse_area()
    Vs_unc = (math.pi / 2.0) * Asp * section.fyh * section.ds / section.spiral_spacing
    Vs_cap = provisions.vs_max_coeff * math.sqrt(section.fc) * Ae
    Vs = min(Vs_unc, Vs_cap)
    Vn = Vc + Vs
    return ShearBreakdown(
        model=provisions.shear_model, alpha=alpha, alpha_raw=alpha_raw, fs=fs,
        fs_raw=fs_raw, axial_factor=f2, axial_factor_raw=f2_raw,
        vc_uncapped=vc_unc, vc_cap=vc_cap, cap_label=cap_label, vc=vc, Ae=Ae,
        Vc=Vc, Vs_uncapped=Vs_unc, Vs_cap=Vs_cap, Vs=Vs, Vn=Vn,
        phiVn=PHI_SHEAR * Vn,
    )


def shear_capacity(section: CircularSection, P: float, mu_d: float,
                   inside_hinge: bool,
                   provisions: CodeProvisions = SDC_2_0,
                   ) -> tuple[float, float, float]:
    """Return (phi*Vn, Vc, Vs) in kip for a circular section."""
    b = shear_breakdown(section, P, mu_d, inside_hinge, provisions)
    return b.phiVn, b.Vc, b.Vs


def max_transverse_spacing(D: float, dbl: float,
                           bundled: bool) -> tuple[float, str]:
    """Max transverse spacing in the plastic hinge region (SGS 8.8.9), in.

    Smallest of D/5 (columns), 6*dbl, and 6 in. (single hoop/spiral) or 8 in.
    (bundled hoops).  Returns (spacing, governing-term label).
    """
    limit_abs = (8.0, "8 in. (bundled hoops)") if bundled else (6.0, "6 in.")
    candidates = [(D / 5.0, "D/5"), (6.0 * dbl, "6·dbl"), limit_abs]
    return min(candidates, key=lambda c: c[0])


def max_transverse_spacing_caltrans(dbl: float) -> tuple[float, str]:
    """Max transverse spacing inside the plastic hinge (SDC 2.1 §8.4.1.1), in.

    Smallest of 6*dbl and 8 in.  (SDC 2.1 drops the D/5 term that AASHTO uses.)
    """
    candidates = [(6.0 * dbl, "6·dbl"), (8.0, "8 in.")]
    return min(candidates, key=lambda c: c[0])


def max_longitudinal_spacing_caltrans(D: float) -> tuple[float, str]:
    """Max c/c lateral spacing of longitudinal bars (SDC 2.1 §8.4.2), in.

    10 in. for Dc <= 5 ft, 12 in. for Dc > 5 ft.
    """
    if D <= 60.0:
        return 10.0, "10 in. (Dc ≤ 5 ft)"
    return 12.0, "12 in. (Dc > 5 ft)"


def longitudinal_bar_spacing(D: float, cover: float, dsp: float, dbl: float,
                             n_bars: int) -> float:
    """Centre-to-centre spacing of longitudinal bars around the cage, in."""
    r_bars = D / 2.0 - cover - dsp - dbl / 2.0
    return 2.0 * math.pi * r_bars / n_bars


def min_transverse_bar_no(long_bar_no: int, long_bundled: bool) -> int:
    """Minimum transverse bar size (SGS 8.8.9): #4, or #5 for #10+/bundled."""
    if long_bundled or long_bar_no >= 10:
        return 5
    return 4


def max_bar_diameter(fc: float, L: float, Dc: float, fye: float) -> float:
    """Max longitudinal bar diameter for bond (SGS 8.8.6-1), in.

    dbl <= 0.79*sqrt(f'c)*(L - 0.5*Dc)/fye, with f'c and fye in ksi, L and Dc
    in in.  ``L`` is contraflexure-to-max-moment length.
    """
    return 0.79 * math.sqrt(fc) * (L - 0.5 * Dc) / fye


def anchorage_length(dbl: float, fye: float, fc: float, bundle: int = 1) -> float:
    """Required anchorage length into a cap beam or footing (SGS 8.8.4-1), in.

    l_ac >= 0.79*dbl*fye/sqrt(f'c), increased 20% for two-bar and 50% for
    three-bar bundles per SGS 8.8.5.  Note this is the cap-beam/footing rule;
    a column embedded in an *oversized* shaft is governed instead by SGS 8.8.10
    (Dc,max + ld staggered), which needs ld from AASHTO LRFD 5.10.8.2.1 and is
    therefore not evaluated here.
    """
    base = 0.79 * dbl * fye / math.sqrt(fc)
    return base * {1: 1.0, 2: 1.2, 3: 1.5}.get(bundle, 1.0)


def effective_bar_diameter(dbl: float, bundle: int) -> float:
    """Effective bar diameter for the SGS 8.8.6 bond check (1.2x / 1.5x)."""
    return dbl * {1: 1.0, 2: 1.2, 3: 1.5}.get(bundle, 1.0)


def min_transverse_ratio(fc: float, fyh: float, Ag: float, Acore: float,
                         c1: float = 0.45, c2: float = 0.12,
                         floor: float = 0.0) -> float:
    """Minimum volumetric transverse steel ratio (ACI/legacy form + floor).

    max of the ACI confinement terms and a constant floor.  Used for AASHTO SGS
    (c1 = c2 = 0, floor = 0.005 per §8.6.5).  **Not** used for Caltrans SDC 2.1,
    whose minimum comes from :func:`caltrans_min_transverse_ratio` (Table
    5.3.8.2-1) — the ACI equations were removed in SDC 2.0.
    """
    return max(c1 * (Ag / Acore - 1.0) * fc / fyh, c2 * fc / fyh, floor)


def axial_load_ratio(P_dead: float, fc: float, Ag: float) -> float:
    """Axial load ratio due to dead load ρdl = Pdl/(f'c·Ag) (SDC 2.1 §5.3.3-1).

    f'c is capped at 5.0 ksi per §5.3.3.  ``P_dead`` kip, ``fc`` ksi (nominal),
    ``Ag`` in^2.  Dimensionless (multiply by 100 for the percentage used in
    Table 5.3.8.2-1).
    """
    return P_dead / (min(fc, 5.0) * Ag)


def caltrans_min_transverse_ratio(
    D: float, Hcol: float, rho_l: float, P_dead: float, fc: float, Ag: float,
) -> tuple[float, bool, str]:
    """Minimum transverse volumetric ratio from SDC 2.1 Table 5.3.8.2-1.

    Ordinary Standard bridges, inside the plastic hinge region.  The table is
    keyed on aspect ratio ``L/Dc``, column diameter ``Dc`` (ft), longitudinal
    ratio ``ρl`` (%), and dead-load axial ratio ``ρdl`` (%).

    Returns ``(rho_s_min, in_table, note)``.  When the section falls outside the
    tabulated parameter ranges (``in_table`` False) the table cannot supply a
    value — SDC 2.1 directs the designer to the PSDC procedure to establish a
    ρs,min giving μc ≥ 3.0 — so the check must be flagged for manual review
    rather than silently passed.

    Recovery Standard bridges (ρs,min = 0.01) are out of this tool's scope.
    """
    Dc_ft = D / 12.0
    L_over_Dc = Hcol / D
    rho_l_pct = rho_l * 100.0
    rho_dl_pct = axial_load_ratio(P_dead, fc, Ag) * 100.0

    def outside(why: str) -> tuple[float, bool, str]:
        return (0.0, False,
                f"outside Table 5.3.8.2-1 ({why}) → establish ρs,min via PSDC "
                f"procedure for μc ≥ 3.0")

    if L_over_Dc > 8.0:
        return outside(f"L/Dc = {L_over_Dc:.1f} > 8.0")
    if rho_dl_pct > 15.0:
        return outside(f"ρdl = {rho_dl_pct:.1f}% > 15%")

    if 3.0 <= Dc_ft <= 6.0:
        if rho_l_pct > 2.3:
            return outside(f"ρl = {rho_l_pct:.2f}% > 2.3% for Dc ≤ 6 ft")
        rho_s_min = 0.006 if rho_dl_pct <= 10.0 else 0.007
    elif 6.0 < Dc_ft <= 11.0:
        if rho_l_pct > 2.15:
            return outside(f"ρl = {rho_l_pct:.2f}% > 2.15% for Dc > 6 ft")
        rho_s_min = 0.007 if rho_dl_pct <= 10.0 else 0.008
    else:
        return outside(f"Dc = {Dc_ft:.1f} ft outside 3–11 ft")

    band = "≤10%" if rho_dl_pct <= 10.0 else "10–15%"
    return (rho_s_min, True,
            f"Table 5.3.8.2-1: Dc={Dc_ft:.1f}ft, L/Dc={L_over_Dc:.1f}, "
            f"ρl={rho_l_pct:.2f}%, ρdl={rho_dl_pct:.1f}% ({band}) → "
            f"ρs,min={rho_s_min:.3f}")


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
    """Full SDC 2.1 ESA assessment of a column on a Type II shaft.

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
        # SGS 4.3.3: magnify the elastic displacement demand for short-period
        # structures (equal-displacement does not hold there).  Caltrans SDC
        # has no equivalent, so this is gated on the selected code.
        if provisions.short_period_magnification:
            demand = magnified_demand(demand, spectrum, delta_y)
        mu_demand = demand.disp_demand / delta_y if delta_y > 0 else float("nan")

        Df = geometry.fixity_depth(mult)
        shaft_moment_fixity = Mo * (geometry.Hcol + Df) / geometry.Hcol

        lle_demand = None
        mu_lle = None
        if lle_spectrum is not None:
            lle_demand = displacement_demand(lle_spectrum, k, weight_mass)
            if provisions.short_period_magnification:
                lle_demand = magnified_demand(lle_demand, lle_spectrum, delta_y)
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

    # 4. transverse reinforcement minimum (column, inside plastic hinge)
    if provisions.transverse_min_model == "caltrans_table":
        rho_s_min, in_table, note = caltrans_min_transverse_ratio(
            column.D, geometry.Hcol, column.rho_l, axial, column.fc, column.Ag)
        # Outside the table the code offers no formula — the designer must run
        # the PSDC procedure — so we cannot certify the minimum is met.
        passed = in_table and column.rho_s >= rho_s_min
        checks.append(Check(
            "Transverse steel ratio (min)", rho_s_min, column.rho_s, passed,
            f"rho_s={column.rho_s:.4f}; {note}",
        ))
    else:
        rho_s_min = min_transverse_ratio(
            column.fc, column.fyh, column.Ag, column.Acore,
            provisions.conf_c1, provisions.conf_c2, provisions.rho_s_min_floor)
        checks.append(Check(
            "Transverse steel ratio (min)", rho_s_min, column.rho_s,
            column.rho_s >= rho_s_min,
            f"rho_s={column.rho_s:.4f} >= {rho_s_min:.4f}",
        ))

    # 5. column shear (inside plastic hinge, governing mu_d)
    phiVn, Vc, Vs = shear_capacity(column, axial, mu_d, inside_hinge=True,
                                   provisions=provisions)
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
    # SGS 8.9: for oversized (Type II) shafts the expected nominal capacity must
    # reach 1.25x the demand generated by the column overstrength hinge.
    gamma = provisions.shaft_demand_factor
    shaft_moment_demand *= gamma
    checks.append(Check(
        "Shaft flexure (capacity protection)", shaft_moment_demand, mc_shaft.Mp,
        mc_shaft.Mp >= shaft_moment_demand,
        f"{gamma:g}·Mo_demand={shaft_moment_demand:.0f} ({shaft_moment_basis}), "
        f"Mn_shaft={mc_shaft.Mp:.0f} kip-in",
    ))

    # 9. Type II shaft shear capacity protection: phiVn_shaft >= Vo
    phiVn_s, Vc_s, Vs_s = shear_capacity(shaft, shaft_axial, mu_d=1.0,
                                         inside_hinge=False,
                                         provisions=provisions)
    checks.append(Check(
        "Shaft shear (capacity protection)", Vo, phiVn_s, phiVn_s >= Vo,
        f"Vo={Vo:.1f} kip, phiVn_shaft={phiVn_s:.1f} kip",
    ))

    # 9b. Oversized-shaft lateral confinement (SGS 8.8.12): the shaft's
    # volumetric ratio must reach 50% of the confinement at the column base.
    # The clause conditions this on the shaft carrying 1.25x the column
    # overstrength demand, which is check 8 above.
    if provisions.shaft_confinement_fraction is not None:
        rho_s_req = provisions.shaft_confinement_fraction * column.rho_s
        checks.append(Check(
            "Shaft confinement (oversized)", rho_s_req, shaft.rho_s,
            shaft.rho_s >= rho_s_req,
            f"rho_s,shaft={shaft.rho_s:.4f} >= "
            f"{provisions.shaft_confinement_fraction:g}·rho_s,col="
            f"{rho_s_req:.4f}",
        ))

    # 10. Detailing limits — max transverse spacing (both codes, different rules)
    if provisions.detailing_checks:
        dbl = bar_diameter(column.long_bar_no)
        if provisions.max_tie_spacing_model == "caltrans_8.4.1":
            s_max, s_gov = max_transverse_spacing_caltrans(dbl)
        else:
            s_max, s_gov = max_transverse_spacing(
                column.D, dbl, column.spiral_bundle > 1)
        checks.append(Check(
            "Transverse spacing (max)", column.spiral_spacing, s_max,
            column.spiral_spacing <= s_max,
            f"s={column.spiral_spacing:g} in <= {s_max:.2f} in ({s_gov} governs)",
        ))

        if provisions.max_tie_spacing_model == "caltrans_8.4.1":
            # SDC 2.1 §8.4.2: max c/c lateral spacing of longitudinal bars.
            dsp = bar_diameter(column.spiral_bar_no)
            s_long = longitudinal_bar_spacing(column.D, column.cover, dsp, dbl,
                                              column.n_bars)
            s_long_max, gov = max_longitudinal_spacing_caltrans(column.D)
            checks.append(Check(
                "Longitudinal bar spacing (max)", s_long, s_long_max,
                s_long <= s_long_max,
                f"s_long={s_long:.1f} in <= {s_long_max:g} in ({gov})",
            ))
        else:
            # AASHTO §8.8.9 min tie size and §8.8.6 bar-diameter bond.
            bar_min = min_transverse_bar_no(column.long_bar_no,
                                            column.long_bundle > 1)
            checks.append(Check(
                "Transverse bar size (min)", bar_min, column.spiral_bar_no,
                column.spiral_bar_no >= bar_min,
                f"#{column.spiral_bar_no} >= #{bar_min} required for "
                f"#{column.long_bar_no} longitudinal bars",
            ))
            dbl_eff = effective_bar_diameter(dbl, column.long_bundle)
            dbl_max = max_bar_diameter(column.fc, geometry.Hcol, column.D,
                                       column.fye)
            checks.append(Check(
                "Longitudinal bar diameter (bond)", dbl_eff, dbl_max,
                dbl_eff <= dbl_max,
                f"dbl_eff={dbl_eff:.3f} in <= 0.79·√f'c·(L−0.5·Dc)/fye "
                f"= {dbl_max:.3f} in",
            ))

    # 11a. Maximum axial load in a ductile member (SGS 8.7.2): Pu <= 0.2 f'c Ag
    if provisions.max_axial_coeff is not None:
        p_cap = provisions.max_axial_coeff * column.fc * column.Ag
        # 8.7.2 applies when mu_D > 2 and no moment-curvature pushover is run.
        # This tool derives its load-displacement response from M-phi (C8.5
        # calls that a pushover), so the check is reported for information.
        applies = mu_d > 2.0
        checks.append(Check(
            "Maximum axial load", axial, p_cap,
            (not applies) or axial <= p_cap,
            f"P={axial:.0f} kip vs {provisions.max_axial_coeff:g}·f'c·Ag "
            f"= {p_cap:.0f} kip"
            + ("" if applies else f" (n/a: mu_d={mu_d:.2f} <= 2)"),
        ))

    # 11b. Axial load ratio limit (SDC 2.1 §5.3.3-1): rho_dl = Pdl/(f'c*Ag) <= 0.15
    if provisions.axial_ratio_limit is not None:
        rho_dl = axial_load_ratio(axial, column.fc, column.Ag)
        lim = provisions.axial_ratio_limit
        checks.append(Check(
            "Axial load ratio", rho_dl, lim, rho_dl <= lim,
            f"rho_dl=Pdl/(f'c·Ag)={rho_dl:.3f} <= {lim:g} "
            f"(f'c capped at 5 ksi)",
        ))

    # 12. Low-level earthquake: structure must remain essentially elastic
    if has_lle:
        mu_lle = max(b.mu_lle for b in bounds)
        checks.append(Check(
            "Low-level EQ (essentially elastic)", mu_lle, lle_mu_limit,
            mu_lle <= lle_mu_limit,
            f"mu_LLE={mu_lle:.2f} <= {lle_mu_limit:g} "
            f"(Dd_LLE={max(b.lle_demand.disp_demand for b in bounds):.2f} in)",
        ))

    return checks
