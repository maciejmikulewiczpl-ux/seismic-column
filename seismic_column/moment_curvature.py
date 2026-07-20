"""Fibre-based moment-curvature analysis with Caltrans bilinear idealisation.

For a circular RC section under constant axial load, the section is discretised
into concrete and steel fibres.  For each imposed curvature the centroidal
strain is solved so the internal axial force matches the applied load, and the
moment is integrated.  The analysis stops at the controlling ultimate limit
(confined-concrete crushing at ``eps_cu`` or longitudinal-bar fracture at the
reduced ultimate tensile strain).  The resulting curve is idealised to the
Caltrans elasto-plastic bilinear (equal-area) form to extract the idealised
yield curvature ``phi_y``, plastic moment ``Mp`` and ultimate curvature
``phi_u``.

Sign convention: compression is positive for strain and stress; a positive
curvature places the extreme compression fibre at the top (y = +R).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .section import CircularSection


@dataclass
class MomentCurvature:
    phi: np.ndarray            # curvature array, 1/in
    M: np.ndarray             # moment array, kip-in
    phi_y: float              # idealised yield curvature, 1/in
    Mp: float                 # idealised plastic moment, kip-in
    phi_u: float              # ultimate curvature, 1/in
    Mu: float                 # moment at ultimate, kip-in
    phi_yield_first: float    # first-yield curvature, 1/in
    M_yield_first: float      # first-yield moment, kip-in
    failure_mode: str         # "concrete" or "steel"
    EI_eff: float             # effective flexural rigidity Mp/phi_y, kip-in^2
    axial: float              # axial load used, kip (compression +)


def _section_response(
    section: CircularSection,
    y_c: np.ndarray,
    A_conf: np.ndarray,
    A_unc: np.ndarray,
    y_s: np.ndarray,
    A_s: np.ndarray,
    conf_stress,
    unc_stress,
    steel_stress,
    eps_a: float,
    kappa: float,
) -> tuple[float, float]:
    """Return internal (axial force N, moment M) for a strain state."""
    eps_c = eps_a + kappa * y_c
    fc_conf = conf_stress(eps_c)
    fc_unc = unc_stress(eps_c)
    n_conc = np.sum(fc_conf * A_conf + fc_unc * A_unc)
    m_conc = np.sum((fc_conf * A_conf + fc_unc * A_unc) * y_c)

    eps_s = eps_a + kappa * y_s
    fs = steel_stress(eps_s)
    fc_at_steel = conf_stress(eps_s)  # concrete displaced by the bars
    n_steel = np.sum((fs - fc_at_steel) * A_s)
    m_steel = np.sum((fs - fc_at_steel) * A_s * y_s)

    return n_conc + n_steel, m_conc + m_steel


def moment_curvature(
    section: CircularSection,
    axial: float,
    n_points: int = 60,
    kappa_max_factor: float = 1.2,
) -> MomentCurvature:
    """Compute the moment-curvature response for ``section`` at ``axial`` load.

    Parameters
    ----------
    section:
        The circular RC section.
    axial:
        Applied axial load, kip (compression positive).
    n_points:
        Number of curvature increments used to trace the curve.
    kappa_max_factor:
        Safety factor on the estimated failure curvature for the sweep bound.
    """
    y_c, A_conf, A_unc = section.concrete_fibres()
    y_s, A_s = section.steel_fibres()

    conf_stress = np.vectorize(section.confined.stress)
    unc_stress = np.vectorize(section.unconfined.stress)
    steel_stress = np.vectorize(section.steel.stress)

    Rc = section.ds / 2.0
    eps_cu = section.confined.eps_cu
    eps_su_r = section.steel.eps_su_r
    eps_ye = section.steel.eps_ye
    y_bar_min = float(np.min(y_s))  # extreme tension bar

    def solve_eps_a(kappa: float) -> float:
        def resid(eps_a: float) -> float:
            n, _ = _section_response(
                section, y_c, A_conf, A_unc, y_s, A_s,
                conf_stress, unc_stress, steel_stress, eps_a, kappa,
            )
            return n - axial
        lo, hi = -0.05, eps_cu * 1.5 + 0.01
        f_lo, f_hi = resid(lo), resid(hi)
        # expand bracket if needed
        tries = 0
        while f_lo * f_hi > 0 and tries < 20:
            lo -= 0.02
            hi += 0.01
            f_lo, f_hi = resid(lo), resid(hi)
            tries += 1
        if f_lo * f_hi > 0:
            raise RuntimeError("axial equilibrium not bracketed")
        return brentq(resid, lo, hi, xtol=1e-9, maxiter=200)

    # estimate a generous max curvature from ultimate concrete strain
    kappa_fail_est = eps_cu / max(Rc, 1e-6) * 3.0
    kappas = np.linspace(0.0, kappa_fail_est * kappa_max_factor, n_points)

    phi_list: list[float] = [0.0]
    m_list: list[float] = [0.0]
    failure_mode = "concrete"
    phi_u = kappas[-1]
    Mu = 0.0

    for kappa in kappas[1:]:
        try:
            eps_a = solve_eps_a(kappa)
        except RuntimeError:
            break
        _, m = _section_response(
            section, y_c, A_conf, A_unc, y_s, A_s,
            conf_stress, unc_stress, steel_stress, eps_a, kappa,
        )
        eps_top_core = eps_a + kappa * Rc          # confined compression fibre
        eps_steel_tension = -(eps_a + kappa * y_bar_min)  # tensile positive

        phi_list.append(kappa)
        m_list.append(m)

        if eps_top_core >= eps_cu or eps_steel_tension >= eps_su_r:
            failure_mode = "concrete" if (eps_top_core / eps_cu) >= (eps_steel_tension / eps_su_r) else "steel"
            phi_u = kappa
            Mu = m
            break
    else:
        phi_u = phi_list[-1]
        Mu = m_list[-1]

    phi = np.array(phi_list)
    M = np.array(m_list)

    # ---- first yield: extreme tension steel = eps_ye OR conc. comp = 0.002 ----
    phi_fy, M_fy = _first_yield(
        section, y_c, A_conf, A_unc, y_s, A_s,
        conf_stress, unc_stress, steel_stress, solve_eps_a,
        Rc, y_bar_min, eps_ye, phi_u,
    )

    # ---- equal-area bilinear idealisation ----
    Mp, phi_y = _idealise(phi, M, phi_fy, M_fy, phi_u)
    EI_eff = Mp / phi_y if phi_y > 0 else float("nan")

    return MomentCurvature(
        phi=phi, M=M, phi_y=phi_y, Mp=Mp, phi_u=phi_u, Mu=Mu,
        phi_yield_first=phi_fy, M_yield_first=M_fy,
        failure_mode=failure_mode, EI_eff=EI_eff, axial=axial,
    )


def _first_yield(
    section, y_c, A_conf, A_unc, y_s, A_s,
    conf_stress, unc_stress, steel_stress, solve_eps_a,
    Rc, y_bar_min, eps_ye, phi_u,
) -> tuple[float, float]:
    """Locate the first-yield curvature by bisection on the governing strain."""

    def governing_margin(kappa: float) -> float:
        eps_a = solve_eps_a(kappa)
        eps_top_core = eps_a + kappa * Rc
        eps_steel_tension = -(eps_a + kappa * y_bar_min)
        # positive once either limit (steel yield / concrete 0.002) reached
        return max(eps_steel_tension - eps_ye, eps_top_core - 0.002)

    lo, hi = 1e-8, phi_u
    if governing_margin(hi) < 0:
        # never yields within range: fall back to ultimate
        eps_a = solve_eps_a(hi)
        _, m = _section_response(
            section, y_c, A_conf, A_unc, y_s, A_s,
            conf_stress, unc_stress, steel_stress, eps_a, hi,
        )
        return hi, m
    phi_fy = brentq(governing_margin, lo, hi, xtol=1e-12, maxiter=200)
    eps_a = solve_eps_a(phi_fy)
    _, m_fy = _section_response(
        section, y_c, A_conf, A_unc, y_s, A_s,
        conf_stress, unc_stress, steel_stress, eps_a, phi_fy,
    )
    return phi_fy, m_fy


def _idealise(
    phi: np.ndarray, M: np.ndarray, phi_fy: float, M_fy: float, phi_u: float
) -> tuple[float, float]:
    """Equal-area elasto-plastic idealisation -> (Mp, phi_y)."""
    # initial (elastic) stiffness from the first-yield point
    k = M_fy / phi_fy if phi_fy > 0 else float("nan")
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    area_actual = float(_trapz(M, phi))
    # solve 0.5/k * Mp^2 - phi_u * Mp + area_actual = 0 for Mp (smaller root)
    disc = phi_u ** 2 - 2.0 * area_actual / k
    disc = max(disc, 0.0)
    Mp = k * (phi_u - np.sqrt(disc))
    phi_y = Mp / k
    return Mp, phi_y
