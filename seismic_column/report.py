"""Per-column printable calculation report (Markdown)."""
from __future__ import annotations

import math

from .batch import RowResult
from .materials import bar_area, bar_diameter
from .sdc_capacity import (
    PHI_SHEAR,
    anchorage_length,
    caltrans_min_transverse_ratio,
    shear_breakdown,
    shear_capacity,
)


def _add_concrete_shear(add, sec, d, b, mu_d, P) -> None:
    """Write the vc derivation lines for whichever shear model ran."""
    Ag = sec.Ag
    if b.model == "aashto":
        fs_note = (f" → ≤ 0.35 = **{b.fs:.3f}**" if b.fs < b.fs_raw else "")
        add(f"- Transverse steel stress:  fs = ρs·fyh "
            f"= {sec.rho_s:.4f}·{d.fyh:.0f} = {b.fs_raw:.3f} ksi{fs_note}  "
            f"(8.6.2-6)")
        add(f"- Shear stress adjustment:  α' = fs/0.15 + 3.67 − μΔ "
            f"= {b.fs:.3f}/0.15 + 3.67 − {mu_d:.2f} = {b.alpha_raw:.2f} "
            f"→ clamp[0.3, 3.0] = **{b.alpha:.2f}**  (8.6.2-5)")
        if b.vc == 0.0:
            add(f"- Axial:  Pc = {P:.0f} kip is not compressive → "
                f"**vc = 0**  (8.6.2-4)")
            return
        add(f"- Axial factor:  1 + Pc/(2·Ag) = 1 + {P:.0f}/(2·{Ag:.0f}) "
            f"= **{b.axial_factor:.3f}**")
        add(f"- Concrete stress:  vc = 0.032·α'·(1 + Pc/2Ag)·√f'c "
            f"= 0.032·{b.alpha:.2f}·{b.axial_factor:.3f}·√{d.fc:.1f} "
            f"= {b.vc_uncapped:.4f} ksi")
        add(f"  ≤ min(0.11√f'c, 0.047·α'·√f'c) = {b.vc_cap:.4f} ksi "
            f"({b.cap_label} governs) → **vc = {b.vc:.4f} ksi**  (8.6.2-3)")
        return

    # Caltrans SDC 2.1 §5.3.7.2, presented in psi as the clause is written.
    fc_psi = d.fc * 1000.0
    add(f"- Ductility factor:  F1 = ρs·fyh/0.15 + 3.67 − μd "
        f"= {sec.rho_s:.4f}·{d.fyh:.0f}/0.15 + 3.67 − {mu_d:.2f} "
        f"= {b.alpha_raw:.2f} → clamp[0.3, 3.0] = **{b.alpha:.2f}**")
    add(f"- Axial factor:  F2 = 1 + P/(2000·Ag) "
        f"= 1 + {max(P, 0.0)*1000.0:.0f}/(2000·{Ag:.0f}) "
        f"= {b.axial_factor_raw:.3f} → ≤ 1.5 = **{b.axial_factor:.3f}**")
    add(f"- Concrete stress:  vc = F1·F2·√f'c "
        f"= {b.alpha:.2f}·{b.axial_factor:.3f}·√{fc_psi:.0f} "
        f"= {b.vc_uncapped*1000.0:.0f} psi (≤ 4√f'c = {b.vc_cap*1000.0:.0f}) "
        f"→ **{b.vc*1000.0:.0f} psi**")


def column_report(rr: RowResult) -> str:
    """Return a Markdown calculation report for a single column result."""
    a = rr.assessment
    d = rr.design
    s = rr.shaft
    lines: list[str] = []
    add = lines.append

    add(f"# Seismic Column Report — {rr.name}")
    add("")
    add(f"**Result:** {'PASS ✅' if rr.feasible else 'FAIL ❌'}  "
        f"({'optimised' if rr.optimized else 'as-entered'})")
    add(f"**Design code:** {a.provisions.name}")
    add("")

    add("## Column section")
    add("")
    add(f"- Diameter: {d.D:.0f} in")
    add(f"- f'c: {d.fc:.1f} ksi")
    add(f"- Cover: {d.cover:.1f} in")
    add(f"- Longitudinal: {d.long_label()}  (ρl = {d.rho_l():.4f})")
    add(f"- Spiral/hoop: {d.spiral_label()} "
        f"(ρs = {d.section().rho_s:.4f})")
    add("")

    add("## Type II shaft section")
    add("")
    add(f"- Diameter: {s.D:.0f} in, f'c: {s.fc:.1f} ksi")
    add(f"- Longitudinal: {s.long_label()}  (ρl = {s.rho_l():.4f})")
    add(f"- Spiral/hoop: {s.spiral_label()}")
    add("")

    add("## Moment-curvature (column)")
    add("")
    mc = a.mc_col
    add(f"- First yield: φ = {mc.phi_yield_first:.3e} 1/in, M = {mc.M_yield_first/12:.0f} kip-ft")
    add(f"- Idealised yield: φy = {mc.phi_y:.3e} 1/in, Mp = {mc.Mp/12:.0f} kip-ft")
    add(f"- Ultimate: φu = {mc.phi_u:.3e} 1/in ({mc.failure_mode} controlled)")
    add(f"- Curvature ductility: μφ = {mc.phi_u/mc.phi_y:.1f}")
    add(f"- Confined f'cc = {d.section().confined.fcc:.2f} ksi, "
        f"εcu = {d.section().confined.eps_cu:.4f}")
    add("")

    add("## Effective stiffness")
    add("")
    add(f"- Column: Ieff = {a.Ieff_col:.0f} in⁴, Ig = {a.Ig_col:.0f} in⁴, "
        f"Ieff/Ig = {a.Ieff_col/a.Ig_col:.3f}")
    add(f"- Shaft: Ieff = {a.Ieff_shaft:.0f} in⁴, Ig = {a.Ig_shaft:.0f} in⁴, "
        f"Ieff/Ig = {a.Ieff_shaft/a.Ig_shaft:.3f}")
    add(f"- Plastic hinge length Lp = {a.Lp:.1f} in")
    add(f"- Overstrength moment Mo = {a.Mo/12:.0f} kip-ft")
    add("")

    add("## Demand & capacity by point of fixity")
    add("")
    has_lle = a.bounds[0].mu_lle is not None
    if has_lle:
        add("| mult | Df (in) | Le (in) | T (s) | Sa (g) | Δd (in) | Δy (in) | Δc (in) | μd | Δd,LLE (in) | μLLE |")
        add("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for b in a.bounds:
            add(f"| {b.multiplier:g} | {b.fixity_depth:.0f} | {b.Le:.0f} | "
                f"{b.demand.period:.2f} | {b.demand.Sa:.3f} | {b.demand.disp_demand:.2f} | "
                f"{b.delta_y:.2f} | {b.delta_c:.2f} | {b.mu_demand:.2f} | "
                f"{b.lle_demand.disp_demand:.2f} | {b.mu_lle:.2f} |")
    else:
        add("| mult | Df (in) | Le (in) | T (s) | Sa (g) | Δd (in) | Δy (in) | Δc (in) | μd |")
        add("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for b in a.bounds:
            add(f"| {b.multiplier:g} | {b.fixity_depth:.0f} | {b.Le:.0f} | "
                f"{b.demand.period:.2f} | {b.demand.Sa:.3f} | {b.demand.disp_demand:.2f} | "
                f"{b.delta_y:.2f} | {b.delta_c:.2f} | {b.mu_demand:.2f} |")
    add("")
    if has_lle:
        add("*μLLE ≤ 1.0 ⇒ column remains essentially elastic under the "
            "low-level earthquake.*")
        add("")

    add("## SDC checks")
    add("")
    add("| Check | Demand | Capacity | D/C | Status |")
    add("|:--|--:|--:|--:|:--:|")
    for c in a.checks:
        add(f"| {c.name} | {c.demand:.1f} | {c.capacity:.1f} | {c.ratio:.2f} | "
            f"{'PASS' if c.passed else 'FAIL'} |")
    add("")
    for c in a.checks:
        add(f"- *{c.name}*: {c.note}")
    add("")

    lines.extend(_detailed_calcs(rr))

    if rr.log:
        add("## Optimiser log")
        add("")
        for entry in rr.log:
            add(f"- {entry}")
        add("")

    return "\n".join(lines)


def _detailed_calcs(rr: RowResult) -> list[str]:
    """Full equations with substituted numbers and references."""
    a = rr.assessment
    d = rr.design
    s = rr.shaft
    sec = d.section()
    conf = sec.confined
    mc = a.mc_col
    L = a.bounds[0].Le - a.bounds[0].fixity_depth          # column height Hcol
    P = mc.axial
    mu_d = max(b.mu_demand for b in a.bounds)

    lines: list[str] = []
    add = lines.append
    add("## Detailed calculations")
    add("")
    add(f"_All equations are shown with substituted values. Code: "
        f"**{a.provisions.name}**; confined-concrete model per Mander, Priestley "
        "& Park (1988). Units: kip, in, ksi (psi where noted)._")
    add("")

    # ------------------------------------------------------------------
    add("### 1 · Confined concrete — Mander model (column)")
    add(f"*Ref: {a.provisions.ref_confined}.*")
    add("")
    dsp = bar_diameter(d.spiral_bar_no)
    asp = bar_area(d.spiral_bar_no) * d.spiral_bundle
    s_clear = d.spiral_spacing - dsp
    add(f"- Concrete modulus:  Ec = 57000·√f'c = 57000·√{d.fc*1000:.0f} "
        f"= {conf.Ec*1000:.0f} psi = {conf.Ec:.0f} ksi")
    add(f"- Core dia. (to spiral C/L):  ds = D − 2·cover − d_sp "
        f"= {d.D:.0f} − 2·{d.cover:.1f} − {dsp:.3f} = {conf.ds:.2f} in")
    add(f"- Transverse steel ratio:  ρs = 4·Asp/(ds·s) "
        f"= 4·{asp:.3f}/({conf.ds:.2f}·{d.spiral_spacing:g}) = {conf.rho_s:.4f}"
        + (f"  (bundled ×{d.spiral_bundle})" if d.spiral_bundle > 1 else ""))
    add(f"- Confinement effectiveness:  ke = (1 − s'/(2·ds))/(1 − ρcc) "
        f"= (1 − {s_clear:.3f}/(2·{conf.ds:.2f}))/(1 − {conf.rho_long:.4f}) = {conf.ke:.3f}")
    add(f"- Effective confining pressure:  f'l = 0.5·ke·ρs·fyh "
        f"= 0.5·{conf.ke:.3f}·{conf.rho_s:.4f}·{d.fyh:.0f} = {conf.fl_eff:.3f} ksi")
    add(f"- Confined strength:  f'cc = f'c·(−1.254 + 2.254·√(1 + 7.94·f'l/f'c) − 2·f'l/f'c) "
        f"= {conf.fcc:.2f} ksi   (f'cc/f'c = {conf.fcc/d.fc:.2f})")
    add(f"- Ultimate confined strain:  εcu = 0.004 + 1.4·ρs·fyh·εsu/f'cc "
        f"= 0.004 + 1.4·{conf.rho_s:.4f}·{d.fyh:.0f}·{conf.eps_su_h:.2f}/{conf.fcc:.2f} "
        f"= {conf.eps_cu:.4f}")
    add("")

    # ------------------------------------------------------------------
    add("### 2 · Flexural capacity — moment-curvature (column)")
    add(f"*Ref: {a.provisions.ref_flexure}, with the elasto-plastic equal-area "
        "idealisation. Fibre integration at constant axial load.*")
    add("")
    add(f"- Axial load:  P = P_dead + column self-weight above hinge "
        f"= {a.axial_entered:.0f} + {a.P_used - a.axial_entered:.1f} "
        f"= {a.P_used:.0f} kip")
    add(f"- First yield:  φ'y = {mc.phi_yield_first:.3e} 1/in,  "
        f"M'y = {mc.M_yield_first/12:.0f} kip-ft")
    add(f"- **Idealised plastic moment (equal-area):  Mp = {mc.Mp/12:.0f} kip-ft "
        f"({mc.Mp:.0f} kip-in)**")
    add(f"- Idealised yield curvature:  φy = Mp/EIeff = {mc.phi_y:.3e} 1/in")
    add(f"- Ultimate curvature ({mc.failure_mode}-controlled):  φu = {mc.phi_u:.3e} 1/in")
    add(f"- Curvature ductility:  μφ = φu/φy = {mc.phi_u/mc.phi_y:.1f}")
    add(f"- Effective flexural rigidity:  EIeff = Mp/φy = {mc.EI_eff:.3e} kip-in²")
    add(f"- Cracked inertia:  Ieff = EIeff/Ec = {a.Ieff_col:.0f} in⁴;  "
        f"Ig = π·D⁴/64 = {a.Ig_col:.0f} in⁴;  Ieff/Ig = {a.Ieff_col/a.Ig_col:.3f}")
    add("")

    # ------------------------------------------------------------------
    add("### 3 · Effective stiffness, yield & plastic displacement")
    add(f"*Ref: {a.provisions.ref_plastic_hinge}; {a.provisions.ref_displacement}. "
        "Plastic hinge in the column at the top of shaft; two-segment equivalent "
        "cantilever fixed at the point of fixity (Df below top of shaft); elastic "
        "flexibility by the unit-load method, M(x) = F·x.*")
    add("")
    add(f"- Column-segment cracked rigidity:  EI_col = Mp/φy = {a.EI_col:.3e} kip-in²")
    add(f"- Shaft-segment cracked rigidity:  EI_shaft = {a.EI_shaft:.3e} kip-in²")
    add(f"- Column height (top of shaft to load point):  L = Le − Df = {L:.0f} in")
    add("")
    add("**(a) Lateral stiffness k** — two flexibility terms (column then shaft):")
    add("")
    add("  f = L³/(3·EI_col) + (Le³ − L³)/(3·EI_shaft)   ;   k = 1/f")
    add("")
    add("| mult | Df (in) | Le (in) | L³/(3·EI_col) | (Le³−L³)/(3·EI_shaft) | f (in/kip) | k = 1/f (kip/in) |")
    add("|---:|---:|---:|---:|---:|---:|---:|")
    for b in a.bounds:
        tc = L ** 3 / (3.0 * a.EI_col)
        ts = (b.Le ** 3 - L ** 3) / (3.0 * a.EI_shaft)
        f = tc + ts
        add(f"| {b.multiplier:g} | {b.fixity_depth:.0f} | {b.Le:.0f} | {tc:.3e} | "
            f"{ts:.3e} | {f:.3e} | {1.0/f:.1f} |")
    add("")
    Fy = mc.Mp / L
    add("**(b) Yield displacement Δy** — lateral force that forms the hinge "
        "(develops Mp at the top of shaft), applied through the elastic "
        "flexibility:")
    add("")
    add(f"  Fy = Mp/L = {mc.Mp:.0f}/{L:.0f} = {Fy:.1f} kip   ;   Δy = Fy·f = Fy/k")
    add("")
    add("| mult | Fy (kip) | k (kip/in) | Δy = Fy/k (in) |")
    add("|---:|---:|---:|---:|")
    for b in a.bounds:
        add(f"| {b.multiplier:g} | {Fy:.1f} | {b.stiffness:.1f} | {b.delta_y:.2f} |")
    add("")
    dbl = bar_diameter(d.long_bar_no)
    lp_raw = 0.08 * L + 0.15 * d.fye * dbl
    lp_min = 0.3 * d.fye * dbl
    add("**(c) Plastic hinge length Lp** (SDC analytical):")
    add(f"- Lp = 0.08·L + 0.15·fye·dbl ≥ 0.3·fye·dbl "
        f"= 0.08·{L:.0f} + 0.15·{d.fye:.0f}·{dbl:.3f} = {lp_raw:.1f} in "
        f"(≥ {lp_min:.1f} in) → **Lp = {a.Lp:.1f} in**")
    add("")
    theta_p = a.Lp * (mc.phi_u - mc.phi_y)
    dp = theta_p * (L - a.Lp / 2.0)
    add("**(d) Plastic displacement Δp** — rigid rotation θp of the hinge acting "
        "through the hinge-to-load arm (L − Lp/2):")
    add(f"- θp = Lp·(φu − φy) = {a.Lp:.1f}·({mc.phi_u:.3e} − {mc.phi_y:.3e}) "
        f"= {theta_p:.4f} rad")
    add(f"- Δp = θp·(L − Lp/2) = {theta_p:.4f}·({L:.0f} − {a.Lp/2:.1f}) "
        f"= **{dp:.2f} in**  (independent of the fixity bound — the arm is the "
        "column height)")
    add("")
    add("**(e) Displacement capacity Δc = Δy + Δp**  and local ductility "
        "μc = Δc/Δy:")
    add("")
    add("| mult | Δy (in) | Δp (in) | Δc = Δy+Δp (in) | μc = Δc/Δy |")
    add("|---:|---:|---:|---:|---:|")
    for b in a.bounds:
        add(f"| {b.multiplier:g} | {b.delta_y:.2f} | {b.delta_p:.2f} | "
            f"{b.delta_c:.2f} | {b.mu_capacity:.2f} |")
    add("")

    # ------------------------------------------------------------------
    add("### 4 · Displacement demand (ESA)")
    add(f"*Ref: {a.provisions.ref_demand}. Effective period from cracked "
        "stiffness; equal-displacement rule Δd = Sa·g·(T/2π)², g = 386.09 in/s².*")
    add("")
    add(f"- Seismic weight (participating mass):  W = W_trib + participation·W_self "
        f"= {a.weight_entered:.0f} + {a.weight_mass - a.weight_entered:.1f} "
        f"= {a.weight_mass:.0f} kip;   m = W/g  (column self-weight W_self "
        f"= {a.W_self:.1f} kip)")
    add("")
    add("| mult | k (kip/in) | T = 2π√(m/k) (s) | Sa (g) | Δd,elastic (in) | "
        "Rd | Δd (in) |")
    add("|---:|---:|---:|---:|---:|---:|---:|")
    for b in a.bounds:
        de = b.demand.disp_elastic or b.demand.disp_demand
        add(f"| {b.multiplier:g} | {b.stiffness:.1f} | {b.demand.period:.2f} | "
            f"{b.demand.Sa:.3f} | {de:.2f} | {b.demand.Rd:.3f} | "
            f"{b.demand.disp_demand:.2f} |")
    add("")
    if any(b.demand.Rd > 1.0 for b in a.bounds):
        add("*Rd = short-period displacement magnification (SGS 4.3.3), "
            "T\\* = 1.25·Ts; solved iteratively with μD.*")
    else:
        add("*Rd = 1.0 (SGS 4.3.3): not a short-period structure, or "
            "essentially elastic (μD → 1).*")
    add("")

    # ------------------------------------------------------------------
    add("### 5 · Shear capacity — column")
    add(f"*Ref: {a.provisions.ref_shear}. Inside the plastic hinge; φ = 0.90.*")
    add("")
    b = shear_breakdown(sec, P, mu_d, inside_hinge=True,
                        provisions=a.provisions)
    Vo = a.Mo / L
    add(f"- Effective shear area:  Ae = 0.8·Ag = 0.8·{sec.Ag:.0f} = {b.Ae:.0f} in²")
    _add_concrete_shear(add, sec, d, b, mu_d, P)
    add(f"- Concrete shear:  Vc = vc·Ae = {b.vc:.3f}·{b.Ae:.0f} = **{b.Vc:.1f} kip**")
    add(f"- Transverse shear:  Vs = (π/2)·Asp·fyh·D'/s "
        f"= (π/2)·{asp:.3f}·{d.fyh:.0f}·{conf.ds:.2f}/{d.spiral_spacing:g} "
        f"= {b.Vs_uncapped:.1f} kip")
    add(f"- Max shear reinf.:  Vs ≤ {a.provisions.vs_max_coeff:g}·√f'c·Ae "
        f"= {a.provisions.vs_max_coeff:g}·√{d.fc:.1f}·{b.Ae:.0f} = {b.Vs_cap:.1f} kip "
        f"→ **Vs = {b.Vs:.1f} kip**  ({a.provisions.ref_max_shear})")
    add(f"- Nominal:  Vn = Vc + Vs = {b.Vn:.1f} kip;  "
        f"**φVn = {PHI_SHEAR}·Vn = {b.phiVn:.1f} kip**")
    add(f"- Demand:  Vo = Mo/L = {a.Mo:.0f}/{L:.0f} = {Vo:.1f} kip  →  "
        f"{'OK' if b.phiVn >= Vo else 'NG'}  (φVn/Vo = {b.phiVn/Vo:.2f})")
    add("")

    # ------------------------------------------------------------------
    add("### 6 · Overstrength & Type II shaft capacity protection")
    add(f"*Ref: {a.provisions.ref_overstrength}. Shaft designed to remain "
        "essentially elastic (capacity protection).*")
    add("")
    m_int = a.bounds[0].shaft_moment_interface
    m_fix = max(b.shaft_moment_fixity for b in a.bounds)
    of = a.provisions.overstrength_factor
    add(f"- Column overstrength moment:  Mo = {of:g}·Mp = {of:g}·{mc.Mp/12:.0f} "
        f"= {a.Mo/12:.0f} kip-ft")
    add(f"- Shaft flexural demand:  interface = Mo = {m_int/12:.0f} kip-ft;  "
        f"fixity-amplified = {m_fix/12:.0f} kip-ft")
    gamma = a.provisions.shaft_demand_factor
    if gamma != 1.0:
        add(f"- Capacity-protection amplification:  γ = {gamma:g} → design "
            f"demand = {gamma:g}·{m_int/12:.0f} = {gamma*m_int/12:.0f} kip-ft "
            f"({a.provisions.ref_shaft_capacity})")
    add(f"- Shaft flexural capacity:  Mn,shaft = {a.mc_shaft.Mp/12:.0f} kip-ft "
        f"(from shaft M-φ at P = {a.mc_shaft.axial:.0f} kip; "
        f"{s.long_label()}, {s.spiral_label()})")
    sh = s.section()
    phiVn_s, Vc_s, Vs_s = shear_capacity(sh, a.mc_shaft.axial, mu_d=1.0,
                                         inside_hinge=False,
                                         provisions=a.provisions)
    label = "α'" if a.provisions.shear_model == "aashto" else "F1"
    add(f"- Shaft shear capacity ({label} = 3.0 outside hinge):  Vc = {Vc_s:.1f}, "
        f"Vs = {Vs_s:.1f},  φVn,shaft = {phiVn_s:.1f} kip  vs  Vo = {Vo:.1f} kip")
    add("")

    # ------------------------------------------------------------------
    add("### 7 · Longitudinal & transverse reinforcement limits")
    add(f"*Ref: {a.provisions.ref_longitudinal}; {a.provisions.ref_transverse}. "
        "Confirm clause numbers against your code copy.*")
    add("")

    # longitudinal limits (pull the limits used from the checks)
    rho_l_min = next((c.demand for c in a.checks
                      if c.name == "Longitudinal steel ratio (min)"), 0.01)
    rho_l_max = next((c.capacity for c in a.checks
                      if c.name == "Longitudinal steel ratio (max)"), 0.04)
    ok_lmin = sec.rho_l >= rho_l_min
    ok_lmax = sec.rho_l <= rho_l_max
    add("**Longitudinal steel (column):**")
    add(f"- Provided:  ρl = Ast/Ag = {sec.Ast:.2f}/{sec.Ag:.0f} = {sec.rho_l:.4f} "
        f"= **{sec.rho_l*100:.2f}%**  ({d.long_label()})")
    add(f"- Minimum:  ρl ≥ {rho_l_min:.3f} ({rho_l_min*100:.1f}%)  →  "
        f"{'OK' if ok_lmin else 'NG'}")
    add(f"- Maximum:  ρl ≤ {rho_l_max:.3f} ({rho_l_max*100:.1f}%)  →  "
        f"{'OK' if ok_lmax else 'NG'}")
    if a.provisions.max_tie_spacing_model == "aashto_8.8.9":
        dbl_r = bar_diameter(d.long_bar_no)
        l_ac = anchorage_length(dbl_r, d.fye, d.fc, d.long_bundle)
        add(f"- Required anchorage into cap beam/footing:  ℓac ≥ "
            f"0.79·dbl·fye/√f'c = {l_ac:.1f} in  (§8.8.4"
            + (", ×1.2 for 2-bar bundles per §8.8.5" if d.long_bundle == 2 else "")
            + "). Informational — a column embedded in an *oversized* shaft is "
            "governed by §8.8.10 instead, which is not evaluated here.")
    add("")

    # transverse (confinement / minimum) — code-specific
    Ag = sec.Ag
    Ac = sec.Acore
    add("**Transverse steel (column spiral/hoop, inside plastic hinge):**")
    add(f"- Provided:  ρs = {sec.rho_s:.4f} = **{sec.rho_s*100:.2f}%**  "
        f"({d.spiral_label()})")
    if a.provisions.transverse_min_model == "caltrans_table":
        rho_s_min, in_table, note = caltrans_min_transverse_ratio(
            d.D, L, sec.rho_l, P, d.fc, sec.Ag)
        add(f"- Minimum (SDC 2.1 Table 5.3.8.2-1, Ordinary Standard):  {note}")
        if in_table:
            add(f"  → ρs,min = **{rho_s_min*100:.2f}%**  →  "
                f"{'OK' if sec.rho_s >= rho_s_min else 'NG'}")
        else:
            add("  → **table does not cover this section**; establish ρs,min "
                "via the PSDC procedure (μc ≥ 3.0). Check flagged, not certified.")
    else:
        c1, c2 = a.provisions.conf_c1, a.provisions.conf_c2
        floor = a.provisions.rho_s_min_floor
        t1 = c1 * (Ag / Ac - 1.0) * d.fc / d.fyh
        t2 = c2 * d.fc / d.fyh
        rho_s_min = max(t1, t2, floor)
        add(f"- Minimum (AASHTO SGS §8.6.5):  ρs ≥ {floor:.3f} = "
            f"{floor*100:.2f}%  →  **{rho_s_min*100:.2f}%**  →  "
            f"{'OK' if sec.rho_s >= rho_s_min else 'NG'}")
    add("- No ρs maximum on the ratio; spiral pitch is limited by clear-spacing / "
        "detailing, and Vs is capped by the max-shear-reinforcement limit (§5).")
    add("")
    add(f"*Shaft (capacity-protected): ρl = {sh.rho_l*100:.2f}% ({s.long_label()}), "
        f"ρs = {sh.rho_s*100:.2f}% ({s.spiral_label()}) — sized for capacity "
        f"protection, not the ductile min/max above.*")
    add("")
    return lines
