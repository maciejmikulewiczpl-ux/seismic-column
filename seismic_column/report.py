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


def _eq(add, lhs, symbolic, substitution, result, ref="", status=None) -> None:
    """Render one equation as *symbolic form* (with a code reference) followed by
    the *substituted numbers → result*, so every value can be checked by hand.

    Produces two lines inside a single Markdown bullet::

        - **lhs** = symbolic            (ref)
          = substitution = **result**   → OK/NG
    """
    ref_txt = f"  &nbsp;*[{ref}]*" if ref else ""
    stat = "" if status is None else ("  →  **OK** ✅" if status else "  →  **NG** ❌")
    add(f"- **{lhs}** = {symbolic}{ref_txt}  ")
    add(f"  = {substitution} = **{result}**{stat}")


def _chk(add, label, symbolic, substituted, ref="", status=None) -> None:
    """Render a pass/fail check: symbolic inequality then substituted → OK/NG."""
    ref_txt = f"  &nbsp;*[{ref}]*" if ref else ""
    stat = "" if status is None else ("  →  **OK** ✅" if status else "  →  **NG** ❌")
    add(f"- **{label}:**  {symbolic}{ref_txt}  ")
    add(f"  {substituted}{stat}")


def _find(checks, name):
    """Return the Check with ``name`` (or ``None``)."""
    return next((c for c in checks if c.name == name), None)


def _add_concrete_shear(add, sec, d, b, mu_d, P) -> None:
    """vc derivation for whichever shear model ran, symbolic then substituted.

    Both models evaluate on the **nominal** f'c (not the expected f'ce used for
    moment-curvature): SDC 2.1 §5.3.7.2 / SGS §8.6.1 specify nominal strength.
    """
    Ag = sec.Ag
    fs_res = (f"{b.fs_raw:.3f} → ≤0.35 = {b.fs:.3f} ksi"
              if b.fs < b.fs_raw else f"{b.fs:.3f} ksi")
    if b.model == "aashto":
        _eq(add, "fs", "ρs·fyh ≤ 0.35 ksi", f"{sec.rho_s:.4f}·{d.fyh:.0f}",
            fs_res, "SGS 8.6.2-6")
        if b.vc == 0.0:
            _eq(add, "vc", "0  (Pc in net tension)", f"Pc = {P:.0f} kip ≤ 0",
                "0 ksi", "SGS 8.6.2-4")
            return
        _eq(add, "α'", "fs/0.15 + 3.67 − μΔ",
            f"{b.fs:.3f}/0.15 + 3.67 − {mu_d:.2f}",
            f"{b.alpha_raw:.2f} → clamp[0.3, 3.0] = {b.alpha:.2f}", "SGS 8.6.2-5")
        _eq(add, "vc",
            "0.032·α'·(1 + Pc/2Ag)·√f'c ≤ min(0.11√f'c, 0.047·α'·√f'c)",
            f"0.032·{b.alpha:.2f}·{b.axial_factor:.3f}·√{d.fc:.1f}",
            f"{b.vc_uncapped:.4f} → ≤{b.vc_cap:.4f} ({b.cap_label}) "
            f"= {b.vc:.4f} ksi", "SGS 8.6.2-3")
        return

    # Caltrans SDC 2.1 §5.3.7.2, evaluated in psi as the clause is written.
    if b.vc == 0.0:
        _eq(add, "vc", "0  (net axial tension)", f"Pc = {P:.0f} kip ≤ 0",
            "0 psi", "SDC 5.3.7.2")
        return
    fc_psi = d.fc * 1000.0
    _eq(add, "fs", "ρs·fyh ≤ 0.35 ksi", f"{sec.rho_s:.4f}·{d.fyh:.0f}",
        fs_res, "SDC 5.3.7.2-5")
    _eq(add, "F1", "fs/0.15 + 3.67 − μd",
        f"{b.fs:.3f}/0.15 + 3.67 − {mu_d:.2f}",
        f"{b.alpha_raw:.2f} → clamp[0.3, 3.0] = {b.alpha:.2f}", "SDC 5.3.7.2-5")
    _eq(add, "F2", "1 + Pc/(2000·Ag) ≤ 1.5",
        f"1 + {P*1000.0:.0f}/(2000·{Ag:.0f})",
        f"{b.axial_factor_raw:.3f} → {b.axial_factor:.3f}", "SDC 5.3.7.2-6")
    _eq(add, "vc", "F1·F2·√f'c ≤ 4√f'c   (psi)",
        f"{b.alpha:.2f}·{b.axial_factor:.3f}·√{fc_psi:.0f}",
        f"{b.vc_uncapped*1000.0:.0f} → ≤{b.vc_cap*1000.0:.0f} "
        f"= {b.vc*1000.0:.0f} psi = {b.vc:.4f} ksi", "SDC 5.3.7.2-3")


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
    """Full equations, each shown symbolically then with substituted numbers and
    a specific code reference, so every value can be verified by hand."""
    a = rr.assessment
    prov = a.provisions
    d = rr.design
    s = rr.shaft
    sec = d.section()
    conf = sec.confined
    mc = a.mc_col
    g = a.governing_bound
    L = a.bounds[0].Le - a.bounds[0].fixity_depth          # column height Hcol
    P = mc.axial
    mu_d = max(b.mu_demand for b in a.bounds)
    is_ct = prov.shear_model == "caltrans"
    fce = sec.fce

    # Code-specific per-equation references.
    R = {
        "fce": "SDC 3.3.6-4" if is_ct else "SGS 8.4.4-1",
        "Ec": "SDC 3.3.6-1" if is_ct else "ACI 318 / AASHTO LRFD 5.4.2.4",
        "mander": "Mander et al. (1988)",
        "ecu": "SDC 3.3.6 / Priestley et al." if is_ct else "SGS 8.4.4",
        "mp": "SDC 5.3.6.3" if is_ct else "SGS 8.5",
        "eieff": "SDC 3.4" if is_ct else "SGS 5.6.2-1",
        "lp": "SDC 5.3.4" if is_ct else "SGS 4.11.6-1",
        "dc": "SDC 5.2" if is_ct else "SGS 4.8.2",
        "mud": "SDC 4.4.1" if is_ct else "SGS 4.9-5",
        "vs": "SDC 5.3.7.3" if is_ct else "SGS 8.6.3-1",
        "vsmax": "SDC 5.3.7.4" if is_ct else "SGS 8.6.4-1",
        "mo": "SDC 4.4.2" if is_ct else "SGS 8.5-1",
        "shaft": "SDC 6.2.5.3" if is_ct else "SGS 8.9",
        "pdelta": "SDC 4.4.4-1" if is_ct else "SGS 4.11.5-1",
        "minv": "SDC 4.4" if is_ct else "SGS 8.7.1-1",
    }

    lines: list[str] = []
    add = lines.append
    add("## Detailed calculations")
    add("")
    add(f"_Code: **{prov.name}**. Each equation is given first **symbolically** "
        "(with its clause reference in brackets), then **with numbers substituted "
        "→ result**. Units: kip, in, ksi (psi where noted). The moment-curvature "
        "/ confined-concrete model uses the **expected** strength f'ce; the shear "
        "and detailing checks use the **nominal** f'c — see each section._")
    add("")

    # ------------------------------------------------------------------
    add("### 1 · Confined concrete — Mander model (column)")
    add(f"*Ref: {prov.ref_confined}. Section response uses the expected strength "
        "f'ce (below), per the code's material model.*")
    add("")
    dsp = bar_diameter(d.spiral_bar_no)
    asp = bar_area(d.spiral_bar_no) * d.spiral_bundle
    s_clear = d.spiral_spacing - dsp
    floor_txt = f", ≥ {prov.fce_floor:g} ksi" if prov.fce_floor else ""
    _eq(add, "f'ce", f"{prov.fce_factor:g}·f'c{floor_txt}",
        f"{prov.fce_factor:g}·{d.fc:.1f}"
        + (f" (≥ {prov.fce_floor:g})" if prov.fce_floor else ""),
        f"{fce:.2f} ksi", R["fce"])
    _eq(add, "Ec", "57000·√f'ce   (psi)", f"57000·√{fce*1000:.0f}",
        f"{conf.Ec*1000:.0f} psi = {conf.Ec:.0f} ksi", R["Ec"])
    _eq(add, "ds", "D − 2·cover − d_sp",
        f"{d.D:.0f} − 2·{d.cover:.1f} − {dsp:.3f}", f"{conf.ds:.2f} in")
    bundle_note = f"  (bundled ×{d.spiral_bundle})" if d.spiral_bundle > 1 else ""
    _eq(add, "ρs", "4·Asp/(ds·s)",
        f"4·{asp:.3f}/({conf.ds:.2f}·{d.spiral_spacing:g})",
        f"{conf.rho_s:.4f}{bundle_note}",
        "SDC 5.3.8.2-1" if is_ct else "SGS 8.6.2-7")
    _eq(add, "ke", "(1 − s'/(2·ds))/(1 − ρcc)",
        f"(1 − {s_clear:.3f}/(2·{conf.ds:.2f}))/(1 − {conf.rho_long:.4f})",
        f"{conf.ke:.3f}", R["mander"])
    _eq(add, "f'l", "0.5·ke·ρs·fyh",
        f"0.5·{conf.ke:.3f}·{conf.rho_s:.4f}·{d.fyh:.0f}",
        f"{conf.fl_eff:.3f} ksi", R["mander"])
    x_conf = conf.fl_eff / fce
    _eq(add, "f'cc",
        "f'ce·(−1.254 + 2.254·√(1 + 7.94·f'l/f'ce) − 2·f'l/f'ce)",
        f"{fce:.2f}·(−1.254 + 2.254·√(1 + 7.94·{x_conf:.4f}) − 2·{x_conf:.4f})",
        f"{conf.fcc:.2f} ksi  (f'cc/f'ce = {conf.fcc/fce:.2f})", R["mander"])
    _eq(add, "εcu", "0.004 + 1.4·ρs·fyh·εsu/f'cc",
        f"0.004 + 1.4·{conf.rho_s:.4f}·{d.fyh:.0f}·{conf.eps_su_h:.2f}/{conf.fcc:.2f}",
        f"{conf.eps_cu:.4f}", R["ecu"])
    add("")

    # ------------------------------------------------------------------
    add("### 2 · Flexural capacity — moment-curvature (column)")
    add(f"*Ref: {prov.ref_flexure}, with the elasto-plastic equal-area "
        "idealisation. Fibre integration at constant axial load, expected "
        "material strengths.*")
    add("")
    _eq(add, "P", "P_dead + column self-weight above hinge",
        f"{a.axial_entered:.0f} + {a.P_used - a.axial_entered:.1f}",
        f"{a.P_used:.0f} kip")
    add(f"- First yield (fibre):  φ'y = {mc.phi_yield_first:.3e} 1/in,  "
        f"M'y = {mc.M_yield_first/12:.0f} kip-ft")
    _eq(add, "Mp", "equal-area idealisation of the M-φ curve",
        "areas balanced beyond first yield",
        f"{mc.Mp/12:.0f} kip-ft ({mc.Mp:.0f} kip-in)", R["mp"])
    _eq(add, "φy", "Mp/EIeff", f"{mc.Mp:.0f}/{mc.EI_eff:.3e}",
        f"{mc.phi_y:.3e} 1/in")
    add(f"- Ultimate curvature ({mc.failure_mode}-controlled):  "
        f"φu = {mc.phi_u:.3e} 1/in")
    _eq(add, "μφ", "φu/φy", f"{mc.phi_u:.3e}/{mc.phi_y:.3e}",
        f"{mc.phi_u/mc.phi_y:.1f}")
    _eq(add, "EIeff", "Mp/φy", f"{mc.Mp:.0f}/{mc.phi_y:.3e}",
        f"{mc.EI_eff:.3e} kip-in²", R["eieff"])
    _eq(add, "Ieff", "EIeff/Ec", f"{mc.EI_eff:.3e}/{conf.Ec:.0f}",
        f"{a.Ieff_col:.0f} in⁴  (Ig = π·D⁴/64 = {a.Ig_col:.0f}; "
        f"Ieff/Ig = {a.Ieff_col/a.Ig_col:.3f})", R["eieff"])
    add("")

    # ------------------------------------------------------------------
    add("### 3 · Effective stiffness, yield & plastic displacement")
    add(f"*Ref: {prov.ref_plastic_hinge}; {prov.ref_displacement}. "
        "Plastic hinge in the column at the top of shaft; two-segment equivalent "
        "cantilever fixed at the point of fixity (Df below top of shaft); elastic "
        "flexibility by the unit-load method, M(x) = F·x.*")
    add("")
    add(f"- Column-segment cracked rigidity:  EI_col = Mp/φy = {a.EI_col:.3e} kip-in²")
    add(f"- Shaft-segment cracked rigidity:  EI_shaft = {a.EI_shaft:.3e} kip-in²")
    add(f"- Column height (top of shaft to load point):  L = Le − Df = {L:.0f} in")
    add("")
    add("**(a) Lateral stiffness k** — symbolic, then per fixity bound:")
    add("")
    add("- **f** = L³/(3·EI_col) + (Le³ − L³)/(3·EI_shaft)   ;   **k** = 1/f  ")
    add("  (flexibility of the column segment plus the shaft segment to fixity)")
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
    add("**(b) Yield displacement Δy** — force that develops Mp at the top of "
        "shaft, applied through the elastic flexibility:")
    add("")
    _eq(add, "Fy", "Mp/L", f"{mc.Mp:.0f}/{L:.0f}", f"{Fy:.1f} kip")
    add("- **Δy** = Fy/k  (per bound below)")
    add("")
    add("| mult | Fy (kip) | k (kip/in) | Δy = Fy/k (in) |")
    add("|---:|---:|---:|---:|")
    for b in a.bounds:
        add(f"| {b.multiplier:g} | {Fy:.1f} | {b.stiffness:.1f} | {b.delta_y:.2f} |")
    add("")
    dbl = bar_diameter(d.long_bar_no)
    lp_raw = 0.08 * L + 0.15 * d.fye * dbl
    lp_min = 0.3 * d.fye * dbl
    add("**(c) Plastic hinge length Lp:**")
    _eq(add, "Lp", "0.08·L + 0.15·fye·dbl  ≥  0.3·fye·dbl",
        f"0.08·{L:.0f} + 0.15·{d.fye:.0f}·{dbl:.3f}",
        f"{lp_raw:.1f} in (≥ {lp_min:.1f}) = {a.Lp:.1f} in", R["lp"])
    add("")
    theta_p = a.Lp * (mc.phi_u - mc.phi_y)
    dp = theta_p * (L - a.Lp / 2.0)
    add("**(d) Plastic displacement Δp** — hinge rotation θp through the "
        "hinge-to-load arm (L − Lp/2):")
    _eq(add, "θp", "Lp·(φu − φy)",
        f"{a.Lp:.1f}·({mc.phi_u:.3e} − {mc.phi_y:.3e})", f"{theta_p:.4f} rad",
        R["lp"])
    _eq(add, "Δp", "θp·(L − Lp/2)",
        f"{theta_p:.4f}·({L:.0f} − {a.Lp/2:.1f})", f"{dp:.2f} in", R["dc"])
    add("")
    add("**(e) Displacement capacity Δc = Δy + Δp**, local ductility μc = Δc/Δy:")
    add("")
    add("| mult | Δy (in) | Δp (in) | Δc = Δy+Δp (in) | μc = Δc/Δy |")
    add("|---:|---:|---:|---:|---:|")
    for b in a.bounds:
        add(f"| {b.multiplier:g} | {b.delta_y:.2f} | {b.delta_p:.2f} | "
            f"{b.delta_c:.2f} | {b.mu_capacity:.2f} |")
    add("")

    # ------------------------------------------------------------------
    add("### 4 · Displacement demand (ESA)")
    add(f"*Ref: {prov.ref_demand}. Effective (cracked) period, equal-displacement "
        "rule, g = 386.09 in/s².*")
    add("")
    _eq(add, "W", "W_trib + participation·W_self",
        f"{a.weight_entered:.0f} + {a.weight_mass - a.weight_entered:.1f}",
        f"{a.weight_mass:.0f} kip  (W_self = {a.W_self:.1f} kip)")
    add("- **m** = W/g  ;  **T** = 2π·√(m/k)  ;  "
        "**Δd** = Sa·g·(T/2π)²  (equal-displacement)  ")
    if prov.short_period_magnification:
        add(f"- **Rd** = (1 − 1/μd)·(T\\*/T) + 1/μd ≥ 1, T\\* = 1.25·Ts  "
            f"&nbsp;*[{'SGS 4.3.3'}]*  (short-period magnification; "
            "**Δd** ← Rd·Δd)")
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
    if not prov.short_period_magnification:
        add("*Rd not applied — this code has no short-period magnification.*")
        add("")

    # governing displacement & ductility demand checks
    add("**Governing displacement checks (worst fixity bound):**")
    _chk(add, "Displacement capacity", "Δc ≥ Δd",
         f"{g.delta_c:.2f} ≥ {g.demand.disp_demand:.2f} in  "
         f"(D/C = {g.demand.disp_demand/g.delta_c:.2f})", R["dc"],
         status=g.delta_c >= g.demand.disp_demand)
    mud_lim = next((c.capacity for c in a.checks
                    if c.name == "Displacement ductility demand"), 5.0)
    _chk(add, "Ductility demand", "μd = Δd/Δy ≤ μd,limit",
         f"{mu_d:.2f} ≤ {mud_lim:g}", R["mud"], status=mu_d <= mud_lim)
    add("")

    # ------------------------------------------------------------------
    add("### 5 · Shear capacity — column (inside plastic hinge)")
    add(f"*Ref: {prov.ref_shear}. φ = 0.90. Uses **nominal** f'c = {d.fc:.1f} ksi "
        "(not f'ce).*")
    add("")
    b = shear_breakdown(sec, P, mu_d, inside_hinge=True, provisions=prov)
    Vo = a.Mo / L
    _eq(add, "Ae", "0.8·Ag", f"0.8·{sec.Ag:.0f}", f"{b.Ae:.0f} in²",
        "SDC 5.3.7.2-2" if is_ct else "SGS 8.6.2-2")
    _add_concrete_shear(add, sec, d, b, mu_d, P)
    _eq(add, "Vc", "vc·Ae", f"{b.vc:.4f}·{b.Ae:.0f}", f"{b.Vc:.1f} kip",
        "SDC 5.3.7.2-1" if is_ct else "SGS 8.6.2-1")
    _eq(add, "Vs", "(π/2)·Asp·fyh·D'/s",
        f"(π/2)·{asp:.3f}·{d.fyh:.0f}·{conf.ds:.2f}/{d.spiral_spacing:g}",
        f"{b.Vs_uncapped:.1f} kip", R["vs"])
    _eq(add, "Vs,max", f"{prov.vs_max_coeff:g}·√f'c·Ae",
        f"{prov.vs_max_coeff:g}·√{d.fc:.1f}·{b.Ae:.0f}",
        f"{b.Vs_cap:.1f} kip → Vs = {b.Vs:.1f} kip", R["vsmax"])
    _eq(add, "φVn", "φ·(Vc + Vs)",
        f"{PHI_SHEAR}·({b.Vc:.1f} + {b.Vs:.1f})", f"{b.phiVn:.1f} kip")
    _eq(add, "Vo", "Mo/L", f"{a.Mo:.0f}/{L:.0f}", f"{Vo:.1f} kip", R["mo"])
    _chk(add, "Column shear", "φVn ≥ Vo",
         f"{b.phiVn:.1f} ≥ {Vo:.1f} kip  (φVn/Vo = {b.phiVn/Vo:.2f})",
         "SDC 5.3.7.2-1" if is_ct else "SGS 8.6.1-1", status=b.phiVn >= Vo)
    add("")

    # ------------------------------------------------------------------
    add("### 6 · Overstrength & Type II shaft capacity protection")
    add(f"*Ref: {prov.ref_overstrength}; {prov.ref_shaft_capacity}. Shaft "
        "capacity-protected (essentially elastic).*")
    add("")
    m_int = a.bounds[0].shaft_moment_interface
    of = prov.overstrength_factor
    _eq(add, "Mo", f"{of:g}·Mp", f"{of:g}·{mc.Mp/12:.0f}",
        f"{a.Mo/12:.0f} kip-ft", R["mo"])
    gamma = prov.shaft_demand_factor
    shaft_dem = gamma * m_int
    _eq(add, "M_D (shaft)", f"γ·Mo   (γ = {gamma:g})",
        f"{gamma:g}·{m_int/12:.0f}", f"{shaft_dem/12:.0f} kip-ft", R["shaft"])
    _chk(add, "Shaft flexure", "Mne,shaft ≥ M_D",
         f"{a.mc_shaft.Mp/12:.0f} ≥ {shaft_dem/12:.0f} kip-ft "
         f"(from shaft M-φ, {s.spiral_label()})", R["shaft"],
         status=a.mc_shaft.Mp >= shaft_dem)
    sh = s.section()
    phiVn_s, Vc_s, Vs_s = shear_capacity(sh, a.mc_shaft.axial, mu_d=1.0,
                                         inside_hinge=False, provisions=prov)
    label = "F1" if is_ct else "α'"
    _chk(add, f"Shaft shear ({label} = 3.0 outside hinge)", "φVn,shaft ≥ Vo",
         f"{phiVn_s:.1f} ≥ {Vo:.1f} kip  (Vc = {Vc_s:.1f}, Vs = {Vs_s:.1f})",
         R["shaft"], status=phiVn_s >= Vo)
    add("")

    # ------------------------------------------------------------------
    add("### 7 · Longitudinal & transverse reinforcement limits")
    add(f"*Ref: {prov.ref_longitudinal}; {prov.ref_transverse}.*")
    add("")
    rho_l_min = next((c.demand for c in a.checks
                      if c.name == "Longitudinal steel ratio (min)"), 0.01)
    rho_l_max = next((c.capacity for c in a.checks
                      if c.name == "Longitudinal steel ratio (max)"), 0.04)
    add("**Longitudinal steel (column):**")
    _eq(add, "ρl", "Ast/Ag", f"{sec.Ast:.2f}/{sec.Ag:.0f}",
        f"{sec.rho_l:.4f} = {sec.rho_l*100:.2f}%  ({d.long_label()})",
        prov.ref_longitudinal)
    _eq(add, "ρl ≥ ρl,min", "minimum longitudinal ratio",
        f"{sec.rho_l:.4f} ≥ {rho_l_min:.3f}", f"{rho_l_min*100:.1f}%",
        status=sec.rho_l >= rho_l_min)
    _eq(add, "ρl ≤ ρl,max", "maximum longitudinal ratio",
        f"{sec.rho_l:.4f} ≤ {rho_l_max:.3f}", f"{rho_l_max*100:.1f}%",
        status=sec.rho_l <= rho_l_max)
    add("")

    add("**Transverse steel (column spiral/hoop, inside plastic hinge):**")
    _eq(add, "ρs (provided)", "4·Asp/(ds·s)",
        f"4·{asp:.3f}/({conf.ds:.2f}·{d.spiral_spacing:g})",
        f"{sec.rho_s:.4f} = {sec.rho_s*100:.2f}%")
    if prov.transverse_min_model == "caltrans_table":
        rho_s_min, in_table, note = caltrans_min_transverse_ratio(
            d.D, L, sec.rho_l, P, d.fc, sec.Ag)
        add(f"- Minimum from **Table 5.3.8.2-1** (Ordinary Standard):  {note}  "
            "&nbsp;*[SDC 5.3.8.2]*")
        if in_table:
            add(f"  → ρs ≥ ρs,min: {sec.rho_s*100:.2f}% ≥ {rho_s_min*100:.2f}%  →  "
                + ("**OK** ✅" if sec.rho_s >= rho_s_min else "**NG** ❌"))
        else:
            add("  → **table does not cover this section**; establish ρs,min via "
                "the PSDC procedure (μc ≥ 3.0). Check flagged, not certified.")
    else:
        floor = prov.rho_s_min_floor
        _eq(add, "ρs ≥ ρs,min", "minimum volumetric ratio (SGS §8.6.5)",
            f"{sec.rho_s:.4f} ≥ {floor:.3f}", f"{floor*100:.2f}%",
            "SGS 8.6.5-3", status=sec.rho_s >= floor)
    add("")
    add(f"*Shaft (capacity-protected): ρl = {sh.rho_l*100:.2f}% "
        f"({s.long_label()}), ρs = {sh.rho_s*100:.2f}% ({s.spiral_label()}).*")
    add("")

    # ------------------------------------------------------------------
    add("### 8 · P-Δ, minimum lateral strength & axial limits")
    add(f"*Ref: {prov.ref_pdelta}; {prov.ref_min_strength}; {prov.ref_max_axial}.*")
    add("")
    pd = a.P_used * g.demand.disp_demand
    pd_cap = prov.pdelta_factor * mc.Mp
    _chk(add, "P-Δ", f"Pdl·Δr ≤ {prov.pdelta_factor:g}·Mp",
         f"{a.P_used:.0f}·{g.demand.disp_demand:.2f} ≤ {prov.pdelta_factor:g}·{mc.Mp:.0f} "
         f"→ {pd:.0f} ≤ {pd_cap:.0f} kip-in", R["pdelta"], status=pd <= pd_cap)
    Vp = mc.Mp / L
    minv = prov.min_strength_factor * a.P_used
    _chk(add, "Min lateral strength",
         f"Vp = Mp/L ≥ {prov.min_strength_factor:g}·Pdl",
         f"{mc.Mp:.0f}/{L:.0f} = {Vp:.1f} ≥ "
         f"{prov.min_strength_factor:g}·{a.P_used:.0f} = {minv:.1f} kip",
         R["minv"], status=Vp >= minv)
    ck_axr = _find(a.checks, "Axial load ratio")
    if ck_axr is not None:
        rho_dl = a.P_used / (min(d.fc, 5.0) * sec.Ag)
        _chk(add, "Axial load ratio", "ρdl = Pdl/(f'c·Ag) ≤ 0.15  (f'c ≤ 5 ksi)",
             f"{a.P_used:.0f}/({min(d.fc,5.0):.1f}·{sec.Ag:.0f}) = "
             f"{rho_dl:.3f} ≤ {ck_axr.capacity:g}", prov.ref_max_axial,
             status=ck_axr.passed)
    ck_axm = _find(a.checks, "Maximum axial load")
    if ck_axm is not None:
        pcap = prov.max_axial_coeff * d.fc * sec.Ag
        _chk(add, f"Max axial load (applies when μd > 2; μd = {mu_d:.2f})",
             "Pu ≤ 0.2·f'c·Ag",
             f"{a.P_used:.0f} ≤ {prov.max_axial_coeff:g}·{d.fc:.1f}·{sec.Ag:.0f} "
             f"= {pcap:.0f} kip", prov.ref_max_axial, status=ck_axm.passed)
    add("")

    # ------------------------------------------------------------------
    add("### 9 · Detailing")
    add(f"*Ref: {prov.ref_detailing}.*")
    add("")
    _detailing_calcs(add, a, d)

    return lines


def _detailing_calcs(add, a, d) -> None:
    """Render each detailing check that ran: symbolic limit + its verified note.

    Values come from the assessment's Check objects so the report always agrees
    with the checks; the symbolic form is written per code where it differs.
    """
    is_ct = a.provisions.max_tie_spacing_model == "caltrans_8.4.1"
    # symbolic form + short reference, keyed by check name
    spacing_sym = ("s ≤ min(6·dbl, 8 in)" if is_ct
                   else "s ≤ min(D/5, 6·dbl, 6 in [8 in bundled])")
    forms = {
        "Transverse spacing (max)": (spacing_sym,
                                     "SDC 8.4.1.1" if is_ct else "SGS 8.8.9"),
        "Longitudinal bar spacing (max)":
            ("s_long ≤ 10 in (Dc ≤ 5 ft) else 12 in", "SDC 8.4.2"),
        "Transverse bar size (min)":
            ("spiral # ≥ #4 (#5 for #10+ or bundled long. bars)", "SGS 8.8.9"),
        "Longitudinal bar diameter (bond)":
            ("dbl,eff ≤ 0.79·√f'c·(L − 0.5·Dc)/fye", "SGS 8.8.6-1"),
        "Shaft confinement (oversized)":
            ("ρs,shaft ≥ 0.5·ρs,col", "SGS 8.8.12"),
    }
    any_shown = False
    for c in a.checks:
        form = forms.get(c.name)
        if form is None:
            continue
        any_shown = True
        _chk(add, c.name, form[0], c.note, form[1], status=c.passed)
    if not any_shown:
        add("- No code-specific detailing checks are enabled for this code.")
