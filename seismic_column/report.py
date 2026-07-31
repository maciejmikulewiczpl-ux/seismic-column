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
from .soil import davisson_fixity_depth


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


_PY_REF = "Matlock (1970); API RP-2A / O'Neill–Murchison (1983); Reese et al. (1974)"


def _add_fixity_source(add, a) -> None:
    """Explain how the point of fixity was obtained (manual multiplier or p-y),
    in the same symbolic-then-numeric style as the other detailed calcs."""
    H_free = a.H_free or (a.bounds[0].Le - a.bounds[0].fixity_depth)
    soil_bounds = [b for b in a.bounds if b.soil_solution is not None]
    is_ct = a.provisions.shear_model == "caltrans"
    # AASHTO LRFD equivalent-fixity equations, adopted by Caltrans as AASHTO-CA BDS
    lrfd_ref = "AASHTO-CA BDS Art. 10.7.3.13.4" if is_ct else "AASHTO LRFD Art. 10.7.3.13.4"

    # --- (i) which method + the fixity depth per bound ---
    if not soil_bounds:
        add("**Point-of-fixity source: assumed multipliers** (user-selected). "
            "No soil model — Df is bracketed as a fraction of the shaft diameter:")
        for b in a.bounds:
            _eq(add, f"Df ({b.multiplier:g}× bound)", "mult · D_shaft",
                f"{b.multiplier:g}·{a.shaft_D or (b.fixity_depth/b.multiplier):.0f}",
                f"{b.fixity_depth:.0f} in", "3× upper- / 6× lower-bound stiffness")
        if is_ct:
            add("An **estimated depth to fixity** (or a soil-spring/LPILE lateral "
                "analysis) is an accepted way to represent shaft flexibility "
                "(Caltrans SDC 2.1 §6.2.6, §C6.2.5.3). The closed-form "
                "depth-to-fixity equations (" + lrfd_ref + ") are "
                "**linear-elastic** and may not suit large deflections without "
                "adjusting soil parameters — use the p-y option when the shaft is "
                "soft/heavily loaded. The 3×/6× bracket spans that uncertainty.")
        else:
            add("An **estimated depth to fixity** is an accepted foundation-"
                "modeling method for pile bents / drilled shafts (AASHTO SGS "
                "Table 5.3.1-1, Foundation Modeling Method I & II; §C5.3.1). The "
                "closed-form depth-to-fixity equations (" + lrfd_ref + ") are "
                "**linear-elastic** and, per §C5.3.1, may not suit large "
                "deflections without adjusting soil parameters — use the p-y "
                "option when the shaft is soft/heavily loaded. The 3×/6× bracket "
                "spans that uncertainty.")
        add("")
        return

    add("**Point-of-fixity source: nonlinear p-y analysis** (user-selected). The "
        "column + Type II shaft are solved as **one continuous beam-column on "
        "layered nonlinear soil springs**; the equivalent depth-to-fixity Df_eq "
        "is the value that reproduces the p-y pile-head flexibility in the "
        "two-segment cantilever. An upper/lower soil-stiffness bracket (×2 / ×0.5 "
        "on the p-y modulus) replaces the assumed 3×/6× pair. "
        f"&nbsp;*[{_PY_REF}]*")
    if is_ct:
        add("This is a code-sanctioned approach: **Caltrans SDC 2.1 §C6.2.5.3** "
            "bases the in-ground shaft moment/shear on soil springs (p-y curves) "
            "and notes the shaft-size iteration this analysis performs; lateral "
            "stability follows §6.2.6, with LPILE as the reference tool. A "
            "**secant** soil stiffness is used here.")
    else:
        add("This is an AASHTO-sanctioned foundation-modeling method: **soil "
            "springs based on P-y curves** are explicitly permitted for pile "
            "bents / drilled shafts (AASHTO SGS Table 5.3.1-1, Foundation "
            "Modeling Method II) and §C5.3.1 calls them *\"a better "
            "representation\"* than an estimated depth to fixity, using **secant "
            "stiffness** — as done here.")
    add("")

    # --- (ii) applied pile-head forces, clearly prescribed ---
    Fy = a.mc_col.Mp / H_free
    Vo = a.Mo / H_free
    add("**Applied pile-head forces** — a lateral point load is applied at the "
        "**column top**; the moment at the top of shaft (= V·H_free) then arises "
        "from the continuous beam, so the shaft carries **both** shear and the "
        "column overturning moment:")
    _eq(add, "F_y  (stiffness / fixity solve)", "Mp / H_free",
        f"{a.mc_col.Mp:.0f}/{H_free:.0f}",
        f"{Fy:.0f} kip  → develops Mp = {a.mc_col.Mp/12:.0f} kip-ft at top of shaft")
    _eq(add, "Vo  (in-ground shaft-design solve)", "Mo / H_free",
        f"{a.Mo:.0f}/{H_free:.0f}",
        f"{Vo:.0f} kip  → develops Mo = {a.Mo/12:.0f} kip-ft (overstrength) at top of shaft")
    add("")

    # --- (iii) equivalent depth-to-fixity per soil bound ---
    add("**Equivalent depth-to-fixity** per soil-stiffness bound "
        "(f_soil = y_head / F_y from the p-y solve; Df_eq inverts the cantilever "
        "flexibility f = H_free³/3EI_col + ((H_free+Df)³ − H_free³)/3EI_shaft):")
    for b in soil_bounds:
        s = b.soil_solution
        if not s.stable:
            _chk(add, f"Fixity — {b.soil_label}", "p-y solution physical?",
                 "**unstable** (soil too soft / shaft too short for F_y; possible "
                 "P-Δ instability) → increase embedment or shaft size, or improve "
                 "the soil", status=False)
            continue
        _eq(add, f"Df_eq — {b.soil_label}", "invert f_soil = y_head/F_y",
            f"y_head={s.head_deflection:.2f} in, f_soil={s.head_flexibility:.3e} in/kip, "
            f"k_head={s.head_stiffness:.0f} kip/in",
            f"Df_eq = {b.fixity_depth:.0f} in = {b.multiplier:.1f}·D_shaft")
    add("")

    # --- (iii-b) closed-form linear cross-check (AASHTO LRFD 10.7.3.13.4) ---
    prof0 = a.soil_profile
    stable_bounds = [b for b in soil_bounds if b.soil_solution.stable]
    if prof0 is not None and stable_bounds and a.shaft_D:
        D = a.shaft_D
        EI_shaft = a.EI_shaft
        lf = davisson_fixity_depth(EI_shaft, prof0, D)
        lo = min(b.fixity_depth for b in stable_bounds)
        hi = max(b.fixity_depth for b in stable_bounds)
        add("**Closed-form cross-check** — the code-referenced *linear-elastic* "
            "equivalent depth to fixity (" + lrfd_ref + ", Davisson & "
            "Robinson 1965): relative-stiffness length R = (EI/k_h)^¼ (clay) or "
            "T = (EI/n_h)^⅕ (sand), then Lf ≈ 1.4R / 1.8T on the initial soil "
            "modulus:")
        _eq(add, "Lf (closed form, linear)", "1.4·R  or  1.8·T",
            f"EI_shaft={EI_shaft:.2e} kip-in², initial-modulus profile",
            f"Lf ≈ {lf:.0f} in = {lf/D:.1f}·D_shaft")
        add(f"The nonlinear p-y solve gives Df_eq = {lo:.0f}–{hi:.0f} in "
            f"({lo/D:.1f}–{hi/D:.1f}·D_shaft) versus the closed-form "
            f"{lf/D:.1f}·D_shaft. **Why they differ** (the two are different "
            "equivalence definitions, so either can be larger): (1) the closed "
            "form uses the *initial* (small-strain) soil modulus, the p-y solve "
            "the **secant** modulus at the actual head force F_y; (2) Davisson's "
            "depth is calibrated to match a *shear-loaded* pile-head deflection, "
            "whereas the p-y Df_eq reproduces the full head flexibility including "
            "the column overturning moment V·H_free; (3) the closed form collapses "
            "the layered profile to one equivalent modulus, the p-y solve honors "
            "each layer; (4) the ×2/×0.5 modulus bracket on Df_eq spans the soil "
            "stiffness uncertainty the single closed-form value cannot capture. "
            + ("The linear closed form is only a **sanity check** (same order of "
               "magnitude confirms the model); the nonlinear p-y (LPILE-equivalent) "
               "value governs the design, consistent with Caltrans SDC §C6.2.5.3."
               if is_ct else
               "Per AASHTO SGS §C5.3.1 the linear equations *\"may not be "
               "appropriate for large deflections\"* — so the closed form is a "
               "**sanity check** (same order of magnitude confirms the model), and "
               "the nonlinear p-y value governs the design."))
        add("")

    # --- (iv) how the p-y curves were developed (per layer) ---
    prof = a.soil_profile
    if prof is not None:
        D = a.shaft_D
        add("**p-y curve development** — each depth gets its own nonlinear curve "
            "(seismic ⇒ cyclic branches). Ultimate resistance pu and the curve "
            "shape by model:")
        add("- clay: **pu = min(3 + σ′v/su + 0.5·z/D, 9)·su·D**; "
            "**p = 0.5·pu·(y/y50)^(1/3)**, y50 = 2.5·ε50·D (Matlock)  "
            f"&nbsp;*[{_PY_REF}]*")
        add("- sand: **pu = min[(C1·z + C2·D), C3·D]·σ′v** (C1–C3 from φ′); "
            "**p = 0.9·pu·tanh(k·z·y/(0.9·pu))** (API, cyclic A=0.9)")
        add("- σ′v = effective overburden (buoyant unit weight below the water "
            "table). Solved by **secant iteration** (Es = p/y) — the full "
            "nonlinear curve, not a bilinear simplification.")
        add("")
        add("| layer | model | mid-depth (ft) | γ′ (pcf) | σ′v (ksi) | strength | "
            "pu (kip/in) |")
        add("|---:|:--|---:|---:|---:|:--|---:|")
        for i, (lyr, top) in enumerate(zip(prof.layers, prof._tops), 1):
            zc = top + 0.5 * lyr.thickness
            pu = prof.p_ult(zc, D)
            sv = prof.sigma_v_eff(zc)
            gamma_pcf = lyr.gamma_eff / (1.0 / (1000.0 * 1728.0))
            if lyr.is_clay:
                strength = (f"su={lyr.su_at(0.5*lyr.thickness)*144:.2f} ksf, "
                            f"ε50={lyr.eps50:g}")
            elif lyr.py_model == "api_sand":
                strength = f"φ′={lyr.phi:.0f}°, k={lyr.k_py*1000:.0f} pci"
            else:
                strength = "elastic"
            add(f"| {i} | {lyr.py_model} | {zc/12:.1f} | {gamma_pcf:.0f} | "
                f"{sv:.4f} | {strength} | {pu:.3f} |")
        add("")
        add("*p vs y curves at each depth are exported (CSV) for use as springs "
            "in the global structural model.*")
        add("")

    # --- (v) in-ground shaft design (bending & shear along depth, at Vo) ---
    ig = a.inground_solution
    if ig is not None and a.inground_moment > 0:
        gamma = a.provisions.shaft_demand_factor
        # in-ground demand is solved at Mo, so it is direction-independent —
        # read it off the envelope rather than the selected view
        m_chk = _find(a.checks, "Shaft flexure in-ground (p-y)")
        v_chk = _find(a.checks, "Shaft shear in-ground (p-y)")
        phiVn = v_chk.capacity if v_chk else 0.0
        add("**In-ground shaft design (bending & shear along the depth)** — with "
            "Vo applied (column at overstrength Mo), the p-y solve gives the "
            "moment & shear the shaft carries **below ground**. The peak moment "
            "is usually **below** the interface, so an interface-only check "
            "under-predicts the shaft demand:")
        _eq(add, "Max in-ground M", "from p-y at Vo (overstrength)",
            f"{a.inground_moment/12:.0f} kip-ft at {ig.max_moment_depth/12:.1f} ft "
            f"below ground",
            f"{a.inground_moment/a.Mo:.2f}·Mo (interface Mo = {a.Mo/12:.0f} kip-ft)")
        _chk(add, "Shaft flexure in-ground", f"{gamma:g}·M ≤ Mne,shaft",
             f"{gamma*a.inground_moment/12:.0f} ≤ {a.mc_shaft.Mp/12:.0f} kip-ft",
             a.provisions.ref_shaft_capacity,
             status=(m_chk.passed if m_chk else None))
        _chk(add, "Shaft shear in-ground", "V ≤ φVn,shaft",
             f"{a.inground_shear:.0f} ≤ {phiVn:.0f} kip "
             f"(interface Vo = {Vo:.0f} kip)", a.provisions.ref_shaft_capacity,
             status=(v_chk.passed if v_chk else None))
        add(f"- Overstrength pile-head deflection = {ig.head_deflection:.1f} in. "
            "Deflection/shear/moment diagrams and their CSV are in the app "
            "drill-down; shaft reinforcement may be curtailed where demand drops.")
        add("")


def _add_rd_derivation(add, a) -> None:
    """Explain the SGS 4.3.3 short-period magnification when it is applied.

    Shows Ts / T* and the Rd fixed-point result for each bound that is actually
    magnified (Rd > 1); otherwise states why Rd = 1.
    """
    magnified = [b for b in a.bounds if b.demand.Rd > 1.0 + 1e-9]
    if not magnified:
        add("*Rd = 1.0 for every bound: not short-period (T ≥ T\\* = 1.25·Ts), "
            "or essentially elastic (μd → 1), or a tabular spectrum with no "
            "resolvable corner period.*")
        add("")
        return
    add("**Short-period magnification (SGS 4.3.3) — how Δd was increased:**")
    add("- **Ts** = SD1/SDS, with SDS = 0.9·max(Sa) and SD1 = 0.9·max(T·Sa) over "
        "1–5 s (the 0.9 factors cancel, so **Ts = max(T·Sa)/max(Sa)**); "
        r"**T\* = 1.25·Ts**  &nbsp;*[SGS 4.3.3 / Art. 3.5]*")
    add(r"- **Rd** = (1 − 1/μd)·(T\*/T) + 1/μd, solved with μd = Δd/Δy "
        "(Rd and μd are interdependent → fixed-point iteration)")
    for b in magnified:
        dm = b.demand
        t_star = 1.25 * dm.Ts
        _eq(add, f"Rd (mult {b.multiplier:g})",
            r"(1 − 1/μd)·(T\*/T) + 1/μd",
            f"(1 − 1/{dm.mu_for_Rd:.2f})·({t_star:.3f}/{dm.period:.3f}) "
            f"+ 1/{dm.mu_for_Rd:.2f}",
            f"{dm.Rd:.3f}  (Ts = {dm.Ts:.3f} s, T\\* = {t_star:.3f} s > "
            f"T = {dm.period:.3f} s)")
        _eq(add, f"Δd (mult {b.multiplier:g})", "Rd·Δd,elastic",
            f"{dm.Rd:.3f}·{dm.disp_elastic:.2f}", f"{dm.disp_demand:.2f} in")
    add("")


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


def _checks_for_view(a, view: str):
    """The check list for ``view``; falls back to the envelope if unavailable."""
    if view != "envelope" and view in a.directions:
        return a.directions[view].checks
    return a.checks


def column_report(rr: RowResult, view: str = "envelope") -> str:
    """Markdown calculation report for one column result.

    ``view`` selects which check list the whole document is written against:
    ``"envelope"`` (the worse of both directions — what the row actually passes
    or fails on), or a single direction by name.  Only the CHECKS differ; the
    section, moment-curvature, p-y and shaft-protection numbers are
    direction-independent and identical in every view.
    """
    a = rr.assessment
    d = rr.design
    s = rr.shaft
    checks = _checks_for_view(a, view)
    lines: list[str] = []
    add = lines.append

    add(f"# Seismic Column Report — {rr.name}")
    add("")
    add(f"**Result:** {'PASS ✅' if rr.feasible else 'FAIL ❌'}  "
        f"({'optimised' if rr.optimized else 'as-entered'})")
    add(f"**Design code:** {a.provisions.name}")
    if len(a.directions) > 1:
        add(f"**Checks shown:** {view}"
            + ("  — the worse of both directions, which is what this row "
               "passes or fails on" if view == "envelope"
               else f"  — the {view} direction ALONE. The row's verdict is the "
                    f"envelope of both; this view is for inspection."))
    add("")

    add("## Geometry")
    add("")
    add(f"- Entered column height Hcol = {a.Hcol_entered/12:.1f} ft "
        f"({a.Hcol_entered:.0f} in)")
    if a.silo > 0:
        add(f"- Column silo (isolation casing) = {a.silo/12:.1f} ft "
            f"({a.silo:.0f} in) — lowers the top of shaft by this much")
        add(f"- **Free column length H_free = Hcol + silo = "
            f"{a.H_free/12:.1f} ft ({a.H_free:.0f} in)**  "
            f"&nbsp;*[{a.provisions.ref_balance_tuning}]*")
        add("  All mechanics below (Lp, self-weight, Δy, Δp, Vo, Vp, P-Δ, the "
            "L/Dc aspect ratio and the bond bar-diameter limit) use H_free; the "
            "plastic hinge stays at the top of shaft, i.e. the bottom of the "
            "silo.  The embedded shaft length is unchanged, so the tip goes "
            "deeper, and the p-y springs start at the bottom of the silo.")
    else:
        add(f"- No column silo, so the free column length H_free = Hcol = "
            f"{a.H_free/12:.1f} ft")
    add(f"- Shaft diameter D_shaft = {s.D:.0f} in")
    add("")

    from .io_schema import TRANSVERSE as _TRANSVERSE

    bent = getattr(rr, "bent", None)
    if bent is not None and bent.multi:
        add("## Bent — multiple columns")
        add("")
        from .io_schema import head_moment_connection as _hmc
        _cap = str(getattr(rr, "cap_fixity", "fixed") or "fixed")
        _moment = _hmc(_cap, _TRANSVERSE)
        add(f"**{bent.n_columns} columns at {bent.spacing/12:.1f} ft centres**, "
            f"cap detail `{_cap}`.")
        add("")
        if _moment:
            add("Transversely this is a portal frame, not a row of cantilevers, "
                "and two things follow. The cap restrains the column heads, so "
                "each column is **fixed-fixed** and develops the two-hinge "
                "mechanism shear `2·Mp/H`. And pushing the bent overturns it, "
                "which is resisted by an axial **couple** between the columns "
                "— so the same section has a different `Mp`, `Vo`, `Df`, `Δy` "
                "and `Δc` at each position.")
        else:
            add("The columns are **pinned to the cap transversely**, so the cap "
                "carries no moment into them: each column is a **cantilever** "
                "on the one-hinge mechanism `Mp/H`, and there is **no push/pull "
                "couple** — statics alone gives `V·H = Σ Mo` with the base "
                "moments taking all of it. Every column therefore sits at the "
                "dead-load axial, and none of them goes into seismic tension. "
                "The positions below are reported for completeness and are "
                "identical by construction.")
            add("")
            add("This is a *choice*, not a saving: it must be detailed as a "
                "genuine pin transversely, and the cap and joint designed for "
                "it.")
        add("")
        add("Longitudinally there is no couple at all: the columns stand at one "
            "station, so they act as "
            f"{bent.n_columns} identical members in parallel at the dead-load "
            "axial, with the end condition `deck_link` gives.")
        add("")
        if _moment:
            _eq(add, "Overturning taken by the couple", "Σ Mo_i",
                " + ".join(f"{p.assessment.Mo/12:.0f}" for p in bent.positions),
                f"{bent.M_overturn/12:.0f} kip-ft",
                ref="cut at the top of shaft: V_bent·H_free = Σ ΔP·x + Σ Mo, and "
                    "V_bent = Σ 2·Mo/H_free, so the couple carries Σ Mo — the "
                    "column base moments take the other half")
            _eq(add, "Axial couple", "ΔPᵢ = M_ot · xᵢ / Σxⱼ²",
                f"{bent.M_overturn/12:.0f} · x / Σx²",
                f"±{bent.delta_P:.0f} kip",
                ref="linear (plane-sections) distribution; sums to zero, so it "
                    "adds no net axial to the bent")
        else:
            _eq(add, "Overturning, taken entirely by the base moments",
                "Σ Mo_i",
                " + ".join(f"{p.assessment.Mo/12:.0f}" for p in bent.positions),
                f"{bent.M_overturn/12:.0f} kip-ft",
                ref="a transverse pin carries no moment into the column head, so "
                    "the base moments resist all the overturning and nothing is "
                    "left for an axial couple: ΔP = 0")
        add(f"- **V_bent** = Σ Voᵢ = **{bent.V_bent:.0f} kip** at overstrength "
            f"({'two hinges' if _moment else 'one hinge'} per column)")
        add("")
        add("| Position | x (ft) | ΔP (kip) | P (kip) | Mp (kip-ft) | Vo (kip) "
            "| Df (ft) | Δy (in) | Δc (in) | Δd (in) | Δc/Δd | |")
        add("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
        for p in bent.positions:
            pa = p.assessment
            g = pa.directions[_TRANSVERSE].governing_bound
            r = (g.delta_c / g.demand.disp_demand
                 if g.demand.disp_demand > 0 else float("nan"))
            add(f"| {p.label} | {p.x/12:+.1f} | {p.delta_P:+.0f} "
                f"| {p.axial:.0f} | {pa.mc_col.Mp/12:.0f} "
                f"| {pa.directions[_TRANSVERSE].Vo:.0f} "
                f"| {g.fixity_depth/12:.1f} | {g.delta_y:.2f} | {g.delta_c:.2f} "
                f"| {g.demand.disp_demand:.2f} | **{r:.2f}** "
                f"| {'NET TENSION ⚠' if p.net_tension else ''} |")
        add("")
        if _moment:
            _n = bent.n_columns
            _s = bent.spacing
            _xmax = max(abs(p.x) for p in bent.positions)
            _sx2 = sum(p.x ** 2 for p in bent.positions)
            _Mo = bent.M_overturn / _n
            _wind = bent.positions[0]
            _P_dead = _wind.axial - _wind.delta_P
            add("**Why the couple does not care how long the column is.** Cut "
                "at the top of shaft and take moments; each column separately "
                "gives `Vᵢ·H_free = M_topᵢ + M_baseᵢ`, so the two combine to")
            add("")
            add("```")
            add("Σ ΔP·x = V·H_free − Σ M_base = Σ M_top")
            add("```")
            add("")
            add("The couple **is** the sum of the column top moments — nothing "
                "else. Elastically `M_top = V·(H_free − a)` and it grows as the "
                "base softens, because a deeper point of fixity drops the "
                "contraflexure point `a`: a rigid base gives `V·H_free/2`, a "
                "fully flexible one `V·H_free`, so up to twice the couple. But "
                "`M_top` cannot exceed `Mo`. Once the head hinges it saturates, "
                "and `H_free` drops out entirely — which is why the value below "
                "is length-independent.")
            add("")
            _eq(add, f"Extreme-column couple (n = {_n}, evenly spaced)",
                "ΔP = n·Mo·x_max / Σxⱼ²",
                f"{_n} · {_Mo/12:.0f} · {_xmax/12:.1f} / {_sx2/144:.0f}",
                f"{bent.delta_P:.0f} kip",
                ref=f"= {2.0 if _n == 2 else _n * _xmax / _sx2 * _s:.2f}·Mo/s "
                    f"for this layout — only Mo and the SPACING move it")
            if _P_dead > 0:
                _s_req = (_n * _Mo * _xmax / _sx2) * _s / _P_dead
                if _s_req > _s:
                    add(f"- Windward column is in **net tension** "
                        f"({_wind.axial:+.0f} kip on {_P_dead:.0f} kip dead "
                        f"load). Keeping it in compression needs "
                        f"**{_s_req/12:.1f} ft** spacing — "
                        f"**{(_s_req - _s)/12:.1f} ft more** than the "
                        f"{_s/12:.1f} ft used — or a proportionally lower `Mo`. "
                        f"Lengthening the column does **not** help.")
                else:
                    add(f"- Windward column stays in compression "
                        f"({_wind.axial:+.0f} kip); tension would start below "
                        f"**{_s_req/12:.1f} ft** spacing.")
            add("")
        _envelope = (f"The checks below are the **envelope** over the positions "
                     f"(transverse) and the dead-load run (longitudinal); "
                     f"**{bent.governing.label}** governs the displacement "
                     f"capacity.")
        if _moment:
            _envelope += (f" The push/pull "
                          f"{'converged' if bent.converged else 'did NOT converge'}"
                          f" in {bent.iterations} pass(es).")
        add(_envelope)
        add("")
        for m in bent.log:
            add(f"- {m}")
        add("")
        add("**Not checked here.** The cap beam itself — its flexure, its shear, "
            "and the column-to-cap joint (SDC §7.4). A multi-column bent needs "
            "all three and none of them is covered by this report; the columns "
            "and their shafts are.")
        add("")

    if len(a.directions) > 1:
        add("## Direction")
        add("")
        add("**Both directions are checked; this column passes only if both "
            "do.** Capacity is direction-independent here — the section is "
            "axisymmetric and the p-y solves run at `F_y = Mp/H_free` and "
            "`Vo = Mo/H_free`, neither of which contains the mass. So Δy, Δp, "
            "Δc, Lp, Df, Mo, Vo, every p-y diagram, the in-ground shaft demand "
            "and all detailing are stated **once** below and hold both ways. "
            "Only the tributary mass differs, and with it T, Sa, Δd, μd, P-Δ "
            "and any shear capacity that degrades with μd.")
        add("")
        add("| Direction | End cond. | W entered (kip) | W + self-wt | Df (ft) "
            "| Df/D | Le (ft) | T (s) | Sa (g) | Δd (in) | Δc (in) | Δc/Δd "
            "| μd | |")
        add("|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
        for dname, dres in a.directions.items():
            g = dres.governing_bound
            ef = ("fixed-fixed" if dres.end_fixity == "fixed" else "fixed-free")
            add(f"| {dname} | {ef} | {dres.weight_entered:.0f} "
                f"| {dres.weight_mass:.0f} | {g.fixity_depth/12:.1f} "
                f"| {g.fixity_depth/a.shaft_D:.2f} | {g.Le/12:.1f} "
                f"| {g.demand.period:.3f} "
                f"| {g.demand.Sa:.4f} | {g.demand.disp_demand:.2f} "
                f"| {g.delta_c:.2f} "
                f"| {g.delta_c / g.demand.disp_demand:.2f} "
                f"| {g.mu_demand:.2f} "
                f"| {'PASS ✅' if dres.passed else 'FAIL ❌'} |")
        add("")
        _dfs = {d: o.governing_bound.fixity_depth for d, o in a.directions.items()}
        if len(set(round(v, 3) for v in _dfs.values())) > 1:
            add("**The depth to fixity differs by direction here**, which it "
                "does whenever the end condition does. Df is not a property of "
                "the ground alone: a fixed-fixed member develops the two-hinge "
                "mechanism shear `2·Mp/H` — twice the cantilever's — so the "
                "soil is driven further along its p-y curve and the fixity "
                "migrates deeper; and the equivalence itself changes, because "
                "the depth is the one reproducing the member's **fixed-fixed** "
                "flexibility `C − B²/A` rather than the cantilever's `C`. "
                "**Use the value for the direction you are modelling** — these "
                "are the equivalent fixed-base cantilever lengths for a "
                "pushover.")
        else:
            add("Df is the same in both directions here, which is correct when "
                "the end condition is: this bent is a fixed-free cantilever "
                "either way, and the p-y solve that sets Df runs at "
                "`F_y = Mp/H_free`, which contains no mass and no directional "
                "term.")
        add("")
        names = sorted({c.name for c in a.checks})
        diff = []
        for n in names:
            per = {dn: next((c for c in o.checks if c.name == n), None)
                   for dn, o in a.directions.items()}
            vals = [c.ratio for c in per.values() if c is not None]
            if len(vals) > 1 and abs(max(vals) - min(vals)) > 1e-9:
                diff.append((n, per))
        if diff:
            add(f"**Checks that differ by direction ({len(diff)} of "
                f"{len(names)}).** The rest come out numerically identical "
                "because they are driven by Mp/Mo and geometry, not mass.")
            add("")
            add("| Check | " + " | ".join(f"{dn} D/C" for dn in a.directions)
                + " | Governs |")
            add("|:--|" + "--:|" * len(a.directions) + ":--|")
            for n, per in diff:
                worst = max(per, key=lambda dn: per[dn].ratio)
                add(f"| {n} | "
                    + " | ".join(f"{per[dn].ratio:.4f}" for dn in a.directions)
                    + f" | {worst} |")
            add("")
        add("The **Checks** section below is the ENVELOPE of the two — the "
            "worse of each named check, with the governing direction noted "
            "where they disagree.")
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

    add("## SDC checks"
        + (f" — {view}" if len(a.directions) > 1 else ""))
    add("")
    add("| Check | Demand | Capacity | D/C | Status |")
    add("|:--|--:|--:|--:|:--:|")
    for c in checks:
        add(f"| {c.name} | {c.demand:.1f} | {c.capacity:.1f} | {c.ratio:.2f} | "
            f"{'PASS' if c.passed else 'FAIL'} |")
    add("")
    for c in checks:
        add(f"- *{c.name}*: {c.note}")
    add("")

    lines.extend(_detailed_calcs(rr, checks))

    if rr.log:
        add("## Optimiser log")
        add("")
        for entry in rr.log:
            add(f"- {entry}")
        add("")

    return "\n".join(lines)


def balance_report(balance) -> str:
    """Markdown calc sheet for the balance checks.

    ``balance`` is a :class:`~seismic_column.balance.BalanceResult`.  Covers the
    whole run, not one column, so it is a separate document from
    :func:`column_report`.
    """
    from .balance import (END_CONDITION_NOTE, GEOMETRY_CHECK,
                          STIFFNESS_ANY_CHECK, STIFFNESS_CHECK)
    from .io_schema import DIRECTIONS, LONGITUDINAL, TRANSVERSE

    cr = balance.criteria
    kap = cr.kappa_symbol
    lines: list[str] = []
    add = lines.append

    add("# Balanced stiffness & balanced frame geometry")
    add("")
    _tha = balance.needs_tha
    _verdict = ("FAIL ❌" if balance.failed else
                ("PASS ✅ — with time-history required" if _tha else "PASS ✅"))
    add(f"**Result:** {_verdict}"
        + ("" if balance.converged else "  —  *did not converge*"))
    add("")
    add("**Scope.** The two rules act at different levels. Balanced *stiffness* "
        "compares bents **inside one frame**, so it has nothing to say about a "
        "run of simply supported spans — each of those is its own single-bent "
        "frame. Balanced *frame geometry* compares **adjacent frames** and "
        "applies everywhere. Both are evaluated **longitudinally and "
        "transversely**, because the tributary mass a bent restrains differs by "
        "direction and so do the periods. A pier with a blank `frame` is left "
        "out of the checks entirely.")
    add("")
    add("**Criteria.**")
    _chk(add, "Balanced stiffness — adjacent bents in a frame",
         f"min(κi, κj) / max(κi, κj) ≥ {cr.k_ratio_min:.2f},  κ = {kap}",
         "evaluated at every fixity bound, like-for-like (stiff-vs-stiff, "
         "soft-vs-soft)", ref=cr.ref_stiffness)
    _chk(add, "Balanced stiffness — any two bents in a frame",
         f"min(κi, κj) / max(κi, κj) ≥ {cr.k_ratio_any:.2f}",
         "reported for NON-adjacent pairs only: an adjacent pair is also 'any "
         f"two', but its {cr.k_ratio_min:.2f} limit is stricter, so the looser "
         "check can never govern there", ref=cr.ref_stiffness_any)
    _chk(add, "Balanced frame geometry — adjacent frames",
         f"min(Ti, Tj) / max(Ti, Tj) ≥ {cr.T_ratio_min:.2f}",
         "T is the FRAME period, not a bent's", ref=cr.ref_geometry)
    add("")
    if cr.mass_normalized:
        add(f"*Note:* with κ = k/m and T = 2π√(m/k), (Ti/Tj)² = κj/κi, so a "
            f"stiffness ratio of {cr.k_ratio_min:.2f} gives a period ratio of "
            f"√{cr.k_ratio_min:.2f} = {math.sqrt(cr.k_ratio_min):.3f} ≥ "
            f"{cr.T_ratio_min:.2f}. Within one frame the stiffness rule "
            "therefore implies the geometry rule; between frames it does not, "
            "because the frame period comes from ΣK and ΣM.")
    else:
        add("*Note:* κ = k (the constant-width form), so the stiffness and "
            "period rules are independent. Mass still enters the period rule "
            "through T = 2π√(m/k) — switching normalisation off removes mass "
            "from §7.1.2, never from §7.1.3.")
    add("")

    # ------------------------------------------------------------------
    add("## Frames")
    add("")
    add("A frame is the set of bents that act together in that direction, "
        "derived from each bent's `deck_link`: an `integral` bent (monolithic, "
        "fixed moment connection) resists both ways; a `bearing` bent released "
        "longitudinally but shear-keyed joins the frame transversely only, and "
        "stands alone longitudinally holding whatever span it is fixed to; a "
        "`free` bent resists neither.")
    add("")
    add("Frame period is the frame's, on the rigid-deck stand-alone "
        "idealisation — the members deflect together, so their stiffnesses add:")
    add("")
    _eq(add, "T_frame", "2π · √(M_frame / K_frame)",
        "K_frame = Σ kᵢ and M_frame = Σ mᵢ over the members that resist in this "
        "direction", "one period per frame, per fixity bound",
        ref=cr.ref_geometry)
    add("")
    add(f"*{END_CONDITION_NOTE}*")
    add("")

    # --- the two frame layouts side by side, so the split is unmissable ---
    add("**The two directions do not see the same structure.** Which bents act "
        "together is derived from `deck_link`, so a bearing released "
        "longitudinally but shear-keyed transversely joins the frame one way "
        "and stands alone the other:")
    add("")
    add("```")
    for d in DIRECTIONS:
        layout = "  |  ".join(
            ("[" + " ".join(fr.names) + "]") if fr.continuous else fr.names[0]
            for fr in balance.frames.get(d, []))
        add(f"{d:14s} {layout}")
    add("```")
    add("")
    add("Square brackets mark a **continuous** frame — the only place a "
        "balanced-stiffness rule applies. Everything else is a single-bent "
        "frame, compared to its neighbours on period alone.")
    add("")

    n_b = max((len(b.k) for b in balance.bents), default=0)
    for direction in DIRECTIONS:
        frames = balance.frames.get(direction, [])
        if not frames:
            continue
        add(f"### {direction.capitalize()}")
        add("")
        hdr = "| Frame | Bents | End condition | M (kip·s²/in) |"
        sep = "|:--|:--|:--|--:|"
        for i in range(n_b):
            lbl = balance.bents[0].label(i) if balance.bents else f"bound {i+1}"
            hdr += f" K [{lbl}] (kip/in) | T [{lbl}] (s) |"
            sep += "--:|--:|"
        add(hdr)
        add(sep)
        shown_share = False
        for f in frames:
            names = " + ".join(f"**[{n}]**" if f.shared(n) else n
                               for n in f.names)
            shown_share |= any(f.shared(n) for n in f.names)
            row = (f"| {f.key} | {names}"
                   f"{' *(continuous)*' if f.continuous else ''} "
                   f"| {f.end_conditions} | {f.M():.3f} |")
            for i in range(n_b):
                if i < f.n_bounds:
                    row += f" {f.K(i):.2f} | {f.T(i):.3f} |"
                else:
                    row += " — | — |"
            add(row)
        add("")
        if shown_share:
            add("A bent in **[brackets]** sits under an expansion joint: it "
                "carries the last span of one frame and the first span of the "
                "next, so it appears in BOTH with its full stiffness in each "
                "and half its tributary mass. A bent on a released bearing "
                "carries no deck longitudinally and is absent from the "
                "longitudinal frames entirely — its own seismic checks still "
                "run, on its own period.")
            add("")

        # --- worked derivation for every CONTINUOUS frame in this direction ---
        for f in frames:
            if not f.continuous:
                continue
            add(f"#### {f.key} — period derivation, {direction}")
            add("")
            add(f"{len(f.members)} bents act together here "
                f"({' + '.join(f.names)}), so the frame has ONE period, not "
                f"{len(f.members)}. Rigid deck, so the members deflect together "
                f"and their stiffnesses add.")
            add("")
            for i in range(f.n_bounds):
                lbl = f.label(i)
                add(f"**Fixity bound: {lbl}**")
                add("")
                ks = " + ".join(f"{b.stiffness(direction, i):.2f}"
                                for b in f.members)
                ms = " + ".join(f"{b.mass(direction):.3f}" for b in f.members)
                _eq(add, f"K_frame [{lbl}]", "Σ kᵢ  over the members that resist",
                    ks, f"{f.K(i):.2f} kip/in",
                    ref=f"members: {', '.join(f.names)} — each at its own end "
                        f"condition ({f.end_conditions})")
                _eq(add, "M_frame", "Σ mᵢ  in this direction", ms,
                    f"{f.M():.4f} kip·s²/in  "
                    f"(W = {f.M() * 386.088:.0f} kip)",
                    ref="entered tributary weight + participating column "
                        "self-weight, ÷ g")
                T = f.T(i)
                add(f"- **T_frame [{lbl}]** = 2π · √(M_frame / K_frame)"
                    f"  &nbsp;*[{cr.ref_geometry}]*  ")
                add(f"  = 2π · √({f.M():.4f} / {f.K(i):.2f}) "
                    f"= 2π · √({f.M() / f.K(i):.6f}) = **{T:.3f} s**")
                add("")
                # what each bent would have said on its own, for contrast
                own = ", ".join(f"{b.name} {b.T(direction, i):.3f} s"
                                for b in f.members)
                add(f"  *Individually the bents would give {own} — the frame "
                    f"period is not any one of them.*")
                add("")
            add("")

    # ------------------------------------------------------------------
    add("## Bents")
    add("")
    add("k is the effective lateral stiffness of the two-segment member at "
        "cracked stiffness EI = Mp/φy. The SECTION is axisymmetric, so k "
        "differs by direction only through the END CONDITION: an integral "
        "(moment-connected) bent is fixed-fixed longitudinally, everything else "
        "— and every bent transversely — is a fixed-free cantilever.")
    add("")
    add("**m** is the tributary mass the codes call for. It is the entered "
        "tributary weight for that direction, **plus** the column self-weight "
        "over the free length H_free at the participation factor in the "
        "settings (default 1/3), **excluding** the embedded shaft — that mass "
        "is restrained by the surrounding soil and does not participate in the "
        "sway mode. The exclusion is a modelling assumption, not a rounding.")
    add("")
    hdr = ("| Pier | Frame | Deck link | Hcol (ft) | Silo (ft) | H_free (ft) "
           "| m long. | m trans. | k long. | k trans. |")
    sep = "|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|"
    for i in range(n_b):
        lbl = balance.bents[0].label(i) if balance.bents else f"bound {i+1}"
        hdr += f" k [{lbl}] (kip/in) |"
        sep += "--:|"
    add(hdr)
    add(sep)
    for b in balance.bents:
        row = (f"| {b.name} | {b.frame} | {b.deck_link} | {b.Hcol/12:.1f} | "
               f"{b.silo/12:.1f} | {b.H_free/12:.1f} | {b.mass_long:.3f} | "
               f"{b.mass_trans:.3f} | {b.stiffness(LONGITUDINAL, 0):.1f} | "
               f"{b.stiffness(TRANSVERSE, 0):.1f} |")
        for i in range(n_b):
            # per BENT, like the two columns before it -- b.k is stored per
            # column, and printing it raw makes a 3-column bent read a third of
            # its own stiffness with nothing to say why
            row += (f" {b.n_columns * b.k[i]:.2f} |" if i < len(b.k) else " — |")
        add(row)
    add("")
    add("All stiffnesses are **per bent**: the columns of a bent act in "
        "parallel, so a bent is `n` times one column. The per-bound `k` "
        "columns are the fixed-free value; `k long.`/`k trans.` apply each "
        "direction's real end condition, so they are four times as stiff "
        "wherever the head is restrained.")
    add("")

    # ------------------------------------------------------------------
    add("## Checks")
    add("")
    if not balance.checks:
        add("*No pairs to check* — fewer than two frames take part (see the "
            "`frame` column).")
    else:
        for direction in DIRECTIONS:
            rows = [c for c in balance.checks if c.direction == direction]
            if not rows:
                continue
            add(f"### {direction.capitalize()}")
            add("")
            add("| Check | Pair | Frame | Bound | Ratio | Limit | Status | Values |")
            add("|:--|:--|:--|:--|--:|--:|:--|:--|")
            for c in rows:
                ratio = "—" if math.isnan(c.ratio) else f"{c.ratio:.3f}"
                add(f"| {c.name} | {c.pair[0]}–{c.pair[1]} | {c.scope or '—'} "
                    f"| {c.bound} | {ratio} | {c.limit:.2f} | "
                    f"{ {'OK': 'OK ✅', 'NG': 'NG ❌',
                        'THA': 'THA ⚠'}[c.status] } | {c.note} |")
            add("")

    if balance.log:
        add("## Balancing log")
        add("")
        add("The tuning lever is the **column silo** (isolation casing), which "
            "lengthens the free column and so softens a stiff pier. A silo is "
            "not free: it changes Lp, Δy, Δp, the displacement demand, "
            "Vo = Mo/H_free, the minimum lateral strength and P-Δ, so every "
            "pier whose silo changed was re-run through the full seismic check "
            "suite (and re-optimised if that run was an optimise run).")
        add("")
        for entry in balance.log:
            add(f"- {entry}")
        add("")

    if _tha:
        add("## Time-history analysis required")
        add("")
        add(f"**{len(_tha)} balanced frame-GEOMETRY pair(s) could not be met by "
            f"practical means.** The silo search pursued them and stopped: "
            f"closing them further would need silo depths or diameter changes "
            f"beyond what it could reach.")
        add("")
        add("| Pair | Direction | Bound | Ratio | Limit | Values |")
        add("|:--|:--|:--|--:|--:|:--|")
        for c in _tha:
            add(f"| {c.pair[0]}–{c.pair[1]} | {c.direction} | {c.bound} "
                f"| {c.ratio:.3f} | {c.limit:.2f} | {c.note} |")
        add("")
        add("These are **referred, not cleared**. SDC §C7.1.3 states that "
            "*\"the use of NTHA is not by itself a justification to waive the "
            "requirement of balanced frame geometry\"*, so accepting a "
            "shortfall against a time-history run is a **project-criteria "
            "decision for the owner**, not a code allowance. The balance rules "
            "exist to let a bridge *avoid* a more rigorous analysis; where they "
            "cannot be met, the shortfall and its magnitude are stated here so "
            "the decision is made explicitly.")
        add("")
        add("**The within-frame stiffness rules are NOT referred here and "
            "remain enforced.** That rule governs how a frame distributes "
            "demand between its own members, and a time-history run does not "
            "excuse it — any such shortfall is a hard failure above.")
        add("")

    add("## How this relates to the seismic checks")
    add("")
    add("- The per-bent suite now **designs** against the frame too: a bent "
        "that shares a frame is checked on the frame's period and its real end "
        "condition, in both directions, and the optimiser sizes it against "
        "that. A bent in a frame of one is unaffected, because there the "
        "frame's K and W are its own.")
    add("- Sizing one member changes K_frame and so the demand on every member "
        "of that frame, so the two iterate to a fixed point. The balancing log "
        "records how many passes that took.")
    add(f"- {END_CONDITION_NOTE}")
    add("")

    return "\n".join(lines)


def frame_seismic_report(frame_checks) -> str:
    """Markdown calc sheet for the frame-level displacement check.

    ``frame_checks`` is the list of
    :class:`~seismic_column.frame_seismic.FrameCheck` from a batch run.
    """
    from .frame_seismic import DECK, TOP_OF_SHAFT
    from .io_schema import DIRECTIONS

    lines: list[str] = []
    add = lines.append

    add("# Frame displacement check — real end conditions")
    add("")
    if not frame_checks:
        add("*No continuous frame in this bridge.* Every bent is its own "
            "single-bent frame, so the per-bent seismic suite already IS the "
            "frame check — same mass, same fixed-free cantilever, same demand.")
        return "\n".join(lines)

    worst = min(fc.worst for fc in frame_checks)
    ok = all(fc.passed for fc in frame_checks)
    add(f"**Result:** {'PASS ✅' if ok else 'FAIL ❌'} — worst Δc/Δd = "
        f"**{worst:.2f}**")
    add("")
    if not ok:
        # a member can fail on capacity or on P-Delta; the headline ratio does
        # not say which, so name the reasons.
        reasons: list[str] = []
        bad = [m for fc in frame_checks for m in fc.members if not m.passed]
        low = [m.name for m in bad if m.ratio < 1.0]
        pd_ = [m.name for m in bad if not m.pdelta_ok]
        if low:
            reasons.append(f"**Δc < Δd** at {', '.join(low)}")
        if pd_:
            reasons.append(f"**P-Δ** at {', '.join(pd_)}")
        add("Failing because of " + "; ".join(reasons) + ".")
        add("")
    add("**Why this section exists.** The per-bent suite treats every column as "
        "a stand-alone fixed-free cantilever: its own SDOF period, one plastic "
        "hinge at the top of shaft. For a bent that is *integral* with a "
        "continuous deck neither half of that holds longitudinally. Here the "
        "demand comes from the **frame** period and the capacity from the "
        "**frame's sway mechanism**, with each member at its real end "
        "condition.")
    add("")
    add("**Demand.** Rigid deck, so the members sway together and share one "
        "displacement:")
    add("")
    add("```")
    add("K_frame = Σ kᵢ   (each member at ITS end condition in this direction)")
    add("M_frame = Σ mᵢ")
    add("T_frame = 2π·√(M_frame / K_frame)   →   Δd from the spectrum")
    add("```")
    add("")
    add("**Capacity.** With x measured down from the deck to the point of "
        "fixity, statics gives a linear moment diagram `M(x) = V·x + M₀`:")
    add("")
    add("| End condition | M₀ | Consequence |")
    add("|:--|:--|:--|")
    add("| fixed-free | 0 | M peaks at the base; **one** hinge is a mechanism |")
    add("| fixed-fixed | −V·B/A | contraflexure at x = B/A; **two** hinges are "
        "needed |")
    add("")
    add("with `A = ∫dx/EI` and `B = ∫x·dx/EI` over the column and shaft "
        "segments. The first hinge forms where `M/Mp` peaks — and Mp is not "
        "constant, since the capacity-protected shaft carries far more than the "
        "column. So the deck, the top of shaft and the point of fixity are all "
        "candidates and the winner is read off the diagram, not assumed.")
    add("")
    add("A fixed-free member is determinate in sway, so the first hinge is "
        "already a mechanism. A fixed-fixed member is indeterminate to degree "
        "one, so it takes two — and the second has to be found by "
        "**redistribution**, not assumed at the far end. Once the first hinge "
        "at `x₁` is pinned at its capacity the diagram becomes")
    add("")
    add("```")
    add("M(x) = V·(x − x₁) + σ·Mp₁        σ = sign of M at the first hinge")
    add("```")
    add("")
    add("so each remaining section yields at `V = (±Mp − σ·Mp₁)/(x − x₁)`, and "
        "the mechanism forms at the smallest such V not below the load that "
        "made the first hinge. Where the first hinge lands at the deck this "
        "reduces to the familiar work equations — `V = 2·Mp_col/H_free` for a "
        "column mechanism, `V = (Mp_col + Mp_shaft)/L` when it runs into the "
        "shaft — but it stays correct when it does not.")
    add("")
    add("The plastic displacement follows the hinge **spacing**: "
        "`Δp = θp·(x₂ − x₁ − Lp)` for two hinges, against "
        "`Δp = θp·(H_free − Lp/2)` for the single-hinge cantilever.")
    add("")

    for direction in DIRECTIONS:
        sel = [fc for fc in frame_checks if fc.direction == direction]
        if not sel:
            continue
        add(f"## {direction.capitalize()}")
        add("")
        for fc in sel:
            add(f"### {fc.frame_key} — {' + '.join(fc.member_names)} "
                f"— {fc.bound_label}")
            add("")
            add(f"End conditions: **{fc.end_conditions}**")
            add("")
            _eq(add, "K_frame", "Σ kᵢ",
                " + ".join(f"{m.k:.2f}" for m in fc.members),
                f"{fc.K:.2f} kip/in",
                ref=f"each member at its own end condition "
                    f"({fc.end_conditions})")
            _eq(add, "M_frame", "Σ mᵢ",
                " + ".join(f"{m.m:.3f}" for m in fc.members),
                f"{fc.M:.4f} kip·s²/in  (W = {fc.W:.0f} kip)")
            add(f"- **T_frame** = 2π·√(M/K) = 2π·√({fc.M:.4f} / {fc.K:.2f}) "
                f"= **{fc.T:.3f} s**  ")
            add(f"- **Sa** = {fc.Sa:.4f} g  →  **Δd = {fc.delta_d:.2f} in** "
                "(shared by every member of the frame)")
            add("")
            add("| Member | End cond. | Sway mechanism | V_mech (kip) | Δy (in) "
                "| Δp (in) | Δc (in) | Δd (in) | Δc/Δd | μd | P-Δ | Status |")
            add("|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|:--|:--|")
            for m in fc.members:
                add(f"| {m.name} | "
                    f"{'fixed-fixed' if m.end_fixity == 'fixed' else 'fixed-free'} "
                    f"| {m.mechanism} | {m.V_mech:.0f} "
                    f"| {m.delta_y:.2f} | {m.delta_p:.2f} | {m.delta_c:.2f} "
                    f"| {m.delta_d:.2f} | **{m.ratio:.2f}** | {m.mu_d:.2f} "
                    f"| {'OK' if m.pdelta_ok else 'NG'} "
                    f"| {'OK ✅' if m.passed else 'NG ❌'} |")
            add("")
            # what the capacity-protected shaft must then be sized for
            if any(m.end_fixity == "fixed" for m in fc.members):
                add("**Deck capacity-design demand at this mechanism**")
                add("")
                add("The deck and its joint are capacity-protected, exactly as "
                    "the shaft is: the plastic hinge belongs in the COLUMN at "
                    "both ends, so the deck must be designed to resist the "
                    "column overstrength moment rather than to yield before it. "
                    "That is what makes `Vo = 2·Mo/H` the requirement here and "
                    "not a conservative choice — a joint that yielded first "
                    "would move the hinge into the superstructure.")
                add("")
                add("| Member | Mo the deck must resist (kip-ft) | Vo delivered "
                    "(kip) | per column |")
                add("|:--|--:|--:|:--|")
                for m in fc.members:
                    add(f"| {m.name} | {m.Mo_interface/12:.0f} "
                        f"| {m.Vo_interface:.0f} | at each column head |")
                add("")
                add("**Not checked here.** Whether the deck, the joint and the "
                    "diaphragm actually have that capacity — that is a "
                    "superstructure check, and this tool does not do it. The "
                    "numbers above are the demand to design them for.")
                add("")

                add("**Shaft capacity-design demand at this mechanism**")
                add("")
                add("The shaft is held elastic by design, so it does not "
                    "compete to hinge — it has to be sized for the column's "
                    "overstrength demand. Both hinges of a column mechanism "
                    "sit at Mp_col, so the interface **moment** is the same Mo "
                    "the fixed-free suite already used. The **shear** is not.")
                add("")
                add("| Member | Mo at interface (kip-ft) | Vo at interface "
                    "(kip) | Vo, fixed-free basis | Amplification "
                    "| M below ground (kip-ft) | Mp shaft | D/C | |")
                add("|:--|--:|--:|--:|--:|--:|--:|--:|:--|")
                for m in fc.members:
                    ig = ("—" if m.shaft_solution is None
                          else f"{m.shaft_moment/12:.0f}")
                    dc = ("—" if m.shaft_solution is None
                          else f"**{m.shaft_dc:.2f}**")
                    verdict = ("" if m.shaft_solution is None else
                               ("YIELDS ❌" if m.shaft_dc > 1.0 else "OK ✅"))
                    add(f"| {m.name} | {m.Mo_interface/12:.0f} "
                        f"| {m.Vo_interface:.0f} | {m.Vo_cantilever:.0f} "
                        f"| **{m.shear_amplification:.2f}×** | {ig} "
                        f"| {m.shaft_Mp/12:.0f} | {dc} | {verdict} |")
                add("")
                add("The below-ground columns are a **p-y solve at this "
                    "mechanism's head condition** — `V = Vo` with `M = Mo` "
                    "applied at the head, so the interface lands on Mo rather "
                    "than the 2·Mo that applying the shear alone would give. "
                    "The head-moment sign is verified against the interface "
                    "moment on every solve, because the wrong sign lands 3·Mo "
                    "there and would look plausible in this table.")
                add("")
                add("**A D/C above 1.00 is reported, not failed.** The "
                    "fixed-fixed mechanism is closed-form plastic analysis, not "
                    "a pushover; it should be confirmed against a real model "
                    "before the shaft is resized on the strength of it.")
                add("")

            # the moment diagram, so the hinge location is auditable
            add("**Moment diagram — shear needed to yield each section**")
            add("")
            add("| Member | Section | x from deck (in) | lever \\|M\\|/V (in) | "
                "Mp (kip-ft) | V to yield (kip) |")
            add("|:--|:--|--:|--:|--:|--:|")
            for m in fc.members:
                order = {h.name: i for i, h in enumerate(m.hinges)}
                for s in m.sections:
                    i = order.get(s.name)
                    mark = "" if i is None else (
                        " ← **hinge 1**" if i == 0 else f" ← **hinge {i+1}**")
                    vy = ("— *(no moment)*" if not math.isfinite(s.V_yield)
                          else f"{s.V_yield:.0f}")
                    add(f"| {m.name} | {s.name}{mark} | {s.x:.0f} | "
                        f"{s.arm:.1f} | {s.Mp/12:.0f} | {vy} |")
            add("")
            if any(len(m.hinges) > 1 for m in fc.members):
                add("*`V to yield` is the ELASTIC value. The second hinge forms "
                    "on the redistributed diagram, so its mechanism load "
                    "differs from the elastic figure in this table.*")
                add("")
            notes = [(m.name, w) for m in fc.members for w in m.warnings]
            if notes:
                add("**Warnings**")
                add("")
                for name, w in notes:
                    add(f"- **{name}** — {w}")
                add("")

    add("## Assumptions and limits")
    add("")
    add("- **This is closed-form plastic analysis, not an incremental "
        "pushover.** It gives the yielding order and the load at which a "
        "mechanism forms, redistributing once past the first hinge — which is "
        "enough for a member indeterminate to degree one. It does not trace "
        "the load-displacement curve, spread of plasticity, or post-mechanism "
        "response, and it takes Mp as fixed rather than tracking the "
        "moment-axial interaction as the frame sways. Where the answer is "
        "close, run the real pushover.")
    add("- The **rigid-deck** assumption makes the members share Δd. That is "
        "reasonable longitudinally; transversely it overstates K_frame on a "
        "long or flexible deck.")
    add("- **Δy is the bilinear idealisation**: the elastic flexibility "
        "projected to V_mech. Past the first hinge the real member is softer, "
        "so the true displacement at V_mech is larger — the standard "
        "idealisation, and the same one the per-bent suite uses.")
    add("- The depth to fixity `Df` is solved for a **free-head** member and "
        "re-used with a fixed head. That is the usual simplification, not an "
        "exact equivalence, and it matters most for the fixed-fixed case.")
    add("- The **point of fixity is an equivalent depth**, not a real section. "
        "A hinge reported there means the shaft is being asked to yield "
        "somewhere below ground; it does not locate the yielding depth.")
    add(f"- A first hinge at the **{DECK}** rather than the "
        f"**{TOP_OF_SHAFT}** means the deck joint, not just the shaft, needs "
        "capacity protection. It does not on its own break the Type II premise: "
        "the second hinge is still in the column, at the top of shaft.")
    add("- **The shaft is assumed capacity-protected**, so it is excluded as a "
        "hinge candidate rather than checked against its as-entered Mp. That is "
        "a design obligation, not an assumption that comes free — the shaft "
        "demand table above is the demand it must be sized for, and it is NOT "
        "verified here. If the shaft cannot be made strong enough within a "
        "buildable diameter, that is a real failure, but a shaft capacity-design "
        "failure rather than a displacement one.")
    add("- Capacity here is direction-independent per member except through the "
        "end condition; the section is axisymmetric.")
    add("")

    return "\n".join(lines)


def _detailed_calcs(rr: RowResult, checks=None) -> list[str]:
    """Full equations, each shown symbolically then with substituted numbers and
    a specific code reference, so every value can be verified by hand."""
    a = rr.assessment
    checks = a.checks if checks is None else checks
    prov = a.provisions
    d = rr.design
    s = rr.shaft
    sec = d.section()
    conf = sec.confined
    mc = a.mc_col
    g = a.governing_bound
    L = a.H_free or (a.bounds[0].Le - a.bounds[0].fixity_depth)   # free length
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
    _add_fixity_source(add, a)
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
        add(r"- **Rd** = (1 − 1/μd)·(T\*/T) + 1/μd ≥ 1  (short-period "
            r"magnification, **Δd ← Rd·Δd**)  &nbsp;*[SGS 4.3.3]*")
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
        add("*Rd not applied — Caltrans SDC has no short-period magnification.*")
        add("")
    else:
        _add_rd_derivation(add, a)

    # governing displacement & ductility demand checks
    add("**Governing displacement checks (worst fixity bound):**")
    _chk(add, "Displacement capacity", "Δc ≥ Δd",
         f"{g.delta_c:.2f} ≥ {g.demand.disp_demand:.2f} in  "
         f"(D/C = {g.demand.disp_demand/g.delta_c:.2f})", R["dc"],
         status=g.delta_c >= g.demand.disp_demand)
    mud_lim = next((c.capacity for c in checks
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
    rho_l_min = next((c.demand for c in checks
                      if c.name == "Longitudinal steel ratio (min)"), 0.01)
    rho_l_max = next((c.capacity for c in checks
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
    add("**Type II shaft reinforcement:**")
    add(f"- Provided: ρl = **{sh.rho_l*100:.2f}%** ({s.long_label()}), "
        f"ρs = **{sh.rho_s*100:.2f}%** ({s.spiral_label()}).")
    add(f"- The Type II shaft is a **capacity-protected** member, not a ductile "
        f"SCM/column, so the ductile min/max ratios above do **not** apply to it. "
        f"Its steel is governed by **capacity** — develop γ·Mo in flexure and Vo "
        f"in shear (§6) — plus the 0.04 \"compression member\" max "
        f"({'SGS 8.8.1' if not is_ct else 'SDC 5.3.9.1'}). "
        f"ρl = {sh.rho_l*100:.2f}% ≤ 4.0% → "
        + ("**OK** ✅" if sh.rho_l <= rho_l_max else "**NG** ❌") + ".")
    add("- No seismic *minimum* longitudinal ratio is prescribed for the "
        "capacity-protected shaft; general drilled-shaft minimums come from the "
        "base bridge-design specifications (AASHTO LRFD / AASHTO-CA BDS §5 & §10), "
        "which lie outside these seismic provisions and are not evaluated here.")
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
    ck_axr = _find(checks, "Axial load ratio")
    if ck_axr is not None:
        rho_dl = a.P_used / (min(d.fc, 5.0) * sec.Ag)
        _chk(add, "Axial load ratio", "ρdl = Pdl/(f'c·Ag) ≤ 0.15  (f'c ≤ 5 ksi)",
             f"{a.P_used:.0f}/({min(d.fc,5.0):.1f}·{sec.Ag:.0f}) = "
             f"{rho_dl:.3f} ≤ {ck_axr.capacity:g}", prov.ref_max_axial,
             status=ck_axr.passed)
    ck_axm = _find(checks, "Maximum axial load")
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
    _detailing_calcs(add, a, d, checks)

    return lines


def _detailing_calcs(add, a, d, checks=None) -> None:
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
    for c in (a.checks if checks is None else checks):
        form = forms.get(c.name)
        if form is None:
            continue
        any_shown = True
        _chk(add, c.name, form[0], c.note, form[1], status=c.passed)
    if not any_shown:
        add("- No code-specific detailing checks are enabled for this code.")
