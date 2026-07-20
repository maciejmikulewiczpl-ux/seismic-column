"""Streamlit GUI for the circular column seismic optimiser (Caltrans SDC 2.0).

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from seismic_column.batch import RowResult, run_batch
from seismic_column.demand import SpectrumSpec
from seismic_column.io_schema import (
    COLUMNS,
    COLUMN_META,
    GlobalConfig,
    default_dataframe,
    project_from_json,
    project_to_json,
    validate,
)
from seismic_column.optimizer import PARAMETERS
from seismic_column.provisions import PROVISIONS
from seismic_column.report import column_report

st.set_page_config(page_title="Seismic Column Optimiser (SDC 2.0)", layout="wide")

INT_COLS = ("n_bars", "long_bar_no", "spiral_bar_no",
            "shaft_n_bars", "shaft_long_bar_no", "shaft_spiral_bar_no")


# ---------------------------------------------------------------------------
# Session-state defaults (single source of truth so projects can be re-loaded)
# ---------------------------------------------------------------------------
def _default_ars(kind: str) -> pd.DataFrame:
    if kind == "design":
        return pd.DataFrame({"T_s": [0.0, 0.2, 0.5, 1.0, 2.0, 4.0],
                             "Sa_g": [0.40, 1.00, 1.00, 0.60, 0.30, 0.15]})
    return pd.DataFrame({"T_s": [0.0, 0.2, 0.5, 1.0, 2.0, 4.0],
                         "Sa_g": [0.16, 0.40, 0.40, 0.24, 0.12, 0.06]})


_DEFAULTS = {
    "batch_df": default_dataframe(3),
    "code": "SDC 2.0",
    "design_kind": "parametric",
    "design_Sds": 1.0, "design_Sd1": 0.6,
    "design_ars": _default_ars("design"),
    "lle_enabled": True,
    "lle_kind": "parametric",
    "lle_Sds": 0.4, "lle_Sd1": 0.24,
    "lle_ars": _default_ars("lle"),
    "lle_mu_limit": 1.0,
    "fye": 68.0, "fue": 95.0, "fyh": 68.0,
    "optimize": True,
    "variable": list(PARAMETERS),
    "priority_txt": ", ".join(PARAMETERS),
    "shaft_basis": "interface",
    "mu_d_limit": 5.0,
    "rho_l_min": 0.01, "rho_l_max": 0.04,
    "min_bar_spacing": 6.0,
    "allow_bundling": False,
    "concrete_unit_weight": 0.150,
    "self_weight_mass_factor": 1.0 / 3.0,
    "self_weight_in_axial": True,
}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def _spectrum_spec(kind: str, sds: float, sd1: float, ars: pd.DataFrame) -> SpectrumSpec:
    if kind == "tabular":
        clean = ars.dropna()
        return SpectrumSpec(kind="tabular",
                            periods=tuple(float(x) for x in clean["T_s"]),
                            accels=tuple(float(x) for x in clean["Sa_g"]))
    return SpectrumSpec(kind="parametric", Sds=sds, Sd1=sd1)


def _build_config() -> GlobalConfig:
    s = st.session_state
    design = _spectrum_spec(s["design_kind"], s["design_Sds"], s["design_Sd1"],
                            s["design_ars"])
    lle = None
    if s["lle_enabled"]:
        lle = _spectrum_spec(s["lle_kind"], s["lle_Sds"], s["lle_Sd1"], s["lle_ars"])
    priority = tuple(p.strip() for p in s["priority_txt"].split(",") if p.strip())
    return GlobalConfig(
        design_spectrum=design, lle_spectrum=lle, lle_mu_limit=s["lle_mu_limit"],
        code=s["code"],
        fye=s["fye"], fue=s["fue"], fyh=s["fyh"], optimize=s["optimize"],
        priority=priority, variable=tuple(s["variable"]),
        shaft_moment_basis=s["shaft_basis"], mu_d_limit=s["mu_d_limit"],
        rho_l_min=s["rho_l_min"], rho_l_max=s["rho_l_max"],
        min_bar_spacing=s["min_bar_spacing"], allow_bundling=s["allow_bundling"],
        concrete_unit_weight=s["concrete_unit_weight"],
        self_weight_mass_factor=s["self_weight_mass_factor"],
        self_weight_in_axial=s["self_weight_in_axial"],
    )


def _load_project_into_state(df: pd.DataFrame, cfg: GlobalConfig) -> None:
    s = st.session_state
    s["batch_df"] = df
    s["code"] = cfg.code
    ds = cfg.design_spectrum
    s["design_kind"] = ds.kind
    s["design_Sds"], s["design_Sd1"] = ds.Sds, ds.Sd1
    if ds.kind == "tabular" and ds.periods:
        s["design_ars"] = pd.DataFrame({"T_s": list(ds.periods), "Sa_g": list(ds.accels)})
    if cfg.lle_spectrum is not None:
        s["lle_enabled"] = True
        ls = cfg.lle_spectrum
        s["lle_kind"] = ls.kind
        s["lle_Sds"], s["lle_Sd1"] = ls.Sds, ls.Sd1
        if ls.kind == "tabular" and ls.periods:
            s["lle_ars"] = pd.DataFrame({"T_s": list(ls.periods), "Sa_g": list(ls.accels)})
    else:
        s["lle_enabled"] = False
    s["lle_mu_limit"] = cfg.lle_mu_limit
    s["fye"], s["fue"], s["fyh"] = cfg.fye, cfg.fue, cfg.fyh
    s["optimize"] = cfg.optimize
    s["variable"] = list(cfg.variable)
    s["priority_txt"] = ", ".join(cfg.priority)
    s["shaft_basis"] = cfg.shaft_moment_basis
    s["mu_d_limit"] = cfg.mu_d_limit
    s["rho_l_min"], s["rho_l_max"] = cfg.rho_l_min, cfg.rho_l_max
    s["min_bar_spacing"] = cfg.min_bar_spacing
    s["allow_bundling"] = cfg.allow_bundling
    s["concrete_unit_weight"] = cfg.concrete_unit_weight
    s["self_weight_mass_factor"] = cfg.self_weight_mass_factor
    s["self_weight_in_axial"] = cfg.self_weight_in_axial


st.title("Circular RC Column on Type II Shaft — Seismic Optimiser")
st.caption("Caltrans SDC 2.0 · Equivalent Static Analysis · Mander confinement · fibre M-φ")


# ---------------------------------------------------------------------------
# Project save / open
# ---------------------------------------------------------------------------
st.header("Project")
pcol1, pcol2 = st.columns([2, 3])
with pcol1:
    proj_up = st.file_uploader("Open project (.json)", type=["json"], key="proj_up")
    if proj_up is not None and not st.session_state.get("_proj_loaded"):
        try:
            df_p, cfg_p = project_from_json(proj_up.getvalue().decode("utf-8"))
            _load_project_into_state(df_p, cfg_p)
            st.session_state["_proj_loaded"] = True
            st.success("Project loaded.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not load project: {exc}")
    if proj_up is None:
        st.session_state["_proj_loaded"] = False
with pcol2:
    try:
        proj_json = project_to_json(st.session_state["batch_df"], _build_config())
        st.download_button("Save project (.json)", proj_json.encode(),
                           "seismic_project.json", "application/json")
    except Exception as exc:
        st.warning(f"Fix inputs to enable save: {exc}")


# ---------------------------------------------------------------------------
# Sidebar: global settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Global settings")

    st.subheader("Design code")
    st.selectbox("Seismic design code", list(PROVISIONS.keys()), key="code",
                 help="Switches code-specific provisions (reinforcement limits, "
                      "overstrength factor, confinement coefficients, ductility / "
                      "P-Δ factors) and the clause references cited in the report. "
                      "The core methodology is common to both codes.")

    st.subheader("Design spectrum (upper-level EQ)")
    st.radio("Input type", ["parametric", "tabular"], key="design_kind",
             horizontal=True)
    if st.session_state["design_kind"] == "parametric":
        st.number_input("Sds (short-period, g)", 0.1, 3.0, key="design_Sds", step=0.05)
        st.number_input("Sd1 (1-second, g)", 0.05, 2.0, key="design_Sd1", step=0.05)
    else:
        st.caption("ARS curve — period (s) vs Sa (g)")
        st.session_state["design_ars"] = st.data_editor(
            st.session_state["design_ars"], num_rows="dynamic",
            key="design_ars_editor", width="stretch")

    st.subheader("Low-level EQ (elastic check)")
    st.checkbox("Enable low-level earthquake check", key="lle_enabled")
    if st.session_state["lle_enabled"]:
        st.caption("Structure must remain essentially elastic (μ ≤ limit).")
        st.number_input("μ limit (essentially elastic)", 0.5, 1.5,
                        key="lle_mu_limit", step=0.05)
        st.radio("LLE input type", ["parametric", "tabular"], key="lle_kind",
                 horizontal=True)
        if st.session_state["lle_kind"] == "parametric":
            st.number_input("LLE Sds (g)", 0.05, 2.0, key="lle_Sds", step=0.05)
            st.number_input("LLE Sd1 (g)", 0.02, 1.5, key="lle_Sd1", step=0.02)
        else:
            st.caption("LLE ARS curve — period (s) vs Sa (g)")
            st.session_state["lle_ars"] = st.data_editor(
                st.session_state["lle_ars"], num_rows="dynamic",
                key="lle_ars_editor", width="stretch")

    st.subheader("Materials")
    st.number_input("fye (ksi)", 40.0, 100.0, key="fye", step=1.0)
    st.number_input("fue (ksi)", 60.0, 140.0, key="fue", step=1.0)
    st.number_input("fyh transverse (ksi)", 40.0, 100.0, key="fyh", step=1.0)

    st.subheader("Column self-weight")
    st.number_input("Concrete unit weight (kcf)", 0.10, 0.20,
                    key="concrete_unit_weight", step=0.005, format="%.3f")
    st.number_input("Self-weight participation in seismic mass", 0.0, 1.0,
                    key="self_weight_mass_factor", step=0.05,
                    help="Fraction of the column self-weight (of the length above "
                         "the hinge) lumped at the top as participating mass. "
                         "~1/3 is a common cantilever approximation; 0 disables. "
                         "Applies to both design and low-level earthquake demand.")
    st.checkbox("Add column self-weight to axial P", key="self_weight_in_axial",
                help="Adds the column self-weight above the hinge to the axial "
                     "load used for M-φ, P-Δ and shear.")

    st.subheader("Optimiser")
    st.checkbox("Optimise (else check as-entered)", key="optimize")
    st.multiselect("Variable parameters", list(PARAMETERS), key="variable")
    st.text_input("Priority order (comma-separated)", key="priority_txt")
    st.number_input("Min longitudinal bar spacing (in)", 3.0, 12.0,
                    key="min_bar_spacing", step=0.5,
                    help="Min centre-to-centre spacing of longitudinal bars along the "
                         "cage. Cage perimeter / this = max number of bar positions.")
    st.checkbox("Allow bundled longitudinal bars (2-bar)", key="allow_bundling",
                help="Permit 2-bar bundles when the perimeter is full. "
                     "Longitudinal bars go up to #14; spirals up to #8, "
                     "with bundled #4 @ 4\" as the max confinement (Caltrans).")

    st.subheader("Checks")
    st.selectbox("Shaft moment demand basis", ["interface", "fixity"], key="shaft_basis")
    st.number_input("Displacement ductility limit μd", 1.0, 8.0, key="mu_d_limit", step=0.5)
    st.number_input("ρl min", 0.005, 0.03, key="rho_l_min", step=0.001, format="%.3f")
    st.number_input("ρl max", 0.02, 0.08, key="rho_l_max", step=0.001, format="%.3f")

cfg = _build_config()


# ---------------------------------------------------------------------------
# Batch table input
# ---------------------------------------------------------------------------
st.header("1 · Column batch")
st.caption("**W (seismic weight)** drives mass → period → demand.  "
           "**P (axial load)** is the sustained compression used for M-φ, "
           "P-Δ and shear.  Hover any header for details.")

upload = st.file_uploader("Import batch table (CSV or Excel)",
                          type=["csv", "xlsx", "xls"], key="batch_up")
if upload is not None:
    try:
        if upload.name.lower().endswith((".xlsx", ".xls")):
            df_in = pd.read_excel(upload)
        else:
            df_in = pd.read_csv(upload)
        st.session_state["batch_df"] = validate(df_in)
        st.success(f"Imported {len(st.session_state['batch_df'])} rows.")
    except Exception as exc:
        st.error(f"Import failed: {exc}")

col_config = {}
for c in COLUMNS:
    label, help_txt = COLUMN_META[c]
    if c == "name":
        col_config[c] = st.column_config.TextColumn(label, help=help_txt)
    elif c in INT_COLS:
        col_config[c] = st.column_config.NumberColumn(label, help=help_txt, step=1)
    else:
        col_config[c] = st.column_config.NumberColumn(label, help=help_txt)

edited = st.data_editor(
    st.session_state["batch_df"], num_rows="dynamic", width="stretch",
    key="editor", column_config=col_config,
)
st.session_state["batch_df"] = edited

col_a, col_b = st.columns(2)
with col_a:
    st.download_button("Export CSV", edited.to_csv(index=False).encode(),
                       "columns.csv", "text/csv")
with col_b:
    xbuf = io.BytesIO()
    edited.to_excel(xbuf, index=False)
    st.download_button("Export Excel", xbuf.getvalue(), "columns.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
st.header("2 · Run")
if st.button("Run batch", type="primary"):
    with st.spinner("Analysing…"):
        try:
            summary, results = run_batch(edited, cfg)
            st.session_state["summary"] = summary
            st.session_state["results"] = results
        except Exception as exc:
            st.error(f"Run failed: {exc}")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if "summary" in st.session_state:
    st.header("3 · Results")
    summary: pd.DataFrame = st.session_state["summary"]
    results: list[RowResult] = st.session_state["results"]

    st.dataframe(summary, width="stretch")
    st.download_button("Export results CSV", summary.to_csv(index=False).encode(),
                       "results.csv", "text/csv")

    if results:
        st.subheader("Drill-down")
        names = [r.name for r in results]
        sel = st.selectbox("Select column", names)
        rr = next(r for r in results if r.name == sel)

        c1, c2 = st.columns(2)
        with c1:
            mc = rr.assessment.mc_col
            fig, ax = plt.subplots()
            ax.plot(mc.phi, mc.M / 12.0, label="column M-φ")
            ax.plot([0, mc.phi_y, mc.phi_u],
                    [0, mc.Mp / 12.0, mc.Mp / 12.0], "r--o", lw=1,
                    label="Caltrans bilinear")
            ax.set_xlabel("curvature φ (1/in)")
            ax.set_ylabel("moment M (kip-ft)")
            ax.set_title("Column moment-curvature")
            ax.legend()
            st.pyplot(fig)
        with c2:
            design_spec = cfg.design_spectrum.build()
            periods = np.linspace(0.01, 5.0, 400)
            fig2, ax2 = plt.subplots()
            ax2.plot(periods, [design_spec.Sa(t) for t in periods], label="design")
            if cfg.lle_spectrum is not None:
                lle_spec = cfg.lle_spectrum.build()
                ax2.plot(periods, [lle_spec.Sa(t) for t in periods], "--", label="low-level")
            for b in rr.assessment.bounds:
                ax2.plot(b.demand.period, b.demand.Sa, "o",
                         label=f"mult {b.multiplier:g}: T={b.demand.period:.2f}s")
            ax2.set_xlabel("period T (s)")
            ax2.set_ylabel("Sa (g)")
            ax2.set_title("Spectra & effective periods")
            ax2.legend(fontsize=8)
            st.pyplot(fig2)

        report_md = column_report(rr)
        st.markdown(report_md)
        st.download_button("Download report (Markdown)", report_md.encode(),
                           f"report_{rr.name}.md", "text/markdown")
