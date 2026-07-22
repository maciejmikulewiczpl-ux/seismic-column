"""Streamlit GUI for the circular column seismic optimiser (Caltrans SDC 2.1).

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from seismic_column.batch import RowResult, results_to_dataframe, run_batch
from seismic_column.demand import SpectrumSpec
from seismic_column.io_schema import (
    COLUMNS,
    COLUMN_META,
    GlobalConfig,
    SOIL_COLUMN_META,
    SOIL_COLUMNS,
    build_soil_profile,
    default_dataframe,
    default_soil_layers,
    load_project,
    load_soil_preset,
    pile_profile_table,
    project_from_json,
    project_to_json,
    py_curves_table,
    save_project,
    soil_preset_names,
    validate,
)
from seismic_column.optimizer import PARAMETERS
from seismic_column.provisions import PROVISIONS
from seismic_column.report import column_report

st.set_page_config(page_title="Seismic Column Optimiser (SDC 2.1)", layout="wide")

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
    "code": "SDC 2.1",
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
    "min_shaft_oversize_in": 24.0,
    "optimize_objective": "balanced",
    "balanced_rho_pct": 2.0,
    "concrete_unit_weight": 0.150,
    "self_weight_mass_factor": 1.0 / 3.0,
    "self_weight_in_axial": True,
    "project_path": "",       # current project file for in-place Save/Open
    "editor_version": 0,      # bump to force the batch editor to re-init
    # soil-structure interaction (point of fixity)
    "fixity_source": "multiplier",
    "water_table_ft": 10.0,
    "shaft_embed_ft": 60.0,
    "soil_stiff_factor": 2.0,
    "soil_soft_factor": 0.5,
    "soil_df": pd.DataFrame(default_soil_layers()),
    "soil_version": 0,
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


def _ss(key):
    """Session-state value, falling back to the default (never KeyErrors).

    A conditionally-rendered widget (e.g. Sds when the spectrum is tabular) can
    have its key garbage-collected by Streamlit, so read defensively.
    """
    return st.session_state.get(key, _DEFAULTS.get(key))


def _build_config() -> GlobalConfig:
    s = _ss
    design = _spectrum_spec(s("design_kind"), s("design_Sds"), s("design_Sd1"),
                            s("design_ars"))
    lle = None
    if s("lle_enabled"):
        lle = _spectrum_spec(s("lle_kind"), s("lle_Sds"), s("lle_Sd1"), s("lle_ars"))
    priority = tuple(p.strip() for p in s("priority_txt").split(",") if p.strip())
    return GlobalConfig(
        design_spectrum=design, lle_spectrum=lle, lle_mu_limit=s("lle_mu_limit"),
        code=s("code"),
        fye=s("fye"), fue=s("fue"), fyh=s("fyh"), optimize=s("optimize"),
        priority=priority, variable=tuple(s("variable")),
        shaft_moment_basis=s("shaft_basis"), mu_d_limit=s("mu_d_limit"),
        rho_l_min=s("rho_l_min"), rho_l_max=s("rho_l_max"),
        min_bar_spacing=s("min_bar_spacing"), allow_bundling=s("allow_bundling"),
        min_shaft_oversize_in=s("min_shaft_oversize_in"),
        optimize_objective=s("optimize_objective"),
        balanced_rho_l=s("balanced_rho_pct") / 100.0,
        concrete_unit_weight=s("concrete_unit_weight"),
        self_weight_mass_factor=s("self_weight_mass_factor"),
        self_weight_in_axial=s("self_weight_in_axial"),
        fixity_source=s("fixity_source"),
        water_table_ft=s("water_table_ft"),
        shaft_embed_ft=s("shaft_embed_ft"),
        soil_stiff_factor=s("soil_stiff_factor"),
        soil_soft_factor=s("soil_soft_factor"),
        soil_layers=tuple(s("soil_df").dropna(how="all").to_dict("records")),
    )


def _load_project_into_state(df: pd.DataFrame, cfg: GlobalConfig) -> None:
    s = st.session_state
    s["batch_df"] = df
    s["editor_version"] = s.get("editor_version", 0) + 1   # re-init the editor
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
    s["min_shaft_oversize_in"] = getattr(cfg, "min_shaft_oversize_in", 24.0)
    s["optimize_objective"] = getattr(cfg, "optimize_objective", "balanced")
    s["balanced_rho_pct"] = getattr(cfg, "balanced_rho_l", 0.02) * 100.0
    s["concrete_unit_weight"] = cfg.concrete_unit_weight
    s["self_weight_mass_factor"] = cfg.self_weight_mass_factor
    s["self_weight_in_axial"] = cfg.self_weight_in_axial
    s["fixity_source"] = getattr(cfg, "fixity_source", "multiplier")
    s["water_table_ft"] = getattr(cfg, "water_table_ft", 10.0)
    s["shaft_embed_ft"] = getattr(cfg, "shaft_embed_ft", 60.0)
    s["soil_stiff_factor"] = getattr(cfg, "soil_stiff_factor", 2.0)
    s["soil_soft_factor"] = getattr(cfg, "soil_soft_factor", 0.5)
    if getattr(cfg, "soil_layers", ()):
        s["soil_df"] = pd.DataFrame(list(cfg.soil_layers))
        s["soil_version"] = s.get("soil_version", 0) + 1


st.title("Circular RC Column on Type II Shaft — Seismic Optimiser")
st.caption("Caltrans SDC 2.1 · Equivalent Static Analysis · Mander confinement · fibre M-φ")


# ---------------------------------------------------------------------------
# Project save / open
# ---------------------------------------------------------------------------
_NATIVE_UNAVAILABLE = (
    "Couldn't open a native file dialog on this machine. Use the "
    "**Browser upload / download** section below instead."
)


def _native_file_dialog(mode: str, initial_path: str = "") -> str | None:
    """Open a native OS file picker and return the chosen path.

    ``mode`` is ``"open"`` or ``"save"``.  The dialog runs in a **subprocess**
    (its own Tk main loop) so it can't collide with Streamlit's script thread.
    Only works when the app runs locally (server == the user's machine).

    Returns the selected path, ``""`` if the user cancelled, or ``None`` if the
    dialog could not be shown (Tk missing, headless/remote session, …).
    """
    initial_dir = os.path.dirname(initial_path) if initial_path else os.getcwd()
    initial_file = (os.path.basename(initial_path) if initial_path
                    else "seismic_project.json")
    script = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
        "ft = [('Project JSON', '*.json'), ('All files', '*.*')]\n"
        f"mode, idir, ifile = {mode!r}, {initial_dir!r}, {initial_file!r}\n"
        "if mode == 'open':\n"
        "    p = filedialog.askopenfilename(title='Open project', filetypes=ft,\n"
        "        initialdir=idir)\n"
        "else:\n"
        "    p = filedialog.asksaveasfilename(title='Save project as', filetypes=ft,\n"
        "        defaultextension='.json', initialdir=idir, initialfile=ifile)\n"
        "r.destroy()\n"
        "import sys; sys.stdout.write(p or '')\n"
    )
    try:
        out = subprocess.run([sys.executable, "-c", script],
                             capture_output=True, text=True, timeout=300)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _do_save(path: str) -> None:
    try:
        save_project(path, st.session_state["batch_df"], _build_config())
        st.session_state["project_path"] = path
        st.session_state["_proj_msg"] = ("success", f"Saved to {path}")
    except Exception as exc:
        st.session_state["_proj_msg"] = ("error", f"Save failed: {exc}")


def _do_open(path: str) -> None:
    try:
        df_p, cfg_p = load_project(path)
        _load_project_into_state(df_p, cfg_p)
        st.session_state["project_path"] = path
        st.session_state["_proj_msg"] = ("success", f"Opened {path}")
    except Exception as exc:
        st.session_state["_proj_msg"] = ("error", f"Open failed: {exc}")


st.header("Project")
_current = st.session_state.get("project_path", "")
st.caption(f"📄 Current file: **{_current}**" if _current
           else "No project file yet — use **Save As…** or **Open…**. The whole "
                "project (every column, including optimised designs, plus all "
                "settings) is stored in one `.json` file.")

pc1, pc2, pc3 = st.columns(3)
save_clicked = pc1.button("💾 Save", type="primary", width="stretch",
    help="Save all columns + settings to the current project file, in place. "
         "If none is set yet, you'll be asked where to save.")
saveas_clicked = pc2.button("💾 Save As…", width="stretch",
    help="Pick a file name and location, then save.")
open_clicked = pc3.button("📂 Open…", width="stretch",
    help="Browse for a project file to open.")

if save_clicked:
    path = st.session_state.get("project_path") or _native_file_dialog("save")
    if path:
        _do_save(path)
    elif path is None:
        st.session_state["_proj_msg"] = ("error", _NATIVE_UNAVAILABLE)
if saveas_clicked:
    path = _native_file_dialog("save", st.session_state.get("project_path", ""))
    if path:
        _do_save(path)
    elif path is None:
        st.session_state["_proj_msg"] = ("error", _NATIVE_UNAVAILABLE)
if open_clicked:
    path = _native_file_dialog("open", st.session_state.get("project_path", ""))
    if path:
        _do_open(path)
        st.rerun()
    elif path is None:
        st.session_state["_proj_msg"] = ("error", _NATIVE_UNAVAILABLE)

msg = st.session_state.pop("_proj_msg", None)
if msg is not None:
    (st.success if msg[0] == "success" else st.error)(msg[1])

with st.expander("Browser upload / download (if not running locally)"):
    try:
        proj_json = project_to_json(st.session_state["batch_df"], _build_config())
        st.download_button(
            "Download project (.json)", proj_json.encode(),
            os.path.basename(_current) or "seismic_project.json",
            "application/json")
    except Exception as exc:
        st.warning(f"Fix inputs to enable download: {exc}")
    proj_up = st.file_uploader("Upload project (.json)", type=["json"],
                               key="proj_up")
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
    st.radio(
        "Objective", ["min_diameter", "balanced", "min_steel"],
        key="optimize_objective",
        format_func=lambda v: {"min_diameter": "Smallest column",
                               "balanced": "Balanced (≤ target steel)",
                               "min_steel": "Least steel"}[v],
        help="Search starts at the minimum column diameter + minimum steel and "
             "grows. **Smallest column**: first size that works (heavier steel). "
             "**Balanced**: smallest column whose longitudinal steel is ≤ the "
             "target below. **Least steel**: the feasible column with the lightest "
             "cage (largest diameter). Least steel scans all diameters, so it is "
             "the slowest — most so in soil p-y mode.")
    if st.session_state["optimize_objective"] == "balanced":
        st.number_input("Target longitudinal steel for balanced (%)", 1.0, 4.0,
                        key="balanced_rho_pct", step=0.25,
                        help="Balanced returns the smallest column whose "
                             "longitudinal reinforcement ratio is at or below this.")
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
    st.number_input("Min shaft oversize (in)", 0.0, 96.0,
                    key="min_shaft_oversize_in", step=6.0,
                    help="When optimising grows the column, the shaft is enlarged "
                         "to the next standard size so it stays at least this many "
                         "inches larger than the column (Type II oversize). Your "
                         "entered shaft is the floor. Default 24 in (2 ft); the "
                         "code minimum (24 in for Caltrans) is always enforced.")

    st.subheader("Checks")
    st.selectbox("Shaft moment demand basis", ["interface", "fixity"], key="shaft_basis")
    st.number_input("Displacement ductility limit μd", 1.0, 8.0, key="mu_d_limit", step=0.5)
    st.number_input("ρl min", 0.005, 0.03, key="rho_l_min", step=0.001, format="%.3f")
    st.number_input("ρl max", 0.02, 0.08, key="rho_l_max", step=0.001, format="%.3f")

    st.subheader("Point of fixity")
    st.radio(
        "How is the depth to fixity determined?", ["multiplier", "soil"],
        key="fixity_source", horizontal=True,
        format_func=lambda v: {"multiplier": "Assumed 3×/6× multiplier",
                               "soil": "Calculated (soil p-y)"}[v],
        help="**Assumed 3×/6× multiplier**: Df = 3× (upper-bound stiffness) and "
             "6× (lower-bound) × shaft diameter, no soil model. "
             "**Calculated (soil p-y)**: nonlinear p-y (LPile-equivalent) analysis "
             "of the column + shaft on the strata below, giving a mechanics-based "
             "depth to fixity and the in-ground shaft moment/shear.")
    st.caption("Assumed = fast, classic bracket. Calculated = enter strata below; "
               "slower but mechanics-based.")
    if st.session_state["fixity_source"] == "soil":
        _presets = ["—"] + soil_preset_names()
        pc1, pc2 = st.columns([3, 1])
        _sel = pc1.selectbox(
            "Load a preset strata profile", _presets, key="soil_preset_sel",
            help="Prefills the strata table + groundwater depth from a saved "
                 "LPile-style profile. Submerged layers already converted to "
                 "total unit weight; 'Ignore' layers modelled as zero-resistance "
                 "(elastic k=0). Review every value against your geotech's report.")
        if pc2.button("Load", key="soil_preset_load") and _sel != "—":
            _wt, _layers = load_soil_preset(_sel)
            st.session_state["soil_df"] = pd.DataFrame(_layers)
            st.session_state["water_table_ft"] = _wt
            st.session_state["soil_version"] = \
                st.session_state.get("soil_version", 0) + 1
            st.rerun()
        st.number_input("Embedded shaft length (ft)", 10.0, 300.0,
                        key="shaft_embed_ft", step=5.0)
        st.number_input("Groundwater depth (ft, below top of shaft)", 0.0, 300.0,
                        key="water_table_ft", step=1.0)
        c1, c2 = st.columns(2)
        c1.number_input("Stiff-soil bound ×", 1.0, 5.0, key="soil_stiff_factor",
                        step=0.5, help="Upper-bound p-y modulus multiplier.")
        c2.number_input("Soft-soil bound ×", 0.1, 1.0, key="soil_soft_factor",
                        step=0.1, help="Lower-bound p-y modulus multiplier.")
        st.caption("Strata (top → bottom), LPile-style inputs — your geotech's "
                   "LPile soil table maps here 1:1.")
        soil_cfg = {c: st.column_config.TextColumn(SOIL_COLUMN_META[c][0],
                                                   help=SOIL_COLUMN_META[c][1])
                    if c in ("layer", "py_model")
                    else st.column_config.NumberColumn(SOIL_COLUMN_META[c][0],
                                                       help=SOIL_COLUMN_META[c][1])
                    for c in SOIL_COLUMNS}
        edited_soil = st.data_editor(
            st.session_state["soil_df"], num_rows="dynamic", width="stretch",
            key=f"soil_editor_{st.session_state['soil_version']}",
            column_config=soil_cfg)
        st.session_state["soil_df"] = edited_soil

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
    key=f"editor_{st.session_state['editor_version']}", column_config=col_config,
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
if cfg.optimize:
    st.caption("Optimised column & shaft designs are written back into the table "
               "above after each run, so a **Save** captures the current progress.")
if st.session_state["fixity_source"] == "soil":
    st.caption("⏳ Soil (p-y) analysis solves a nonlinear pile per column "
               "(more so when optimising), so a large batch can take a minute or "
               "more — the progress bar below shows it is working.")
if st.button("Run batch", type="primary"):
    n_total = len(edited)
    bar = st.progress(0.0, text=f"Starting {n_total} columns…")
    tally = st.empty()
    counts = {"PASS": 0, "FAIL": 0, "ERROR": 0}
    t0 = time.time()

    done_rows = {"n": 0}

    def _progress(done, total, name, status):
        done_rows["n"] = done
        counts[status] = counts.get(status, 0) + 1
        rate = (time.time() - t0) / max(done, 1)
        eta = rate * (total - done)
        bar.progress(done / total,
                     text=f"Analysing {done}/{total} — last: {name} [{status}]"
                          + (f" · ~{eta:.0f}s left" if done < total else ""))
        tally.caption(f"✅ {counts['PASS']} pass · ❌ {counts['FAIL']} fail · "
                      f"⚠️ {counts['ERROR']} error")

    def _on_candidate(name, it):
        # live movement WITHIN a column (a soil p-y optimise can take a while per
        # column, and per-row progress alone can't show that it's working).
        row_i = done_rows["n"] + 1
        bar.progress(done_rows["n"] / n_total,
                     text=f"Analysing {row_i}/{n_total} — {name}: "
                          f"trying design {it}… ({time.time() - t0:.0f}s)")

    try:
        summary, results = run_batch(edited, cfg, progress=_progress,
                                     on_candidate=_on_candidate)
    except Exception as exc:
        st.error(f"Run failed: {exc}")
    else:
        bar.progress(1.0, text=f"Done — {n_total} columns in "
                               f"{time.time() - t0:.0f}s")
        st.session_state["summary"] = summary
        st.session_state["results"] = results
        if cfg.optimize and results:
            # Fold the optimised designs back into the table so the batch is the
            # current design of record and Save persists progress.  A write-back
            # glitch must NOT read as "Run failed" — the analysis already
            # succeeded and the results below are valid.  (st.rerun() raises a
            # control-flow exception, so it stays OUTSIDE the try.)
            wrote_back = False
            try:
                st.session_state["batch_df"] = results_to_dataframe(results, edited)
                st.session_state["editor_version"] += 1
                wrote_back = True
            except Exception as exc:
                st.warning(f"Results are ready below, but writing the optimised "
                           f"designs back into the table failed: {exc}")
            if wrote_back:
                st.rerun()


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

        # p-y pile response: deflection / shear / moment diagrams (soil fixity)
        ig = rr.assessment.inground_solution
        soil_bounds = [b for b in rr.assessment.bounds if b.soil_solution]
        if ig is not None or soil_bounds:
            st.markdown("**Pile response diagrams (p-y)** — distance below the "
                        "column top; ground line (top of shaft) dashed. Solid = "
                        "**shaft-design demand at column overstrength Mo**; "
                        "dashed = yield-level stiffness bounds.")
            cols = st.columns(3)
            for ax_col, attr, xlabel, scale in (
                    (cols[0], "y", "deflection (in)", 1.0),
                    (cols[1], "shear", "shear (kip)", 1.0),
                    (cols[2], "moment", "moment (kip-ft)", 1.0 / 12.0)):
                fig3, ax3 = plt.subplots()
                for b in soil_bounds:                       # yield-level bounds
                    s = b.soil_solution
                    ax3.plot(getattr(s, attr) * scale, s.x, "--", lw=0.8,
                             alpha=0.6, label=f"{b.soil_label} (yield)")
                if ig is not None:                          # overstrength design
                    ax3.plot(getattr(ig, attr) * scale, ig.x, "k-", lw=1.6,
                             label="overstrength (design)")
                ref = ig if ig is not None else soil_bounds[0].soil_solution
                ax3.axhline(ref.x[ref.ground_index], ls="--", color="0.5", lw=1)
                ax3.axvline(0, color="0.7", lw=0.6)
                ax3.invert_yaxis()
                ax3.set_xlabel(xlabel)
                ax3.set_ylabel("dist. from column top (in)")
                ax3.legend(fontsize=7)
                ax_col.pyplot(fig3)

            # p-y curves at representative depths + exports for the global model
            prof = build_soil_profile(cfg)
            if prof is not None:
                Dsh = rr.shaft.D
                depths = prof.representative_depths(Dsh, cfg.shaft_embed_ft * 12.0)
                e1, e2 = st.columns([3, 2])
                with e1:
                    figpy, axpy = plt.subplots()
                    for z in depths:
                        ys, ps = prof.py_curve(z, Dsh)
                        axpy.plot(ys, ps, label=f"{z/12:.1f} ft")
                    axpy.set_xlabel("y (in)")
                    axpy.set_ylabel("p (kip/in)")
                    axpy.set_title("p-y curves by depth")
                    axpy.legend(fontsize=7, title="depth")
                    st.pyplot(figpy)
                with e2:
                    st.caption("Export for the global structural model:")
                    _sol = ig if ig is not None else \
                        rr.assessment.governing_bound.soil_solution
                    prof_df = pile_profile_table(_sol)
                    st.download_button(
                        "Pile deflection/shear/moment (CSV)",
                        prof_df.to_csv(index=False).encode(),
                        f"pile_profile_{rr.name}.csv", "text/csv")
                    py_df = py_curves_table(prof, Dsh, depths)
                    st.download_button(
                        "p-y curves (CSV)", py_df.to_csv(index=False).encode(),
                        f"py_curves_{rr.name}.csv", "text/csv")

        report_md = column_report(rr)
        st.markdown(report_md)
        st.download_button("Download report (Markdown)", report_md.encode(),
                           f"report_{rr.name}.md", "text/markdown")
