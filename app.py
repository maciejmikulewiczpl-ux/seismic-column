"""Streamlit GUI for the circular column seismic optimiser (Caltrans SDC 2.1).

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import io
import math
import os
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st

from seismic_column.balance import (GEOMETRY_CHECK, STIFFNESS_ANY_CHECK,
                                    STIFFNESS_CHECK)
from seismic_column.batch import (RowResult, results_to_dataframe,
                                  run_batch_balanced)
from seismic_column.demand import SpectrumSpec
from seismic_column.io_schema import (
    COLUMNS,
    COLUMN_META,
    GlobalConfig,
    SOIL_COLUMN_META,
    SOIL_COLUMNS,
    DECK_LINKS,
    DIRECTIONS,
    TEXT_COLUMNS,
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
from seismic_column.report import (balance_report, column_report,
                                   frame_seismic_report)

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
    "optimize_objective": "min_diameter",
    "target_rho_pct": 2.0,
    "concrete_unit_weight": 0.150,
    "self_weight_mass_factor": 1.0 / 3.0,
    "self_weight_in_axial": True,
    "project_path": "",       # current project file for in-place Save/Open
    "editor_version": 0,      # bump to force the batch editor to re-init
    # balanced stiffness / balanced frame geometry between adjacent piers
    "balance_check": True,
    "balance_mass_normalized": True,
    "balance_k_ratio_min": 0.75,
    "balance_T_ratio_min": 0.70,
    "balance_auto_silo": True,
    "balance_strategy": "min_silo",
    "balance_direction": DIRECTIONS[0],
    "max_silo_ft": 20.0,
    "silo_step_ft": 1.0,
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
        target_rho_l=s("target_rho_pct") / 100.0,
        concrete_unit_weight=s("concrete_unit_weight"),
        self_weight_mass_factor=s("self_weight_mass_factor"),
        self_weight_in_axial=s("self_weight_in_axial"),
        balance_check=s("balance_check"),
        balance_mass_normalized=s("balance_mass_normalized"),
        balance_k_ratio_min=s("balance_k_ratio_min"),
        balance_T_ratio_min=s("balance_T_ratio_min"),
        balance_auto_silo=s("balance_auto_silo"),
        balance_strategy=s("balance_strategy"),
        max_silo_ft=s("max_silo_ft"),
        silo_step_ft=s("silo_step_ft"),
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
    s["optimize_objective"] = getattr(cfg, "optimize_objective", "min_diameter")
    s["target_rho_pct"] = getattr(cfg, "target_rho_l", 0.02) * 100.0
    s["concrete_unit_weight"] = cfg.concrete_unit_weight
    s["self_weight_mass_factor"] = cfg.self_weight_mass_factor
    s["self_weight_in_axial"] = cfg.self_weight_in_axial
    s["balance_check"] = getattr(cfg, "balance_check", True)
    s["balance_mass_normalized"] = getattr(cfg, "balance_mass_normalized", True)
    s["balance_k_ratio_min"] = getattr(cfg, "balance_k_ratio_min", 0.75)
    s["balance_T_ratio_min"] = getattr(cfg, "balance_T_ratio_min", 0.70)
    s["balance_auto_silo"] = getattr(cfg, "balance_auto_silo", True)
    s["balance_strategy"] = getattr(cfg, "balance_strategy", "min_silo")
    s["max_silo_ft"] = getattr(cfg, "max_silo_ft", 20.0)
    s["silo_step_ft"] = getattr(cfg, "silo_step_ft", 1.0)
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
        "Objective", ["min_diameter", "target_steel", "min_steel",
                      "fixed_diameter"],
        key="optimize_objective",
        format_func=lambda v: {"min_diameter": "Smallest column",
                               "target_steel": "Target steel %",
                               "min_steel": "Least steel",
                               "fixed_diameter": "Fixed diameter (min steel)"}[v],
        help="Search starts at the minimum column diameter and grows to the "
             "smallest size that works. **Smallest column**: steel escalated as "
             "needed (heaviest cage). **Target steel %**: hold longitudinal steel "
             "at the ratio below and find the smallest column that works there. "
             "**Least steel**: smallest column that works at the ~1% minimum. "
             "**Fixed diameter**: keep the column diameter you entered and find "
             "the smallest reinforcement ratio that works.")
    if st.session_state["optimize_objective"] == "target_steel":
        st.number_input("Target longitudinal steel (%)", 1.0, 4.0,
                        key="target_rho_pct", step=0.25,
                        help="The optimiser holds the longitudinal ratio at about "
                             "this value and returns the smallest column diameter "
                             "that satisfies every check.")
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

    st.subheader("Balanced stiffness & frame geometry")
    st.checkbox(
        "Check balanced stiffness & frame geometry", key="balance_check",
        help="Caltrans SDC 2.1 §7.1.2 / §7.1.3 (AASHTO SGS §4.1.2 / §4.1.3), "
             "between ADJACENT piers only. Piers are grouped by the 'frame' "
             "column of the batch table and compared in table row order; a "
             "blank 'frame' cell leaves that pier out. Untick to switch the "
             "whole feature off — no checks, no silos, results unchanged.")
    _bal = st.session_state.get("balance_check", True)
    if not _bal:
        st.caption("Balance checks off — adjacent piers are not compared and no "
                   "column silo is added.")
    st.checkbox(
        "Auto-size column silos to satisfy them", key="balance_auto_silo",
        disabled=not _bal,
        help="Lengthens the free column of the stiffer pier of a failing pair "
             "(isolation casing, SDC C7.1.2 / SGS §4.1.4) and re-runs its full "
             "seismic check suite — a silo raises the displacement demand and "
             "lowers Vo and Vp, so it is not free. An entered 'silo_ft' is a "
             "minimum and is never reduced. Unticked = report the shortfall "
             "without changing any design.")
    st.checkbox(
        "Normalise stiffness by tributary mass (k/m)",
        key="balance_mass_normalized", disabled=not _bal,
        help="Ticked: compare k/m — the Caltrans Table 7.1.2-1 form, and the "
             "AASHTO variable-width form (Eq. 4.1.2-4). Untick to compare k "
             "alone (AASHTO constant-width, Eq. 4.1.2-3). Spans in series with "
             "differing tributary masses want the normalised form.")
    bc1, bc2 = st.columns(2)
    bc1.number_input("Min adjacent k ratio", 0.5, 1.0, key="balance_k_ratio_min",
                     step=0.05, disabled=not _bal,
                     help="min(ki,kj)/max(ki,kj). Code minimum 0.75; a stricter "
                          "entry is honoured, a looser one is ignored.")
    bc2.number_input("Min adjacent period ratio", 0.5, 1.0,
                     key="balance_T_ratio_min", step=0.05, disabled=not _bal,
                     help="min(Ti,Tj)/max(Ti,Tj). Code minimum 0.70.")
    st.selectbox(
        "Silo sizing strategy", ["min_silo", "greedy"], key="balance_strategy",
        disabled=not (_bal and st.session_state.get("balance_auto_silo", True)),
        format_func=lambda v: {"min_silo": "Minimum total silo (exact)",
                               "greedy": "Pairwise repair (fast)"}[v],
        help="**Minimum total silo**: a dynamic program over each frame finds "
             "the cheapest combination of buildable silo depths that satisfies "
             "every adjacent pair — the true optimum on the grid, not an "
             "approximation of it. **Pairwise repair**: fix one failing pair at "
             "a time, deepening monotonically. Both are re-verified against a "
             "real analysis each pass and, on the projects measured so far, "
             "reach the same answer — the exact search buys a guarantee rather "
             "than a shallower silo.")
    bs1, bs2 = st.columns(2)
    bs1.number_input("Max silo depth (ft)", 0.0, 60.0, key="max_silo_ft",
                     step=1.0, disabled=not (_bal and
                                             st.session_state.get("balance_auto_silo", True)),
                     help="How deep the tool may size a silo by itself. If the "
                          "cap binds, the run reports the shortfall instead of "
                          "going deeper. A deeper silo typed into the table is "
                          "your call and is honoured as-is.")
    bs2.number_input("Silo increment (ft)", 0.25, 5.0, key="silo_step_ft",
                     step=0.25, disabled=not (_bal and
                                              st.session_state.get("balance_auto_silo", True)),
                     help="Silo depths are rounded UP to a whole number of "
                          "these — a buildable increment.")

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
if st.session_state.get("optimize", True):
    st.caption("💡 When optimising you may leave the **reinforcement and f′c "
               "cells blank** — the optimiser always starts from the minimum and "
               "sizes them. Height, loads, diameters and cover are still required. "
               "(A **check** run needs the reinforcement filled in.)")

upload = st.file_uploader("Import batch table (CSV or Excel)",
                          type=["csv", "xlsx", "xls"], key="batch_up")
if upload is not None:
    try:
        if upload.name.lower().endswith((".xlsx", ".xls")):
            df_in = pd.read_excel(upload)
        else:
            df_in = pd.read_csv(upload)
        # tolerate blank rebar/f'c on import (optimise runs may leave them blank)
        st.session_state["batch_df"] = validate(df_in, optimize=True)
        st.success(f"Imported {len(st.session_state['batch_df'])} rows.")
    except Exception as exc:
        st.error(f"Import failed: {exc}")

col_config = {}
for c in COLUMNS:
    label, help_txt = COLUMN_META[c]
    if c == "deck_link":
        col_config[c] = st.column_config.SelectboxColumn(
            label, help=help_txt, options=list(DECK_LINKS), required=False)
    elif c in TEXT_COLUMNS:
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

    stage = {"txt": ""}      # set once the balance stage takes over

    def _on_candidate(name, it):
        # live movement WITHIN a column (a soil p-y optimise can take a while per
        # column, and per-row progress alone can't show that it's working).
        row_i = min(done_rows["n"] + 1, n_total)
        where = stage["txt"] or f"Analysing {row_i}/{n_total}"
        bar.progress(min(done_rows["n"] / n_total, 1.0),
                     text=f"{where} — {name}: trying design {it}… "
                          f"({time.time() - t0:.0f}s)")

    def _balance_progress(msg):
        # the balance stage re-runs an unpredictable subset of rows, so it gets
        # a message rather than a done-count.
        stage["txt"] = msg
        bar.progress(1.0, text=f"{msg} ({time.time() - t0:.0f}s)")

    try:
        outcome = run_batch_balanced(edited, cfg, progress=_progress,
                                     on_candidate=_on_candidate,
                                     balance_progress=_balance_progress)
        summary, results = outcome.summary, outcome.results
    except Exception as exc:
        st.error(f"Run failed: {exc}")
    else:
        bar.progress(1.0, text=f"Done — {n_total} columns in "
                               f"{time.time() - t0:.0f}s")
        st.session_state["summary"] = summary
        st.session_state["results"] = results
        st.session_state["balance"] = outcome.balance
        st.session_state["frame_checks"] = outcome.frame_checks
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
# Balance plots
# ---------------------------------------------------------------------------
# Pass/fail is a STATUS encoding.  Failures are the signal and compliant pairs
# are context, so only the failures get a saturated colour (status "critical");
# passing marks stay recessive grey.  That is also the accessible choice: the
# obvious red/green pair separates by only deutan ΔE 4.1 — a deuteranope sees
# one colour — whereas critical-vs-grey separates by 9.1, clearing the ≥8 bar,
# because it differs in chroma rather than hue.  Colour still never carries the
# verdict alone: failing marks are dashed/hatched, labelled "✗ <ratio>", and the
# same numbers sit in the checks table above.
_BAD = "#d03b3b"                        # status: critical
_INK, _MUTED, _GRID = "#52514e", "#898781", "#e1e0d9"
_OK = _MUTED                            # compliant = recessive, not green


def _bound_status(balance, check_name: str, direction: str,
                  frames=None) -> dict:
    """{(pier_i, pier_j, bound_index): BalanceCheck} for one check type.

    The geometry rule is keyed on FRAMES, so when ``frames`` is given each
    frame key is resolved to a representative pier — the frame's first member —
    so the check can be drawn as a link on the pier axis.
    """
    labels = balance.bents[0].bound_labels if balance.bents else ()
    at = {lbl: i for i, lbl in enumerate(labels)}
    rep = {f.key: f.members[0].name for f in (frames or [])}
    out = {}
    for c in balance.checks:
        if c.name != check_name or c.direction != direction:
            continue
        a, b = rep.get(c.pair[0], c.pair[0]), rep.get(c.pair[1], c.pair[1])
        out[(a, b, at.get(c.bound, 0))] = c
    return out


def _profile_fig(bents, value_of, ylabel, status):
    """Metric along the bridge, with each adjacent LINK drawn pass/fail.

    Piers are markers; the link between adjacent piers is the thing being
    checked, so it carries the verdict.  Piers in different frames are simply
    not linked, which makes frame boundaries visible.
    """
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    idx = {b.name: i for i, b in enumerate(bents)}
    n_bounds = min((len(b.k) for b in bents), default=0)

    for (ni, nj, bnd), c in sorted(status.items(), key=lambda kv: kv[0][2]):
        if ni not in idx or nj not in idx:
            continue
        xi, xj = idx[ni], idx[nj]
        yi, yj = value_of(bents[xi], bnd), value_of(bents[xj], bnd)
        ax.plot([xi, xj], [yi, yj],
                color=_BAD if not c.passed else _OK,
                ls="--" if not c.passed else "-",           # not colour alone
                lw=2.0, zorder=2, solid_capstyle="round")
        if not c.passed:                                     # label only failures
            # stagger by bound: the bound lines converge at the flexible end,
            # so a fixed offset collides there.
            dy = 9 if bnd % 2 == 0 else -14
            mid_y = math.sqrt(yi * yj) if min(yi, yj) > 0 else (yi + yj) / 2
            ax.annotate(f"✗ {c.ratio:.2f}", ((xi + xj) / 2, mid_y),
                        textcoords="offset points", xytext=(0, dy),
                        ha="center", fontsize=8, color=_BAD, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.85), zorder=4)

    marks = ["o", "s", "^"]
    siloed = [b.silo > 0 for b in bents]
    for bnd in range(n_bounds):
        ys = [value_of(b, bnd) for b in bents]
        mk = marks[bnd % len(marks)]
        # hollow marker = as-built height; filled = a silo was added here
        ax.plot(range(len(bents)), ys, ls="none", marker=mk,
                ms=8, mfc="white", mec=_INK, mew=1.4, zorder=3,
                label=bents[0].label(bnd))
        xs_s = [i for i, s in enumerate(siloed) if s]
        if xs_s:
            ax.plot(xs_s, [ys[i] for i in xs_s], ls="none", marker=mk,
                    ms=8, mfc=_INK, mec=_INK, mew=1.4, zorder=4)

    ax.set_xticks(range(len(bents)))
    # the silo depth rides on the pier axis: always visible, never over the data
    ax.set_xticklabels([f"{b.name}\n+{b.silo / 12:g} ft" if b.silo > 0 else b.name
                        for b in bents])
    total = sum(b.silo for b in bents) / 12.0
    ax.set_xlabel("pier (in table order)"
                  + (f"  ·  filled = silo added, {total:g} ft total"
                     if total else "  ·  no silos"))
    # The check is a RATIO, so a log axis makes it read directly: the same ratio
    # is the same vertical distance anywhere on the plot, and a stiff short pier
    # no longer squashes its flexible neighbours into the baseline.
    vals = [value_of(b, i) for b in bents for i in range(n_bounds)]
    if vals and min(vals) > 0 and max(vals) / min(vals) > 8.0:
        ax.set_yscale("log")
        ylabel += "  (log)"
        # plain engineering numbers, not 6x10^1
        fmt = mticker.FuncFormatter(
            lambda v, _p: f"{v:,.0f}" if v >= 1 else f"{v:,.2g}")
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_formatter(fmt)
        ax.tick_params(axis="y", which="minor", labelsize=7)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=_GRID, lw=0.8, which="both")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_MUTED)
    if n_bounds >= 2:
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def _frame_profile_fig(bents, frames, status, ylabel="T (s)"):
    """Frame period along the bridge, with each adjacent-FRAME link drawn.

    The geometry rule compares FRAMES, not bents, so this chart has to be drawn
    on frames or the picture lies.  Plotting per-bent periods and linking a
    frame by its first member — which is what the shared ``_profile_fig`` did —
    put endpoints on the chart that were not the numbers being compared, and
    left every other member of a continuous frame with no line touching it.

    The pier axis is kept so the chart still reads along the bridge, but a
    continuous frame is drawn as a horizontal SEGMENT spanning its members at
    the single frame period: `[A8 A9 A10]` reads as a plateau, which is exactly
    what "these three share one period" means.
    """
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    idx = {b.name: i for i, b in enumerate(bents)}
    n_bounds = min((f.n_bounds for f in frames), default=0)
    marks = ["o", "s", "^"]

    def span(f):
        xs = [idx[n] for n in f.names if n in idx]
        return (min(xs), max(xs)) if xs else (0, 0)

    # --- links between adjacent frames: from the right edge of one to the left
    #     edge of the next, so the line never runs back through a frame ---
    by_key = {f.key: f for f in frames}
    for (ki, kj, bnd), c in sorted(status.items(), key=lambda kv: kv[0][2]):
        fi, fj = by_key.get(ki), by_key.get(kj)
        if fi is None or fj is None or bnd >= n_bounds:
            continue
        xi, xj = span(fi)[1], span(fj)[0]
        yi, yj = fi.T(bnd), fj.T(bnd)
        ax.plot([xi, xj], [yi, yj], color=_BAD if not c.passed else _OK,
                ls="--" if not c.passed else "-", lw=2.0, zorder=2,
                solid_capstyle="round")
        if not c.passed:
            dy = 9 if bnd % 2 == 0 else -14
            mid = math.sqrt(yi * yj) if min(yi, yj) > 0 else (yi + yj) / 2
            ax.annotate(f"✗ {c.ratio:.2f}", ((xi + xj) / 2, mid),
                        textcoords="offset points", xytext=(0, dy),
                        ha="center", fontsize=8, color=_BAD, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.85), zorder=4)

    # --- the frames themselves ---
    first_cont = next((f.key for f in frames if len(f.names) > 1), None)
    for bnd in range(n_bounds):
        mk = marks[bnd % len(marks)]
        for j, f in enumerate(frames):
            x0, x1 = span(f)
            y = f.T(bnd)
            if x1 > x0:                      # continuous frame: one period, so
                ax.plot([x0, x1], [y, y], color=_INK, lw=4.0, alpha=0.30,
                        solid_capstyle="round", zorder=2,
                        label=("continuous frame — one period"
                               if bnd == 0 and f.key == first_cont else None))
            ax.plot([x0, x1] if x1 > x0 else [x0], [y] * (2 if x1 > x0 else 1),
                    ls="none", marker=mk, ms=8, mfc="white", mec=_INK, mew=1.4,
                    zorder=3,
                    label=frames[0].label(bnd) if j == 0 else None)
            # A silo belongs to a PIER, not to the frame, so mark it at the
            # pier's own x — filling the frame's end markers would read as
            # "both ends are siloed", which is not what the table says.
            for b in f.members:
                if b.silo > 0 and b.name in idx:
                    ax.plot([idx[b.name]], [y], ls="none", marker=mk, ms=8,
                            mfc=_INK, mec=_INK, mew=1.4, zorder=4,
                            label=("pier with a silo"
                                   if bnd == 0 and not ax.get_legend_handles_labels()[1]
                                   .count("pier with a silo") else None))

    ax.set_xticks(range(len(bents)))
    ax.set_xticklabels([f"{b.name}\n+{b.silo / 12:g} ft" if b.silo > 0 else b.name
                        for b in bents])
    total = sum(b.silo for b in bents) / 12.0
    ax.set_xlabel("pier (in table order)"
                  + (f"  ·  filled = silo added, {total:g} ft total"
                     if total else "  ·  no silos"))
    vals = [f.T(i) for f in frames for i in range(n_bounds)]
    if vals and min(vals) > 0 and max(vals) / min(vals) > 8.0:
        ax.set_yscale("log")
        ylabel += "  (log)"
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=_GRID, lw=0.8, which="both")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_MUTED)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def _ratio_fig(checks, limit, title):
    """Every adjacent-pair ratio against its limit — the 'which ones fail' chart."""
    fig, ax = plt.subplots(figsize=(5.4, max(2.2, 0.42 * len(checks) + 1.2)))
    labels = [f"{c.pair[0]}–{c.pair[1]}  [{c.bound}]" for c in checks]
    ys = range(len(checks))
    for y, c in zip(ys, checks):
        r = 0.0 if np.isnan(c.ratio) else c.ratio
        ax.barh(y, r, height=0.5, zorder=2,
                color=_OK if c.passed else _BAD,
                alpha=0.55 if c.passed else 0.85,
                hatch="" if c.passed else "///",         # not colour alone
                edgecolor="white", linewidth=0.0)
        ax.annotate(("n/a" if np.isnan(c.ratio)
                     else f"{'✗ ' if not c.passed else ''}{c.ratio:.3f}"),
                    (r, y), textcoords="offset points", xytext=(6, 0),
                    va="center", fontsize=8, color=_BAD if not c.passed else _INK,
                    fontweight="bold" if not c.passed else "normal")
    ax.axvline(limit, color=_INK, lw=1.2, zorder=3)
    # Anchor to the x-axis transform (data x, axes-fraction y): the data y-axis
    # is inverted, so a data-y here lands off-canvas.  Sits just INSIDE the top
    # of the plot so it never collides with the title.
    ax.annotate(f"limit {limit:.2f}", xy=(limit, 1.0),
                xycoords=ax.get_xaxis_transform(),
                textcoords="offset points", xytext=(5, -11), fontsize=8,
                color=_INK, zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.85))
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()                       # first pair on the bridge at the top
    ax.set_xlim(0, max(1.05, max((c.ratio for c in checks
                                 if not np.isnan(c.ratio)), default=1.0) * 1.15))
    ax.set_xlabel("min / max  (1.00 = perfectly matched)")
    ax.set_title(title, fontsize=10, color=_INK)
    ax.grid(axis="x", color=_GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_GRID)
    ax.tick_params(colors=_MUTED)
    fig.tight_layout()
    return fig


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

    balance = st.session_state.get("balance")
    if balance is not None:
        cr = balance.criteria
        st.subheader("Balanced stiffness & frame geometry")
        if not balance.checks:
            st.info("No adjacent pairs to check — fewer than two piers share a "
                    "frame. Set the **frame** column in the batch table.")
        elif balance.passed:
            st.success(f"All adjacent pairs comply "
                       f"(κ ratio ≥ {cr.k_ratio_min:.2f}, "
                       f"T ratio ≥ {cr.T_ratio_min:.2f}).")
        else:
            st.error(f"{len(balance.failed)} adjacent-pair check(s) fail."
                     + ("" if balance.converged else
                        " The silo search did not converge — see the log."))

        # A silo lengthens the free column, which changes Lp, the displacement
        # demand, Vo, Vp and P-Delta — so every siloed pier was re-run through
        # the FULL seismic suite.  That re-verification is the whole point of
        # the coupling, so state it plainly rather than leaving it implicit in
        # the status column.
        siloed = [r for r in results if r.silo > 0]
        if siloed:
            bad = [r for r in siloed if not r.feasible]
            detail = ", ".join(
                f"{r.name} {r.assessment.Hcol_entered/12:.0f}→"
                f"{r.assessment.H_free/12:.0f} ft"
                f"{'' if r.feasible else ' ⚠️'}" for r in siloed)
            msg = (f"**Seismic re-verification:** {len(siloed)} pier(s) carry a "
                   f"silo and were re-analysed over the lengthened free column "
                   f"— {detail}. ")
            if bad:
                st.warning(msg + f"**{len(bad)} now FAIL** their seismic checks: "
                                 + "; ".join(f"{r.name} ("
                                             + ", ".join(c.name for c in
                                                         r.assessment.checks
                                                         if not c.passed) + ")"
                                             for r in bad))
            else:
                st.info(msg + "All still pass the full seismic check suite at "
                              "the lengthened length.")
        elif balance.checks:
            st.info("**Seismic re-verification:** no silo was needed, so no "
                    "pier was re-analysed.")

        direction = st.radio(
            "Direction", list(DIRECTIONS), horizontal=True,
            key="balance_direction",
            format_func=str.capitalize,
            help="Tributary mass differs by direction, and so do the frames: a "
                 "bearing released longitudinally but shear-keyed joins the "
                 "continuous frame transversely only. Everything below is for "
                 "the selected direction; both are checked.")
        frames = balance.frames.get(direction, [])
        dir_checks = [c for c in balance.checks if c.direction == direction]

        if frames:
            st.markdown(
                f"**Frames acting {direction}.** A frame is the set of bents "
                f"that resist together, derived from `deck_link`. Its period "
                f"comes from the frame (`K = Σkᵢ`, `M = Σmᵢ`, rigid deck), not "
                f"from one bent. Balanced *stiffness* only applies inside a "
                f"**continuous** frame; balanced *geometry* applies between "
                f"adjacent frames everywhere.")
            frame_rows = []
            for f in frames:
                row = {"frame": f.key, "bents": " + ".join(f.names),
                       "continuous": "yes" if f.continuous else "—",
                       "end_condition": f.end_conditions,
                       "M_kip_s2_in": round(f.M(), 3)}
                for i in range(f.n_bounds):
                    row[f"K [{f.label(i)}]"] = round(f.K(i), 2)
                    row[f"T [{f.label(i)}]"] = round(f.T(i), 3)
                frame_rows.append(row)
            st.dataframe(pd.DataFrame(frame_rows), width="stretch")

        if balance.bents:
            bent_rows = []
            for b in balance.bents:
                row = {"pier": b.name, "frame": b.frame,
                       "deck_link": b.deck_link,
                       "Hcol_ft": round(b.Hcol / 12, 1),
                       "silo_ft": round(b.silo / 12, 1),
                       "H_free_ft": round(b.H_free / 12, 1),
                       "end_fixity": b.end_fixity(direction),
                       "m_kip_s2_in": round(b.mass(direction), 3)}
                for i in range(len(b.k)):
                    row[f"k [{b.label(i)}]"] = round(b.stiffness(direction, i), 2)
                    row[f"T [{b.label(i)}]"] = round(b.T(direction, i), 3)
                    row[f"κ [{b.label(i)}]"] = round(
                        b.kappa(direction, i, cr.mass_normalized), 3)
                bent_rows.append(row)
            st.dataframe(pd.DataFrame(bent_rows), width="stretch")

        if dir_checks:
            st.dataframe(pd.DataFrame([{
                "check": c.name, "pair": f"{c.pair[0]}–{c.pair[1]}",
                "frame": c.scope or "—", "bound": c.bound,
                "ratio": None if np.isnan(c.ratio) else round(c.ratio, 3),
                "limit": c.limit,
                "status": "PASS" if c.passed else "FAIL",
                "values": c.note,
            } for c in dir_checks]), width="stretch")

            # --- along the bridge: each adjacent LINK drawn pass/fail ---
            k_status = _bound_status(balance, STIFFNESS_CHECK, direction)
            # geometry is keyed on FRAMES; _frame_profile_fig draws on frames,
            # so leave the keys alone rather than collapsing to a member pier
            t_status = _bound_status(balance, GEOMETRY_CHECK, direction)
            _silo_ft = {b.name: b.silo / 12.0 for b in balance.bents if b.silo > 0}
            _silo_txt = (
                "  Silo depths are on the pier axis (filled marker = silo): "
                + ", ".join(f"**{n} +{v:g} ft**" for n, v in _silo_ft.items())
                + f" — **{sum(_silo_ft.values()):g} ft total** over "
                  f"{len(_silo_ft)} pier(s)."
            ) if _silo_ft else "  No silo was added."
            st.markdown(
                f"**Along the bridge ({direction}).** Markers are piers, one "
                f"per fixity bound. A line between two piers is a check — plain "
                f"grey where it complies, dashed red with `✗ ratio` where it "
                f"does not."
                f"\n\n**Left — balanced stiffness.** The rule compares BENTS "
                f"inside one continuous frame, so markers are piers and "
                f"unlinked piers are in different frames, where the rule does "
                f"not apply. Only adjacent pairs are drawn; the any-two pairs "
                f"are in the ratio chart below."
                f"\n\n**Right — balanced frame geometry.** The rule compares "
                f"FRAMES, so markers are frames: a continuous frame is one "
                f"period drawn as a bar spanning its members, and links run "
                f"frame-to-frame."
                + _silo_txt)
            bk1, bk2 = st.columns(2)
            bk1.pyplot(_profile_fig(
                balance.bents,
                lambda b, i: b.kappa(direction, i, cr.mass_normalized),
                f"κ = {cr.kappa_symbol}", k_status))
            bk2.pyplot(_frame_profile_fig(balance.bents, frames, t_status))

            # --- the ratios themselves against their limits ---
            # The two stiffness rules have DIFFERENT limits (0.75 adjacent,
            # 0.50 any-two), so they cannot share a chart — one limit line
            # cannot judge both, and an any-two bar at 0.60 would read as a
            # failure against the 0.75 line while actually passing.
            k_adj = [c for c in dir_checks if c.name == STIFFNESS_CHECK]
            k_any = [c for c in dir_checks if c.name == STIFFNESS_ANY_CHECK]
            t_checks = [c for c in dir_checks if c.name == GEOMETRY_CHECK]
            n_bad = sum(1 for c in dir_checks if not c.passed)
            st.markdown(
                f"**Every pair against its limit ({direction}).** Bars left of "
                f"the limit line fail"
                + (f" — {n_bad} of {len(dir_checks)} here." if n_bad
                   else " — none do here.")
                + f" Each rule is charted against **its own** limit: "
                  f"{len(k_adj)} adjacent-bent pair(s) at {cr.k_ratio_min:.2f}, "
                  f"{len(k_any)} any-two pair(s) at {cr.k_ratio_any:.2f}, "
                  f"{len(t_checks)} frame-period pair(s) at "
                  f"{cr.T_ratio_min:.2f}.")
            rb1, rb2 = st.columns(2)
            if k_adj:
                rb1.pyplot(_ratio_fig(
                    k_adj, cr.k_ratio_min,
                    "Balanced stiffness — adjacent bents (SDC 7.1.2 / SGS 4.1.2)"))
            else:
                rb1.info("No balanced-stiffness check in this direction: every "
                         "frame holds a single bent, so there is nothing to "
                         "compare inside one.")
            if t_checks:
                rb2.pyplot(_ratio_fig(
                    t_checks, cr.T_ratio_min,
                    "Balanced frame geometry (SDC 7.1.3 / SGS 4.1.3)"))
            if k_any:
                # separate row: the any-two list grows as n^2 in a long frame
                st.pyplot(_ratio_fig(
                    k_any, cr.k_ratio_any,
                    f"Balanced stiffness — any two bents in a frame "
                    f"(limit {cr.k_ratio_any:.2f}, looser than the "
                    f"{cr.k_ratio_min:.2f} adjacent rule)"))

        if balance.log:
            # open by default whenever the tool changed a design or failed —
            # the user needs to see what was altered on their behalf.
            with st.expander("Balancing log (what the silo search did)",
                             expanded=bool(siloed) or not balance.passed):
                for entry in balance.log:
                    st.markdown(f"- {entry}")

        st.download_button("Download balance report (Markdown)",
                           balance_report(balance).encode("utf-8"),
                           "balance_report.md", "text/markdown")

    # ------------------------------------------------------------------
    frame_checks = st.session_state.get("frame_checks") or []
    if frame_checks:
        st.subheader("Frame displacement check — real end conditions")
        st.markdown(
            "The per-bent suite above designs every column as a stand-alone "
            "**fixed-free cantilever** on its own period. For a bent that is "
            "*integral* with a continuous deck that is wrong longitudinally on "
            "both counts. Here the demand comes from the **frame period** and "
            "the capacity from the **frame's sway mechanism**, with each member "
            "at its real end condition. Where this disagrees with the per-bent "
            "result, this is the governing check for those bents.")

        worst = min(fc.worst for fc in frame_checks)
        all_ok = all(fc.passed for fc in frame_checks)
        m1, m2, m3 = st.columns(3)
        m1.metric("Frame check", "PASS" if all_ok else "FAIL")
        m2.metric("Worst Δc/Δd", f"{worst:.2f}")
        m3.metric("Frames checked", f"{len(frame_checks)}")

        if not all_ok:
            # a member can fail on capacity or on P-Δ; the headline ratio does
            # not say which, so name the reasons.
            bad = [m for fc in frame_checks for m in fc.members if not m.passed]
            reasons = []
            if [m.name for m in bad if m.ratio < 1.0]:
                reasons.append("**Δc < Δd** at "
                               + ", ".join(m.name for m in bad if m.ratio < 1.0))
            if [m.name for m in bad if not m.pdelta_ok]:
                reasons.append("**P-Δ** at "
                               + ", ".join(m.name for m in bad
                                           if not m.pdelta_ok))
            st.error("Failing because of " + "; ".join(reasons) + ".")

        for fc in frame_checks:
            head = (f"{fc.direction.capitalize()} · {fc.frame_key} "
                    f"({' + '.join(fc.member_names)}) · {fc.end_conditions} · "
                    f"T = {fc.T:.3f} s · Δd = {fc.delta_d:.2f} in · "
                    f"{'PASS' if fc.passed else 'FAIL'}")
            with st.expander(head, expanded=not fc.passed):
                st.markdown("\n".join([
                    f"- `K_frame = Σkᵢ = "
                    f"{' + '.join(f'{m.k:.1f}' for m in fc.members)} "
                    f"= {fc.K:.1f} kip/in`  — each member at its own end "
                    f"condition ({fc.end_conditions})",
                    f"- `M_frame = Σmᵢ = "
                    f"{' + '.join(f'{m.m:.3f}' for m in fc.members)} "
                    f"= {fc.M:.4f} kip·s²/in`  (W = {fc.W:.0f} kip)",
                    f"- `T_frame = 2π·√(M/K) = 2π·√({fc.M:.4f}/{fc.K:.1f}) "
                    f"= {fc.T:.3f} s`  →  `Sa = {fc.Sa:.4f} g`  →  "
                    f"`Δd = {fc.delta_d:.2f} in`, shared by every member",
                ]))
                st.dataframe(pd.DataFrame([{
                    "Member": m.name,
                    "End cond.": ("fixed-fixed" if m.end_fixity == "fixed"
                                  else "fixed-free"),
                    "Sway mechanism": m.mechanism,
                    "V_mech (kip)": round(m.V_mech),
                    "Δy (in)": round(m.delta_y, 2),
                    "Δp (in)": round(m.delta_p, 2),
                    "Δc (in)": round(m.delta_c, 2),
                    "Δd (in)": round(m.delta_d, 2),
                    "Δc/Δd": round(m.ratio, 2),
                    "μd": round(m.mu_d, 2),
                    "P-Δ": "OK" if m.pdelta_ok else "NG",
                    "Status": "PASS" if m.passed else "FAIL",
                } for m in fc.members]), width="stretch", hide_index=True)

                if any(m.end_fixity == "fixed" for m in fc.members):
                    st.markdown(
                        "**Shaft capacity-design demand at this mechanism.** "
                        "The shaft is held elastic by design, so it does not "
                        "compete to hinge — it must be sized for the column's "
                        "overstrength demand. Both hinges of a column "
                        "mechanism sit at Mp_col, so the interface *moment* is "
                        "the same Mo the fixed-free suite used. The *shear* is "
                        "not.")
                    st.dataframe(pd.DataFrame([{
                        "Member": m.name,
                        "Mo interface (kip-ft)": round(m.Mo_interface / 12),
                        "Vo interface (kip)": round(m.Vo_interface),
                        "Vo fixed-free (kip)": round(m.Vo_cantilever),
                        "Amplification": f"{m.shear_amplification:.2f}×",
                        "M below ground (kip-ft)": (
                            None if m.shaft_solution is None
                            else round(m.shaft_moment / 12)),
                        "Mp shaft (kip-ft)": round(m.shaft_Mp / 12),
                        "D/C": (None if m.shaft_solution is None
                                else round(m.shaft_dc, 2)),
                        "Shaft": ("—" if m.shaft_solution is None
                                  else ("YIELDS" if m.shaft_dc > 1 else "OK")),
                    } for m in fc.members]), width="stretch", hide_index=True)
                    st.caption(
                        "The below-ground columns are a p-y solve at this "
                        "mechanism's head condition — V = Vo with M = Mo "
                        "applied at the head, so the interface lands on Mo "
                        "rather than the 2·Mo that applying the shear alone "
                        "would give. The sign is verified against the "
                        "interface moment on every solve. A D/C above 1.00 is "
                        "reported, not failed: this is closed-form plastic "
                        "analysis, not a pushover.")

                st.caption(
                    "Moment diagram — the shear that brings each candidate "
                    "section to its Mp, elastically. Hinges are marked in "
                    "order of formation."
                    + (" The second forms on the redistributed diagram, so "
                       "its mechanism load differs from the elastic figure "
                       "here." if any(len(m.hinges) > 1 for m in fc.members)
                       else ""))
                md_rows = []
                for m in fc.members:
                    order = {h.name: i for i, h in enumerate(m.hinges)}
                    for sec in m.sections:
                        i = order.get(sec.name)
                        md_rows.append({
                            "Member": m.name, "Section": sec.name,
                            "x from deck (in)": round(sec.x),
                            "lever |M|/V (in)": round(sec.arm, 1),
                            "Mp (kip-ft)": round(sec.Mp / 12),
                            "V to yield (kip)": (
                                None if not math.isfinite(sec.V_yield)
                                else round(sec.V_yield)),
                            "Hinge": "" if i is None else f"{i + 1}",
                        })
                st.dataframe(pd.DataFrame(md_rows), width="stretch",
                             hide_index=True)

                for m in fc.members:
                    for w in m.warnings:
                        st.warning(f"**{m.name}** — {w}")

        st.caption("Closed-form plastic analysis, not an incremental pushover: "
                   "it gives the first yielding section and the load at which a "
                   "mechanism forms, not the hinging sequence or "
                   "post-mechanism response. Rigid deck. Δy is the bilinear "
                   "idealisation. See the report for the full assumption list.")
        st.download_button("Download frame check (Markdown)",
                           frame_seismic_report(frame_checks).encode("utf-8"),
                           "frame_check.md", "text/markdown")

    if results:
        st.subheader("Drill-down")
        names = [r.name for r in results]
        sel = st.selectbox("Select column", names)
        rr = next(r for r in results if r.name == sel)

        # --- what actually differs by direction, and what does not ---
        _dirs = rr.assessment.directions
        if len(_dirs) > 1:
            st.markdown(
                "**Both directions are checked and the row passes only if "
                "both do.** Capacity here is direction-independent — the "
                "section is axisymmetric and the p-y solves run at "
                "`F_y = Mp/H_free` and `Vo = Mo/H_free`, none of which contain "
                "the mass — so only the *tributary mass* differs, and with it "
                "the period, Sa, Δd and everything downstream.")
            rows = []
            for dname, dres in _dirs.items():
                g = dres.governing_bound
                rows.append({
                    "Direction": dname,
                    "W entered (kip)": round(dres.weight_entered),
                    "W + self-wt (kip)": round(dres.weight_mass),
                    "T (s)": round(g.demand.period, 3),
                    "Sa (g)": round(g.demand.Sa, 4),
                    "Δd (in)": round(g.demand.disp_demand, 2),
                    "Δc (in)": round(g.delta_c, 2),
                    "Δc/Δd": round(g.delta_c / g.demand.disp_demand, 2),
                    "μd": round(g.mu_demand, 2),
                    "Status": "PASS" if dres.passed else "FAIL",
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            # which checks the two directions actually disagree on
            _names = sorted({c.name for c in rr.assessment.checks})
            _diff = []
            for n in _names:
                per = {d: next((c for c in o.checks if c.name == n), None)
                       for d, o in _dirs.items()}
                vals = [c.ratio for c in per.values() if c is not None]
                if len(vals) > 1 and abs(max(vals) - min(vals)) > 1e-9:
                    _diff.append({"Check": n, **{
                        f"{d} D/C": round(c.ratio, 4)
                        for d, c in per.items() if c is not None},
                        "Governs": max(per, key=lambda d: per[d].ratio)})
            if _diff:
                with st.expander(f"Checks that differ by direction "
                                 f"({len(_diff)} of {len(_names)})"):
                    st.dataframe(pd.DataFrame(_diff), width="stretch",
                                 hide_index=True)
                    st.caption(
                        "The rest come out numerically identical because they "
                        "are driven by Mp/Mo and geometry rather than mass. "
                        "Column shear can also tie when α' saturates at its "
                        "3.0 clamp, where a small μd change no longer moves it.")
            if rr.frame and any(len(f.member_names) > 1
                                for f in (st.session_state.get("frame_checks")
                                          or [])
                                if rr.name in f.member_names):
                st.info(
                    f"**{rr.name} shares a frame**, so the periods above are "
                    f"the FRAME's, not this bent's own — the deck ties the "
                    f"members together and they are pushed to the same "
                    f"displacement. An integral bent is fixed-fixed "
                    f"longitudinally too, so its capacity there comes from the "
                    f"two-hinge mechanism rather than the cantilever. A bent "
                    f"in a frame of one is unaffected: the frame's K and W are "
                    f"its own.")

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
            # one figure, both directions — the shift between them is the point
            _mk = {"longitudinal": "o", "transverse": "s"}
            for dname, dres in (rr.assessment.directions.items()
                                or [("", None)]):
                for b in dres.bounds:
                    ax2.plot(b.demand.period, b.demand.Sa,
                             _mk.get(dname, "o"), ms=7,
                             label=f"{dname[:5]}. mult {b.multiplier:g}: "
                                   f"T={b.demand.period:.2f}s")
            if not rr.assessment.directions:
                for b in rr.assessment.bounds:
                    ax2.plot(b.demand.period, b.demand.Sa, "o",
                             label=f"mult {b.multiplier:g}: "
                                   f"T={b.demand.period:.2f}s")
            ax2.set_xlabel("period T (s)")
            ax2.set_ylabel("Sa (g)")
            ax2.set_title("Spectra & effective periods")
            ax2.legend(fontsize=7)
            st.pyplot(fig2)
            st.caption("Same stiffness, different tributary mass — so the "
                       "directions sit at different periods on the same "
                       "spectrum. Circles longitudinal, squares transverse.")

        # p-y pile response: deflection / shear / moment diagrams (soil fixity)
        ig = rr.assessment.inground_solution
        soil_bounds = [b for b in rr.assessment.bounds if b.soil_solution]
        if ig is not None or soil_bounds:
            st.markdown("**Pile response diagrams (p-y)** — distance below the "
                        "column top; ground line (top of shaft) dashed. Solid = "
                        "**shaft-design demand at column overstrength Mo**; "
                        "dashed = yield-level stiffness bounds. "
                        "**Direction-independent:** these solve at "
                        "`F_y = Mp/H_free` and `Vo = Mo/H_free`, neither of "
                        "which contains the tributary mass, so there is one "
                        "set of diagrams, not one per direction.")
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

        _views = ["envelope", *rr.assessment.directions]
        _view = "envelope"
        if len(rr.assessment.directions) > 1:
            _view = st.radio(
                "Checks shown", _views, horizontal=True,
                key=f"report_view_{rr.name}", format_func=str.capitalize,
                help="The row passes or fails on the ENVELOPE — the worse of "
                     "both directions. The single-direction views are for "
                     "inspection; only the checks change, since the section, "
                     "moment-curvature, p-y and shaft-protection numbers are "
                     "direction-independent.")
        report_md = column_report(rr, view=_view)
        st.markdown(report_md)
        st.download_button("Download report (Markdown)", report_md.encode(),
                           f"report_{rr.name}_{_view}.md", "text/markdown")
