"""Tabular batch input/output schema and helpers.

Each row of the batch table describes one column (a simply-supported span
support).  Global settings (design spectrum, materials, optimiser priority) are
shared across the whole batch and are held separately in :class:`GlobalConfig`.

Length inputs in the table:
    Hcol_ft            : column height, ft
    D_shaft_in         : shaft diameter, in
    Dcol_in            : starting/fixed column diameter, in
    all *_in spacings  : in
Loads:
    weight_kip         : tributary weight, kip
    axial_kip          : axial dead load, kip
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd

from .demand import SpectrumSpec

# Batch table columns (order preserved for display/export)
COLUMNS: tuple[str, ...] = (
    "name",
    "Hcol_ft",
    "D_shaft_in",
    "weight_kip",
    "axial_kip",
    "Dcol_in",
    "fc_ksi",
    "cover_in",
    "n_bars",
    "long_bar_no",
    "long_bundle",
    "spiral_bar_no",
    "spiral_spacing_in",
    "spiral_bundle",
    "mult_lb",
    "mult_ub",
    # shaft (capacity-protected) reinforcement
    "shaft_fc_ksi",
    "shaft_cover_in",
    "shaft_n_bars",
    "shaft_long_bar_no",
    "shaft_long_bundle",
    "shaft_spiral_bar_no",
    "shaft_spiral_spacing_in",
    "shaft_spiral_bundle",
)

NUMERIC_COLUMNS = tuple(c for c in COLUMNS if c != "name")

# Human-friendly labels and help text for each table column (used by the GUI).
COLUMN_META: dict[str, tuple[str, str]] = {
    "name": ("Column ID", "A label for this column / bent (e.g. Pier 3)."),
    "Hcol_ft": ("Column height (ft)",
                "Clear height from top of shaft to the point of load / "
                "contraflexure (deck level)."),
    "D_shaft_in": ("Shaft dia. (in)",
                   "Type II shaft (enlarged pile) outside diameter."),
    "weight_kip": ("Seismic weight W (kip)",
                   "Tributary weight this column carries that participates as "
                   "seismic MASS (drives period and displacement demand). "
                   "Superstructure dead load in the tributary span + cap."),
    "axial_kip": ("Axial load P (kip)",
                  "Sustained axial COMPRESSION on the column section used for "
                  "moment-curvature, P-Delta and shear (the P in the P-M "
                  "interaction). Often close to W but not identical - e.g. "
                  "excludes non-tributary effects, includes column self-weight."),
    "Dcol_in": ("Column dia. (in)",
                "Starting (or fixed) column diameter. The optimiser may grow "
                "this if 'diameter' is a variable parameter."),
    "fc_ksi": ("Column f'c (ksi)", "Column concrete compressive strength."),
    "cover_in": ("Column cover (in)", "Clear cover to the spiral/hoop."),
    "n_bars": ("Long. bar count", "Number of longitudinal bars in the column."),
    "long_bar_no": ("Long. bar #", "Longitudinal bar size (US # designation)."),
    "long_bundle": ("Long. bundle", "Bars per longitudinal bundle (1 = single). "
                    "Set by the optimiser when bundling is allowed."),
    "spiral_bar_no": ("Spiral bar #", "Transverse spiral/hoop bar size (US #)."),
    "spiral_spacing_in": ("Spiral pitch (in)", "Centre-to-centre spiral pitch."),
    "spiral_bundle": ("Spiral bundle", "Bars per spiral/hoop bundle (1 = single)."),
    "mult_lb": ("Fixity mult. (upper stiffness)",
                "Depth-to-fixity = this x shaft dia. Smaller = stiffer "
                "(upper-bound stiffness). Default 3."),
    "mult_ub": ("Fixity mult. (lower stiffness)",
                "Depth-to-fixity = this x shaft dia. Larger = softer "
                "(lower-bound stiffness). Default 6."),
    "shaft_fc_ksi": ("Shaft f'c (ksi)", "Shaft concrete strength."),
    "shaft_cover_in": ("Shaft cover (in)", "Shaft clear cover to transverse steel."),
    "shaft_n_bars": ("Shaft long. count", "Number of shaft longitudinal bars."),
    "shaft_long_bar_no": ("Shaft long. bar #", "Shaft longitudinal bar size (US #)."),
    "shaft_long_bundle": ("Shaft long. bundle", "Bars per shaft longitudinal "
                          "bundle (1 = single)."),
    "shaft_spiral_bar_no": ("Shaft spiral #", "Shaft transverse bar size (US #)."),
    "shaft_spiral_spacing_in": ("Shaft spiral pitch (in)",
                                "Shaft transverse steel centre-to-centre pitch."),
    "shaft_spiral_bundle": ("Shaft spiral bundle",
                            "Bars per shaft spiral/hoop bundle (1 = single)."),
}


@dataclass
class GlobalConfig:
    """Batch-wide settings shared by every column."""

    design_spectrum: SpectrumSpec = field(default_factory=SpectrumSpec)
    lle_spectrum: SpectrumSpec | None = None   # low-level (elastic) earthquake
    lle_mu_limit: float = 1.0
    code: str = "SDC 2.1"                        # design code provisions key
    fye: float = 68.0
    fue: float = 95.0
    fyh: float = 68.0
    optimize: bool = True
    priority: tuple[str, ...] = ("longitudinal", "confinement", "diameter", "fc")
    variable: tuple[str, ...] = ("longitudinal", "confinement", "diameter", "fc")
    shaft_moment_basis: str = "interface"
    mu_d_limit: float = 5.0
    rho_l_min: float = 0.01
    rho_l_max: float = 0.04
    min_bar_spacing: float = 6.0        # min c/c longitudinal spacing, in
    allow_bundling: bool = False        # allow 2-bar longitudinal bundles
    concrete_unit_weight: float = 0.150  # kcf (kip/ft^3)
    self_weight_mass_factor: float = 1.0 / 3.0   # fraction of col self-wt in seismic mass
    self_weight_in_axial: bool = True    # add col self-wt to axial P


def default_row(name: str = "C1") -> dict:
    """A sensible starting row."""
    return {
        "name": name,
        "Hcol_ft": 22.0,
        "D_shaft_in": 84.0,
        "weight_kip": 800.0,
        "axial_kip": 800.0,
        "Dcol_in": 48.0,
        "fc_ksi": 4.0,
        "cover_in": 2.0,
        "n_bars": 16,
        "long_bar_no": 9,
        "long_bundle": 1,
        "spiral_bar_no": 5,
        "spiral_spacing_in": 4.0,
        "spiral_bundle": 1,
        "mult_lb": 3.0,
        "mult_ub": 6.0,
        "shaft_fc_ksi": 4.0,
        "shaft_cover_in": 3.0,
        "shaft_n_bars": 36,
        "shaft_long_bar_no": 11,
        "shaft_long_bundle": 1,
        "shaft_spiral_bar_no": 6,
        "shaft_spiral_spacing_in": 4.0,
        "shaft_spiral_bundle": 1,
    }


def default_dataframe(n: int = 3) -> pd.DataFrame:
    """A starter batch with ``n`` rows and varying heights/masses."""
    rows = []
    for i in range(n):
        r = default_row(f"C{i+1}")
        r["Hcol_ft"] = 18.0 + 4.0 * i
        r["weight_kip"] = 700.0 + 100.0 * i
        r["axial_kip"] = 700.0 + 100.0 * i
        rows.append(r)
    return pd.DataFrame(rows, columns=list(COLUMNS))


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a batch table from CSV or Excel and coerce column types."""
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)
    return validate(df)


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Write a batch table to CSV or Excel based on the file extension."""
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        df.to_excel(p, index=False)
    else:
        df.to_csv(p, index=False)


def validate(df: pd.DataFrame, min_shaft_oversize: float = 0.0) -> pd.DataFrame:
    """Validate and normalise a batch table, filling defaults for missing cols.

    ``min_shaft_oversize`` is the required ``D_shaft - Dcol`` in inches: 0 for
    AASHTO SGS ("larger in diameter", Owner's discretion) and 24 for Caltrans
    SDC, whose Type II definition demands at least 24 in.
    """
    df = df.copy()
    missing_required = {"Hcol_ft", "D_shaft_in", "weight_kip", "axial_kip", "Dcol_in"}
    absent = missing_required - set(df.columns)
    if absent:
        raise ValueError(f"Batch table missing required columns: {sorted(absent)}")

    defaults = default_row()
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = defaults[col]
    df = df[list(COLUMNS)]

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    int_cols = ("n_bars", "long_bar_no", "long_bundle", "spiral_bar_no",
                "spiral_bundle", "shaft_n_bars", "shaft_long_bar_no",
                "shaft_long_bundle", "shaft_spiral_bar_no", "shaft_spiral_bundle")
    for col in int_cols:
        df[col] = df[col].round().astype("Int64")

    if df[list(NUMERIC_COLUMNS)].isna().any().any():
        bad = df[df[list(NUMERIC_COLUMNS)].isna().any(axis=1)].index.tolist()
        raise ValueError(f"Non-numeric or missing values in rows: {bad}")

    # An "oversized" (Type II) shaft is by definition larger in diameter than
    # the column it supports (AASHTO SGS, Section 2 definitions).  The whole
    # model — hinge held in the column at the top of shaft, capacity protection
    # per SGS 8.9 / 8.8.12 — depends on it.
    gap = df["D_shaft_in"] - df["Dcol_in"]
    bad = df[gap <= max(min_shaft_oversize, 0.0)] if min_shaft_oversize <= 0         else df[gap < min_shaft_oversize]
    if not bad.empty:
        need = (f"at least {min_shaft_oversize:g} in larger than"
                if min_shaft_oversize > 0 else "larger than")
        rows = ", ".join(
            f"{r['name']} (shaft {r['D_shaft_in']:g}, column {r['Dcol_in']:g} in)"
            for _, r in bad.iterrows())
        raise ValueError(
            f"Type II shaft diameter must be {need} the column diameter: {rows}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Project persistence (save / re-open)
# ---------------------------------------------------------------------------
PROJECT_VERSION = 1


def config_to_dict(cfg: GlobalConfig) -> dict:
    """Serialise a GlobalConfig (with nested spectra) to a plain dict."""
    d = asdict(cfg)
    # asdict already expands SpectrumSpec dataclasses; ensure lists (JSON-safe)
    for key in ("design_spectrum", "lle_spectrum"):
        spec = d.get(key)
        if spec is not None:
            spec["periods"] = list(spec.get("periods", []))
            spec["accels"] = list(spec.get("accels", []))
    d["priority"] = list(d["priority"])
    d["variable"] = list(d["variable"])
    return d


def config_from_dict(d: dict) -> GlobalConfig:
    """Reconstruct a GlobalConfig from a plain dict."""
    d = dict(d)
    ds = d.get("design_spectrum")
    d["design_spectrum"] = SpectrumSpec(
        kind=ds.get("kind", "parametric"), Sds=ds.get("Sds", 1.0),
        Sd1=ds.get("Sd1", 0.6), periods=tuple(ds.get("periods", [])),
        accels=tuple(ds.get("accels", [])),
    ) if ds else SpectrumSpec()
    ls = d.get("lle_spectrum")
    d["lle_spectrum"] = SpectrumSpec(
        kind=ls.get("kind", "parametric"), Sds=ls.get("Sds", 1.0),
        Sd1=ls.get("Sd1", 0.6), periods=tuple(ls.get("periods", [])),
        accels=tuple(ls.get("accels", [])),
    ) if ls else None
    d["priority"] = tuple(d.get("priority", ()))
    d["variable"] = tuple(d.get("variable", ()))
    valid = {f for f in GlobalConfig.__dataclass_fields__}
    return GlobalConfig(**{k: v for k, v in d.items() if k in valid})


def project_to_json(df: pd.DataFrame, cfg: GlobalConfig) -> str:
    """Serialise the whole project (batch table + settings) to a JSON string."""
    payload = {
        "version": PROJECT_VERSION,
        "config": config_to_dict(cfg),
        "columns": validate(df).astype(object).where(pd.notna(validate(df)), None)
                    .to_dict(orient="records"),
    }
    return json.dumps(payload, indent=2)


def project_from_json(text: str) -> tuple[pd.DataFrame, GlobalConfig]:
    """Load a project from a JSON string -> (batch DataFrame, GlobalConfig)."""
    payload = json.loads(text)
    cfg = config_from_dict(payload.get("config", {}))
    df = validate(pd.DataFrame(payload.get("columns", [])))
    return df, cfg


def save_project(path: str | Path, df: pd.DataFrame, cfg: GlobalConfig) -> None:
    """Write the project to a ``.json`` file."""
    Path(path).write_text(project_to_json(df, cfg), encoding="utf-8")


def load_project(path: str | Path) -> tuple[pd.DataFrame, GlobalConfig]:
    """Read a project ``.json`` file -> (batch DataFrame, GlobalConfig)."""
    return project_from_json(Path(path).read_text(encoding="utf-8"))
