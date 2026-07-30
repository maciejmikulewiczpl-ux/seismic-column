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
    weight_long_kip    : tributary weight restrained longitudinally, kip
    weight_trans_kip   : tributary weight restrained transversely, kip
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
    "frame",
    "deck_link",
    "Hcol_ft",
    "silo_ft",
    "n_columns",
    "col_spacing_ft",
    "cap_fixity",
    "D_shaft_in",
    "weight_long_kip",
    "weight_trans_kip",
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

TEXT_COLUMNS: tuple[str, ...] = ("name", "frame", "deck_link", "cap_fixity")

# How the columns of a MULTI-column bent meet the cap, TRANSVERSELY.  (The
# longitudinal head condition comes from ``deck_link``, so "pinned one way,
# moment the other" is expressed by setting the two independently.)
#
#   fixed   monolithic cap: hinge at the cap AND the base, so the bent is a
#           portal frame -- fixed-fixed, V = 2*Mp/H, and the overturning
#           develops an axial push/pull couple of sum(Mo) between the columns.
#   pinned  a detailed pin: hinge at the base only, so each column is an
#           INDEPENDENT cantilever -- fixed-free, V = Mp/H, and NO couple at
#           all.  Statics: V*H = sum(Mo) and the base moments are sum(Mo), so
#           the couple is exactly zero.
#
# Pinning therefore does three things at once -- roughly quarters the transverse
# stiffness, halves Vo, and removes the axial swing -- which makes it a powerful
# lever when a fixed-cap bent will not work.
CAP_FIXITIES: tuple[str, ...] = ("fixed", "pinned", "pinned_long",
                                "pinned_trans")


def head_moment_connection(cap_fixity: str, direction: str) -> bool:
    """Does the COLUMN-TO-CAP connection transmit moment in ``direction``?

    A detail can be a moment connection one way and a pin the other, which is
    why this is per direction:

        fixed         moment both ways (monolithic) -- the default
        pinned        a pin both ways
        pinned_long   pinned longitudinally, moment transversely
        pinned_trans  moment longitudinally, pinned transversely

    This is only HALF of what fixes a column head.  A moment connection needs
    something above able to resist it: longitudinally the deck has to be
    integral, and transversely there has to be a cap spanning to a second
    column.  A pin, though, is sufficient on its own to release the head --
    whatever sits above cannot hold a rotation the connection does not carry.
    """
    cf = (cap_fixity or "fixed").strip().lower()
    if cf == "pinned":
        return False
    if cf == "pinned_long":
        return direction != LONGITUDINAL
    if cf == "pinned_trans":
        return direction != TRANSVERSE
    return True
NUMERIC_COLUMNS = tuple(c for c in COLUMNS if c not in TEXT_COLUMNS)

# How the deck attaches at a bent.  This is the physical detail; both which
# directions the bent resists AND its end condition are DERIVED from it (see
# ``balance.frames_for`` and ``BentStiffness.end_fixity``):
#
#                     resists long. | resists trans. | end condition
#   integral   monolithic moment        yes                yes         FIXED-FIXED long.,
#              connection                                              fixed-free trans.
#   pinned     bearings restrained      yes                yes         fixed-free both
#              both ways (no moment)
#   bearing    released long.,          no                 yes         fixed-free
#              shear key trans.
#   free       unrestrained             no                 no          --
#
# Only an INTEGRAL bent is fixed-fixed, and only longitudinally: the deep
# continuous girder holds the column head against rotation as the frame sways
# along the bridge.  Transversely even an integral single-column bent behaves as
# a cantilever, so it stays fixed-free.  `pinned` is the default because a
# simply supported span sits on bearings and transmits no moment.
DECK_LINKS: tuple[str, ...] = ("integral", "pinned", "bearing", "free")
# What bridge engineers actually say, mapped to the stored vocabulary.  A "pin"
# and a "roller" are both bearings, so calling only the released one `bearing`
# reads oddly at a table; accept either word and keep whichever was typed.
DECK_LINK_ALIASES: dict[str, str] = {
    "roller": "bearing", "expansion": "bearing", "exp": "bearing",
    "guided": "bearing", "released": "bearing",
    "pin": "pinned", "fixed": "pinned",
    "monolithic": "integral",
}


def deck_link_word(link: str) -> str:
    """How to SHOW a stored link -- the word an engineer would use."""
    return {"bearing": "roller (expansion)", "pinned": "pin (fixed bearing)",
            "integral": "integral", "free": "free"}.get(link, link)

# Longitudinal / transverse.  Used as dict keys and check labels throughout.
LONGITUDINAL, TRANSVERSE = "longitudinal", "transverse"
DIRECTIONS: tuple[str, ...] = (LONGITUDINAL, TRANSVERSE)

# A ``frame`` cell holding one of these means "leave this pier out of the
# balanced-stiffness / balanced-geometry adjacency checks".
NO_FRAME: frozenset[str] = frozenset({"", "-", "none", "nan", "na"})


def frame_keys(frame: object) -> tuple[str, ...]:
    """The frame(s) this pier belongs to, in the order entered.

    A pier under an expansion joint carries the last span of one frame and the
    first span of the next, so it genuinely belongs to BOTH.  Name them in one
    cell -- ``"F1, F2"`` -- and it is a member of each, bringing its full
    stiffness to both and splitting its tributary mass between them.  A single
    name is the ordinary case and behaves exactly as it always has.
    """
    txt = str(frame).strip()
    if txt.lower() in NO_FRAME:
        return ()
    keys = [k.strip() for k in txt.replace(";", ",").split(",")]
    out: list[str] = []
    for k in keys:                      # keep order, drop blanks and repeats
        if k and k.lower() not in NO_FRAME and k not in out:
            out.append(k)
    return tuple(out)


def in_frame(frame: object) -> bool:
    """True if this ``frame`` value takes part in the balance checks."""
    return bool(frame_keys(frame))


def deck_links(link: object, n_frames: int = 1) -> tuple[str, ...]:
    """How the deck attaches, ONE ENTRY PER FRAME this pier carries.

    A pier between two decks can meet them differently -- a free bearing under
    the end of a continuous frame and a pin under the simple span beside it is
    the ordinary case, and a single value cannot say so.  Give one link per
    frame, in the same order: ``frame = "FA7, C1"`` with
    ``deck_link = "pinned, bearing"`` reads *pinned to FA7, free bearing to C1*.

    A single value is broadcast to every frame, which is what every table
    before this meant.
    """
    txt = str(link).strip().lower()
    parts = [k.strip() for k in txt.replace(";", ",").split(",") if k.strip()]
    if not parts:
        parts = ["pinned"]
    parts = [DECK_LINK_ALIASES.get(k, k) for k in parts]
    if len(parts) == 1:
        return tuple(parts * max(n_frames, 1))
    return tuple(parts)

# Human-friendly labels and help text for each table column (used by the GUI).
COLUMN_META: dict[str, tuple[str, str]] = {
    "name": ("Column ID", "A label for this column / bent (e.g. Pier 3)."),
    "frame": ("Frame / structure",
              "Groups piers for the balanced-stiffness and balanced-geometry "
              "checks. Piers sharing a frame are compared in TABLE ROW ORDER, "
              "so consecutive rows are treated as adjacent. Leave blank or "
              "enter '-' to exclude this pier from those checks. A pier under "
              "an EXPANSION JOINT carries the last span of one frame and the "
              "first span of the next, so it belongs to both - name them both "
              "here, 'F1, F2'. It then joins each frame with its FULL stiffness "
              "and HALF its tributary mass. Where you name only one, a bearing "
              "pier between two continuous frames is shared automatically; "
              "naming both is clearer and always wins."),
    "n_columns": ("Columns in bent",
                  "How many columns this bent carries (1 = single-column). More "
                  "than one makes the bent a PORTAL FRAME transversely: the cap "
                  "restrains the column heads (fixed-fixed, so each develops "
                  "2*Mp/H) and pushing the bent overturns it, resisted by an "
                  "axial push/pull couple between the columns. That changes the "
                  "axial, and so Mp, Vo, Df and capacity, at each position. "
                  "Longitudinally the columns simply act in parallel. The cap "
                  "beam and the column-to-cap joint are NOT checked."),
    "cap_fixity": ("Column-to-cap (transverse)",
                   "How the columns meet the cap, TRANSVERSELY (longitudinally "
                   "the head condition comes from 'Deck connection', so a pin "
                   "one way and a moment connection the other is set with the "
                   "two together). 'fixed' = monolithic cap, so the bent is a "
                   "portal frame: fixed-fixed, Vo = 2*Mo/H, and the overturning "
                   "develops an axial push/pull couple between the columns. "
                   "'pinned' = a detailed pin, so each column is an INDEPENDENT "
                   "cantilever: fixed-free, Vo = Mo/H, and NO couple at all. "
                   "'pinned_long' / 'pinned_trans' pin ONE direction and keep "
                   "the moment connection in the other. A pin releases the "
                   "column head on its own; a moment connection only fixes it "
                   "if something above can resist — an integral deck "
                   "longitudinally, or a cap spanning to a second column "
                   "transversely. "
                   "Pinning roughly quarters the transverse stiffness, halves "
                   "Vo and removes the axial swing. Ignored if the bent has one "
                   "column."),
    "col_spacing_ft": ("Column spacing (ft)",
                       "Centre-to-centre spacing, needed only when the bent has "
                       "more than one column. This is the LEVER ARM of the "
                       "transverse push/pull couple, so closer columns take a "
                       "larger axial swing — close enough spacing puts the "
                       "windward column into net tension, where the concrete "
                       "shear term vc drops to zero."),
    "Hcol_ft": ("Column height (ft)",
                "Clear height to the point of load / contraflexure (deck "
                "level), measured from the top of shaft — or, if there is a "
                "column silo, from the original ground line at the top of the "
                "silo. EXCLUDES the silo; the free length is Hcol + silo."),
    "silo_ft": ("Column silo (ft)",
                "Column silo / isolation casing depth. Lowers the top of shaft "
                "by this much and lengthens the free column by the same amount "
                "(free length = Hcol + silo); the embedded shaft length is "
                "unchanged. Used to soften a stiff pier for the balanced-"
                "stiffness rule. An entered value is a MINIMUM - the auto-silo "
                "may increase it, never reduce it."),
    "D_shaft_in": ("Shaft dia. (in)",
                   "Type II shaft (enlarged pile) outside diameter."),
    "weight_long_kip": ("Seismic weight W, long. (kip)",
                        "Tributary weight this bent restrains LONGITUDINALLY. "
                        "A bent on a longitudinally-released bearing carries "
                        "none of the deck it releases - only the span it is "
                        "fixed to. This is the weight the per-bent seismic "
                        "checks use (drives period and displacement demand)."),
    "weight_trans_kip": ("Seismic weight W, trans. (kip)",
                         "Tributary weight this bent restrains TRANSVERSELY. "
                         "Shear keys engage a bent transversely even where the "
                         "bearing is released longitudinally, so this normally "
                         "includes half of each adjacent span. At an expansion "
                         "joint those two half-spans belong to DIFFERENT frames, "
                         "and the balance checks split this figure 50/50 between "
                         "them - exact where the adjacent spans match, "
                         "approximate where they do not. Blank = same as "
                         "the longitudinal weight. Used by the balance checks "
                         "only."),
    "deck_link": ("Deck connection",
                  "How the deck attaches here. 'integral' = monolithic moment "
                  "connection, so the bent is FIXED-FIXED longitudinally "
                  "(12EI/L3) and fixed-free transversely. 'pinned' = bearings "
                  "restrained both ways but no moment, fixed-free both ways - "
                  "the normal simply supported case, and the default. "
                  "'bearing' = released longitudinally with a shear key, so it "
                  "resists transversely only - it carries NO deck longitudinally "
                  "(only its own cap and column self weight), so it takes no part "
                  "in the LONGITUDINAL balance checks at all, and transversely it "
                  "splits its tributary mass 50/50 with the frame across the "
                  "joint. 'free' = resists neither. Drives "
                  "both the frame layout and the end condition in each "
                  "direction."),
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
    min_shaft_oversize_in: float = 24.0  # optimiser keeps shaft >= column + this
    # optimiser objective:
    # "min_diameter" | "target_steel" | "min_steel" | "fixed_diameter"
    optimize_objective: str = "min_diameter"
    target_rho_l: float = 0.02           # longitudinal ratio for "target_steel"
    concrete_unit_weight: float = 0.150  # kcf (kip/ft^3)
    self_weight_mass_factor: float = 1.0 / 3.0   # fraction of col self-wt in seismic mass
    self_weight_in_axial: bool = True    # add col self-wt to axial P
    # --- balanced stiffness (SDC 7.1.2 / SGS 4.1.2) & balanced frame geometry
    #     (SDC 7.1.3 / SGS 4.1.3), between ADJACENT piers ---
    # ``balance_check`` is the master switch behind the sidebar tick: when False
    # nothing below runs, no silo is ever added, and results are identical to a
    # run without the feature.
    balance_check: bool = True
    balance_mass_normalized: bool = True  # compare k/m (Caltrans form / SGS variable width)
    # Hold the two piers of a SIMPLY SUPPORTED span to the balanced-stiffness
    # rule as well.  Off by default: a run of simple spans is modelled one
    # frame per pier, so there is no pair inside a frame and they are matched on
    # period alone.  Whether the span itself counts as a frame whose supports
    # must balance is a modelling choice, so it is yours to make.
    balance_simple_span_stiffness: bool = False
    balance_k_ratio_min: float = 0.75     # min(ki,kj)/max(ki,kj), adjacent bents
    balance_k_ratio_any: float = 0.50     # ... any two bents in a frame
    balance_T_ratio_min: float = 0.70     # min(Ti,Tj)/max(Ti,Tj) for adjacent piers
    balance_auto_silo: bool = True        # let the tool size column silos to comply
    # How the silo depths are chosen:
    #   "min_silo" — exact minimum TOTAL silo on the buildable grid (a dynamic
    #                program over each frame), re-calibrated and verified each
    #                pass.  Costs a little more analysis, gives the cheapest
    #                buildable answer.
    #   "greedy"   — repair one failing pair at a time, monotonically deepening.
    #                Faster, but pays for the cascade each local fix sets off.
    balance_strategy: str = "min_silo"
    max_silo_ft: float = 20.0             # cap on any silo depth
    silo_step_ft: float = 1.0             # silo depths quantised (rounded up) to this
    balance_max_outer: int = 6            # outer silo <-> seismic re-check iterations
    # --- soil-structure interaction (point of fixity) ---
    fixity_source: str = "multiplier"    # "multiplier" (3x/6x) | "soil" (p-y)
    water_table_ft: float = 100.0        # depth to groundwater below top of shaft
    shaft_embed_ft: float = 60.0         # embedded shaft length (ft)
    soil_stiff_factor: float = 2.0       # upper-bound soil-stiffness bracket
    soil_soft_factor: float = 0.5        # lower-bound soil-stiffness bracket
    soil_layers: tuple[dict, ...] = field(default_factory=tuple)  # strata (eng. units)


# Strata table columns (engineering units, matching LPile input).
SOIL_COLUMNS: tuple[str, ...] = (
    "layer", "py_model", "thickness_ft", "gamma_pcf",
    "su_top_ksf", "su_bot_ksf", "eps50", "phi_deg", "k_pci",
)

SOIL_COLUMN_META: dict[str, tuple[str, str]] = {
    "layer": ("Layer", "Layer name / number (top to bottom)."),
    "py_model": ("p-y model", "matlock_soft_clay | welch_stiff_clay | api_sand | "
                 "elastic_subgrade"),
    "thickness_ft": ("Thickness (ft)", "Layer thickness."),
    "gamma_pcf": ("Unit wt γ (pcf)", "Total unit weight; buoyant below the water "
                  "table is applied automatically."),
    "su_top_ksf": ("su top (ksf)", "Undrained shear strength at layer top (clay)."),
    "su_bot_ksf": ("su bot (ksf)", "Undrained shear strength at layer bottom (clay)."),
    "eps50": ("ε50", "Strain at 50% strength (clay)."),
    "phi_deg": ("φ′ (deg)", "Effective friction angle (sand)."),
    "k_pci": ("k (pci)", "Initial subgrade modulus (sand / stiff clay)."),
}


def default_soil_layers() -> list[dict]:
    """A starter strata table (LPile-style, engineering units)."""
    return [
        {"layer": "1 clay", "py_model": "matlock_soft_clay", "thickness_ft": 20.0,
         "gamma_pcf": 120.0, "su_top_ksf": 1.0, "su_bot_ksf": 1.5, "eps50": 0.01,
         "phi_deg": 0.0, "k_pci": 0.0},
        {"layer": "2 sand", "py_model": "api_sand", "thickness_ft": 40.0,
         "gamma_pcf": 125.0, "su_top_ksf": 0.0, "su_bot_ksf": 0.0, "eps50": 0.0,
         "phi_deg": 36.0, "k_pci": 90.0},
    ]


def _row(layer, py_model, thickness_ft, gamma_pcf, *, su=0.0, eps50=0.0,
         phi=0.0, k=0.0) -> dict:
    """Build one strata-table row (all SOIL_COLUMNS keys present)."""
    return {"layer": layer, "py_model": py_model, "thickness_ft": thickness_ft,
            "gamma_pcf": gamma_pcf, "su_top_ksf": su, "su_bot_ksf": su,
            "eps50": eps50, "phi_deg": phi, "k_pci": k}


# Named LPile-style strata presets (project profiles).  Submerged layers are
# entered as TOTAL unit weight = geotech's effective (buoyant) weight + 62.4 pcf,
# because the app re-applies buoyancy from ``water_table_ft``.  "Ignore" layers
# (no lateral resistance in LPile) are modelled as ``elastic_subgrade`` with
# k = 0 — zero p-y reaction, but their weight still counts toward overburden.
# ``k`` for the medium sand and ``eps50`` for the stiff clay replace LPile
# "program default" cells with standard values — review against your report.
SOIL_PROFILE_PRESETS: dict[str, dict] = {
    "SeaTac Piers A8–A11 (GWT 10 ft)": {
        "water_table_ft": 10.0,
        "layers": [
            _row("1 ignore (0–5 ft)", "elastic_subgrade", 5.0, 120.0),
            _row("2 sand (Reese)", "api_sand", 5.0, 120.0, phi=32.0, k=50.0),
            _row("3 sand liquefied", "api_sand", 5.0, 129.4, phi=21.0, k=35.0),
            _row("4 stiff clay (hard)", "welch_stiff_clay", 115.0, 135.0,
                 su=10.0, eps50=0.004),
        ],
    },
    "SeaTac Piers B2–B18 (GWT 5 ft)": {
        "water_table_ft": 5.0,
        "layers": [
            _row("1 ignore (0–5 ft)", "elastic_subgrade", 5.0, 120.0),
            _row("2 sand liquefied", "api_sand", 5.0, 129.4, phi=18.0, k=13.0),
            _row("3 sand liquefied", "api_sand", 10.0, 129.4, phi=18.0, k=13.0),
            _row("4 stiff clay", "welch_stiff_clay", 10.0, 135.0,
                 su=6.0, eps50=0.004),
            _row("5 stiff clay", "welch_stiff_clay", 90.0, 135.0,
                 su=8.0, eps50=0.004),
        ],
    },
}


def soil_preset_names() -> list[str]:
    """Names of the available strata presets, for a UI dropdown."""
    return list(SOIL_PROFILE_PRESETS)


def load_soil_preset(name: str) -> tuple[float, list[dict]]:
    """Return ``(water_table_ft, layers)`` for preset ``name`` (fresh copies)."""
    p = SOIL_PROFILE_PRESETS[name]
    return float(p["water_table_ft"]), [dict(r) for r in p["layers"]]


def build_soil_profile(cfg: "GlobalConfig"):
    """Build a :class:`~seismic_column.soil.SoilProfile` from the config, or None.

    Layers are marked submerged (buoyant unit weight) when their top is at or
    below ``water_table_ft``.  Returns ``None`` when no strata are defined.
    """
    from .soil import SoilLayer, SoilProfile
    if not cfg.soil_layers:
        return None
    layers, top_ft = [], 0.0
    for d in cfg.soil_layers:
        th = float(d.get("thickness_ft", 0.0))
        if th <= 0.0:
            continue
        submerged = top_ft >= float(cfg.water_table_ft)
        layers.append(SoilLayer.from_engineering(
            th, str(d.get("py_model", "matlock_soft_clay")),
            float(d.get("gamma_pcf", 120.0)),
            su_top_ksf=float(d.get("su_top_ksf", 0.0)),
            su_bot_ksf=float(d.get("su_bot_ksf", 0.0)) or None,
            eps50=float(d.get("eps50", 0.01)) or 0.01,
            phi_deg=float(d.get("phi_deg", 0.0)),
            k_pci=float(d.get("k_pci", 0.0)),
            submerged=submerged,
        ))
        top_ft += th
    return SoilProfile(tuple(layers)) if layers else None


def pile_profile_table(sol) -> pd.DataFrame:
    """Deflection / shear / moment along the pile, for diagrams and CSV export."""
    import numpy as np
    x = sol.x
    zg = x[sol.ground_index]
    return pd.DataFrame({
        "dist_from_top_ft": np.round(x / 12.0, 3),
        "depth_below_ground_ft": np.round((x - zg) / 12.0, 3),
        "deflection_in": np.round(sol.y, 5),
        "shear_kip": np.round(sol.shear, 2),
        "moment_kipft": np.round(sol.moment / 12.0, 1),
    })


def py_curves_table(profile, D: float, depths, y_max: float | None = None,
                    n: int = 41) -> pd.DataFrame:
    """Long-format p-y curves at ``depths`` (in) — spring definitions for a
    global structural model. Columns: depth_ft, y_in, p_kip_per_in."""
    rows = []
    for z in depths:
        ys, ps = profile.py_curve(z, D, y_max, n)
        for y, p in zip(ys, ps):
            rows.append({"depth_ft": round(z / 12.0, 2),
                         "y_in": round(float(y), 5),
                         "p_kip_per_in": round(float(p), 5)})
    return pd.DataFrame(rows)


def default_row(name: str = "C1") -> dict:
    """A sensible starting row."""
    return {
        "name": name,
        "frame": "F1",
        "deck_link": "pinned",
        "Hcol_ft": 22.0,
        "silo_ft": 0.0,
        # 1 = a single-column bent, which is what every table meant before these
        # two columns existed.  Spacing is only read when n_columns > 1.
        "n_columns": 1,
        "col_spacing_ft": 0.0,
        "cap_fixity": "fixed",
        "D_shaft_in": 84.0,
        "weight_long_kip": 800.0,
        "weight_trans_kip": 800.0,
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
        # simply supported spans are the default: one frame per bent, so no
        # balanced-stiffness rule applies between them
        r["frame"] = f"F{i + 1}"
        r["Hcol_ft"] = 18.0 + 4.0 * i
        r["weight_long_kip"] = 700.0 + 100.0 * i
        r["weight_trans_kip"] = 700.0 + 100.0 * i
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


# Reinforcement / f'c the optimiser DETERMINES; these may be left blank on an
# optimise run ("you decide") — validate fills a minimum placeholder the
# optimiser overwrites.  Geometry, loads, cover and fixity multipliers are not
# optimiser-chosen, so a blank there is still an error.
OPTIMIZED_COLUMNS: frozenset[str] = frozenset({
    "fc_ksi", "n_bars", "long_bar_no", "long_bundle",
    "spiral_bar_no", "spiral_spacing_in", "spiral_bundle",
    "shaft_fc_ksi", "shaft_n_bars", "shaft_long_bar_no", "shaft_long_bundle",
    "shaft_spiral_bar_no", "shaft_spiral_spacing_in", "shaft_spiral_bundle",
})


def validate(df: pd.DataFrame, min_shaft_oversize: float = 0.0,
             optimize: bool = False, max_silo_ft: float | None = None) -> pd.DataFrame:
    """Validate and normalise a batch table, filling defaults for missing cols.

    ``min_shaft_oversize`` is the required ``D_shaft - Dcol`` in inches: 0 for
    AASHTO SGS ("larger in diameter", Owner's discretion) and 24 for Caltrans
    SDC, whose Type II definition demands at least 24 in.

    ``optimize``: when True, blank reinforcement / f'c cells (the values the
    optimiser determines, :data:`OPTIMIZED_COLUMNS`) are filled with a minimum
    placeholder instead of erroring — so an optimise run can leave the rebar
    blank.  Other blanks (loads, height, diameters, cover) still error.

    ``max_silo_ft``: when given, ``silo_ft`` entries deeper than this error.
    The batch runner does *not* pass it — the auto-silo cap limits what the tool
    adds, not what an engineer deliberately types in — but an importer or a GUI
    that wants the stricter reading can.
    """
    df = df.copy()
    # --- migrate an older table BEFORE anything else looks at it -------------
    # weight_kip was the single tributary weight; it is the LONGITUDINAL one.
    if "weight_kip" in df.columns and "weight_long_kip" not in df.columns:
        df = df.rename(columns={"weight_kip": "weight_long_kip"})
    # A table with no deck_link column predates the frame model.  Its `frame`
    # column (if any) was filled with a single id for every row, which under the
    # new rules would read as ONE giant continuous frame and start applying the
    # any-two-bents rule.  Simply supported spans are the safe reading, so give
    # every bent its own frame -- and record it, because silently regrouping
    # someone's structure would be indefensible.
    legacy_frames = "deck_link" not in df.columns
    df.attrs.setdefault("migrations", [])
    if legacy_frames:
        df.attrs["migrations"] = df.attrs["migrations"] + [
            "Table predates the deck_link column, so every bent was given its "
            "own frame (simply supported spans). Set `frame` and `deck_link` "
            "to declare a continuous frame."]

    missing_required = {"Hcol_ft", "D_shaft_in", "axial_kip", "Dcol_in"}
    absent = missing_required - set(df.columns)
    if absent:
        raise ValueError(f"Batch table missing required columns: {sorted(absent)}")
    if "weight_long_kip" not in df.columns:
        raise ValueError("Batch table missing required column: 'weight_long_kip' "
                         "(older tables may use 'weight_kip')")

    defaults = default_row()
    for col in COLUMNS:
        if col not in df.columns:
            # weight_trans_kip must fall back to the LONGITUDINAL weight of the
            # same row, not to a starter constant, so leave it blank for the
            # fillna below to resolve.
            df[col] = float("nan") if col == "weight_trans_kip" else defaults[col]
    df = df[list(COLUMNS)]

    # ``frame`` groups piers into a frame for the balance checks.  A blank CELL
    # is meaningful and preserved: that pier opts out (see ``in_frame``).
    df["frame"] = df["frame"].fillna("").astype(str).str.strip()
    if legacy_frames:
        df["frame"] = [f"F{i + 1}" for i in range(len(df))]
    # deck_link: how the deck attaches, which derives directional participation
    # `pinned` (bearings, no moment) is the safe default: it is the simply
    # supported case and keeps the fixed-free stiffness a legacy table had.
    df["deck_link"] = (df["deck_link"].fillna("").astype(str).str.strip()
                       .str.lower().replace("", "pinned"))
    bad_link = sorted({tok for v in df["deck_link"]
                       for tok in deck_links(v)} - set(DECK_LINKS))
    if bad_link:
        raise ValueError(f"Unknown deck_link {bad_link}; choose from "
                         f"{list(DECK_LINKS)}")
    # one link per frame, or a single link broadcast to all of them
    for nm, fr, lk in zip(df["name"], df["frame"], df["deck_link"]):
        nf, nl = len(frame_keys(fr)), len(deck_links(lk))
        if nf and nl != 1 and nl != nf:
            raise ValueError(
                f"{nm}: deck_link lists {nl} link(s) but frame lists {nf} "
                f"frame(s) — give one link per frame, in the same order, or a "
                f"single link for all of them")
    # A silo is optional everywhere; blank = none.
    df["silo_ft"] = pd.to_numeric(df["silo_ft"], errors="coerce").fillna(0.0)
    # A bent is single-column unless told otherwise, which is exactly what every
    # table before this column meant.
    df["n_columns"] = (pd.to_numeric(df["n_columns"], errors="coerce")
                       .fillna(1.0).round().astype("Int64"))
    df["col_spacing_ft"] = pd.to_numeric(
        df["col_spacing_ft"], errors="coerce").fillna(0.0)
    # A monolithic cap is the usual multi-column detail, so it is the default;
    # it is ignored entirely on a single-column bent.
    df["cap_fixity"] = (df["cap_fixity"].fillna("").astype(str).str.strip()
                        .str.lower().replace("", "fixed"))
    bad_cap = sorted(set(df["cap_fixity"]) - set(CAP_FIXITIES))
    if bad_cap:
        raise ValueError(f"Unknown cap_fixity {bad_cap}; choose from "
                         f"{list(CAP_FIXITIES)}")
    # Transverse weight defaults to the longitudinal one, so a table that never
    # distinguished them behaves exactly as before.
    df["weight_trans_kip"] = pd.to_numeric(
        df["weight_trans_kip"], errors="coerce").fillna(
            pd.to_numeric(df["weight_long_kip"], errors="coerce"))

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if optimize:                                   # blank rebar/f'c = "you decide"
        for col in OPTIMIZED_COLUMNS:
            df[col] = df[col].fillna(defaults[col])
    int_cols = ("n_bars", "long_bar_no", "long_bundle", "spiral_bar_no",
                "spiral_bundle", "shaft_n_bars", "shaft_long_bar_no",
                "shaft_long_bundle", "shaft_spiral_bar_no", "shaft_spiral_bundle")
    for col in int_cols:
        df[col] = df[col].round().astype("Int64")
    # Force the non-integer design columns to float.  ``pd.to_numeric`` infers
    # int64 for an all-whole-number column (e.g. spacings entered as 4, 4, …);
    # writing a fractional optimiser result back (a 3.5 in spiral pitch, a 4.5 ksi
    # f'c) into that int64 column then raises "Invalid value '3.5' for dtype
    # 'int64'".  Pinning them float keeps the write-back safe.
    for col in NUMERIC_COLUMNS:
        if col not in int_cols:
            df[col] = df[col].astype("float64")

    if df[list(NUMERIC_COLUMNS)].isna().any().any():
        bad = df[df[list(NUMERIC_COLUMNS)].isna().any(axis=1)].index.tolist()
        raise ValueError(f"Non-numeric or missing values in rows: {bad}")

    if (df["n_columns"] < 1).any():
        rows = ", ".join(str(n) for n in df.loc[df["n_columns"] < 1, "name"])
        raise ValueError(f"n_columns must be at least 1: {rows}")
    # Without a spacing there is no lever for the transverse push/pull couple,
    # so a multi-column bent would silently behave as n stacked single columns.
    _multi_no_gap = (df["n_columns"] > 1) & (df["col_spacing_ft"] <= 0)
    if _multi_no_gap.any():
        rows = ", ".join(str(n) for n in df.loc[_multi_no_gap, "name"])
        raise ValueError(
            f"col_spacing_ft must be > 0 for a multi-column bent: {rows}. The "
            f"spacing is the lever arm of the transverse push/pull couple; "
            f"without it the overturning axial cannot be distributed.")

    if (df["silo_ft"] < 0).any():
        rows = ", ".join(str(n) for n in df.loc[df["silo_ft"] < 0, "name"])
        raise ValueError(f"Column silo depth cannot be negative: {rows}")
    if max_silo_ft is not None and (df["silo_ft"] > max_silo_ft).any():
        over = df[df["silo_ft"] > max_silo_ft]
        rows = ", ".join(f"{r['name']} ({r['silo_ft']:g} ft)"
                         for _, r in over.iterrows())
        raise ValueError(
            f"Column silo depth exceeds the {max_silo_ft:g} ft maximum: {rows}")

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
    d["soil_layers"] = [dict(l) for l in d.get("soil_layers", ())]
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
    d["soil_layers"] = tuple(dict(l) for l in d.get("soil_layers", ()))
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
