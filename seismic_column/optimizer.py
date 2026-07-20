"""Greedy, priority-ordered optimiser for the column/shaft design.

The user chooses which design parameters are *fixed* and which are *variable*,
and a priority order.  Variable parameters are increased along discrete "ladders"
in priority order (cheapest first) until every SDC check passes:

    default priority : longitudinal -> confinement -> diameter -> f'c

After a feasible column is found, the Type II shaft reinforcement is sized (its
own longitudinal and transverse steel) to satisfy the capacity-protection
checks.  The result is the first feasible design that favours changes to the
higher-priority (cheaper) parameters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .demand import DesignSpectrum
from .geometry import Geometry
from .materials import bar_area, bar_diameter
from .moment_curvature import moment_curvature
from .provisions import SDC_2_0, CodeProvisions
from .section import CircularSection
from .sdc_capacity import (
    UNIT_WEIGHT_DEFAULT,
    ColumnAssessment,
    column_self_weight,
    evaluate_column,
    shear_capacity,
)

# ---------------------------------------------------------------------------
# Design representation
# ---------------------------------------------------------------------------
PARAMETERS = ("longitudinal", "confinement", "diameter", "fc")
DEFAULT_PRIORITY = ("longitudinal", "confinement", "diameter", "fc")


@dataclass
class ColumnDesign:
    """Mutable design variables for the column (or shaft)."""

    D: float
    fc: float
    cover: float
    n_bars: int
    long_bar_no: int
    spiral_bar_no: int
    spiral_spacing: float
    long_bundle: int = 1
    spiral_bundle: int = 1
    fye: float = 68.0
    fue: float = 95.0
    fyh: float = 68.0
    hoops: bool = False

    def section(self) -> CircularSection:
        return CircularSection(
            D=self.D, fc=self.fc, cover=self.cover, n_bars=self.n_bars,
            long_bar_no=self.long_bar_no, spiral_bar_no=self.spiral_bar_no,
            spiral_spacing=self.spiral_spacing, fye=self.fye, fue=self.fue,
            fyh=self.fyh, hoops=self.hoops,
            long_bundle=self.long_bundle, spiral_bundle=self.spiral_bundle,
        )

    def rho_l(self) -> float:
        return self.section().rho_l

    def long_label(self) -> str:
        b = f"{self.n_bars}-#{self.long_bar_no}"
        return b + (f" (bundled x{self.long_bundle})" if self.long_bundle > 1 else "")

    def spiral_label(self) -> str:
        b = f"#{self.spiral_bar_no}@{self.spiral_spacing:g}"
        return b + (f" (bundled x{self.spiral_bundle})" if self.spiral_bundle > 1 else "")


@dataclass
class OptimizeSpec:
    """Which parameters may vary, the priority order and the search ladders."""

    variable: set[str] = field(default_factory=lambda: set(PARAMETERS))
    priority: tuple[str, ...] = DEFAULT_PRIORITY
    # longitudinal ladder
    bar_numbers: tuple[int, ...] = (8, 9, 10, 11, 14)
    min_bar_spacing: float = 6.0        # min c/c longitudinal spacing along cage, in
    allow_bundling: bool = False        # allow 2-bar longitudinal bundles
    n_bars_min: int = 6
    # confinement ladder (single bars capped at #8; bundled #4 per Caltrans)
    spiral_bar_numbers: tuple[int, ...] = (4, 5, 6, 7, 8)
    spiral_spacings: tuple[float, ...] = (6.0, 5.0, 4.0, 3.5, 3.0)
    bundled_spiral_bar: int = 4
    bundled_spiral_spacings: tuple[float, ...] = (6.0, 5.0, 4.0)
    # section-level ladders
    diameters: tuple[float, ...] = (36, 42, 48, 54, 60, 66, 72, 78, 84)
    fc_values: tuple[float, ...] = (4.0, 5.0, 6.0)
    rho_l_min: float = 0.01
    rho_l_max: float = 0.04


@dataclass
class OptimizeResult:
    design: ColumnDesign
    shaft: ColumnDesign
    assessment: ColumnAssessment
    feasible: bool
    iterations: int
    log: list[str]


@dataclass
class _Ctx:
    """Bundle of fixed context passed around the optimiser."""

    geometry: Geometry
    spectrum: object
    axial: float
    weight: float
    mults: tuple[float, ...]
    shaft_moment_basis: str
    lle_spectrum: object
    lle_mu_limit: float
    spec: OptimizeSpec
    concrete_unit_weight: float = UNIT_WEIGHT_DEFAULT
    self_weight_mass_factor: float = 1.0 / 3.0
    self_weight_in_axial: bool = True
    provisions: CodeProvisions = SDC_2_0

    def effective_axial(self, D: float) -> float:
        """Axial incl. column self-weight above the hinge for a given diameter."""
        if not self.self_weight_in_axial:
            return self.axial
        Ag = math.pi * D ** 2 / 4.0
        W = column_self_weight(Ag, self.geometry.Hcol, self.concrete_unit_weight)
        return self.axial + W


# ---------------------------------------------------------------------------
# Ladder generators (each returns candidate values in ascending order)
# ---------------------------------------------------------------------------
def max_bar_positions(design: ColumnDesign, bar_no: int, spec: OptimizeSpec) -> int:
    """Maximum number of bar positions around the cage from the min spacing.

    Effective cage perimeter = 2*pi*r_bars (bar-centreline circle); the number
    of positions is limited to perimeter / min c/c spacing.
    """
    dsp = bar_diameter(design.spiral_bar_no)
    dbl = bar_diameter(bar_no)
    r_bars = design.D / 2.0 - design.cover - dsp - dbl / 2.0
    perimeter = 2.0 * math.pi * max(r_bars, 1.0)
    return max(spec.n_bars_min, int(perimeter // spec.min_bar_spacing))


def _longitudinal_ladder(design: ColumnDesign, spec: OptimizeSpec) -> list[tuple[int, int, int]]:
    """Ascending (n_bars, bar_no, bundle) combos within rho and spacing limits."""
    Ag = math.pi * design.D ** 2 / 4.0
    bundles = (1, 2) if spec.allow_bundling else (1,)
    combos = []
    for bar_no in spec.bar_numbers:
        max_pos = max_bar_positions(design, bar_no, spec)
        for bundle in bundles:
            for n in range(spec.n_bars_min, max_pos + 1):
                ast = n * bundle * bar_area(bar_no)
                rho = ast / Ag
                if spec.rho_l_min <= rho <= spec.rho_l_max:
                    combos.append((ast, n, bar_no, bundle))
    combos.sort()
    return [(n, b, bd) for _, n, b, bd in combos]


def _confinement_ladder(spec: OptimizeSpec) -> list[tuple[int, float, int]]:
    """Ascending (spiral_bar_no, spacing, bundle) combos by transverse amount."""
    combos = []
    for bar_no in spec.spiral_bar_numbers:            # single bars, max #8
        for s in spec.spiral_spacings:
            combos.append((bar_area(bar_no) / s, bar_no, s, 1))
    for s in spec.bundled_spiral_spacings:            # bundled #4 (Caltrans max)
        combos.append((2.0 * bar_area(spec.bundled_spiral_bar) / s,
                       spec.bundled_spiral_bar, s, 2))
    combos.sort()
    return [(b, s, bd) for _, b, s, bd in combos]


def _ascending_from(values: tuple[float, ...], current: float) -> list[float]:
    return [v for v in sorted(values) if v >= current]


# ---------------------------------------------------------------------------
# Cached plastic moment (for fast shaft sizing across many column candidates)
# ---------------------------------------------------------------------------
_MP_CACHE: dict[tuple, float] = {}


def _plastic_moment(design: ColumnDesign, axial: float) -> float:
    key = (design.D, design.fc, design.cover, design.n_bars, design.long_bar_no,
           design.long_bundle, design.spiral_bar_no, design.spiral_spacing,
           design.spiral_bundle, design.fye, design.fue, design.fyh, round(axial, 3))
    if key not in _MP_CACHE:
        _MP_CACHE[key] = moment_curvature(design.section(), axial).Mp
    return _MP_CACHE[key]


# ---------------------------------------------------------------------------
# Shaft capacity-protection sizing
# ---------------------------------------------------------------------------
def size_shaft(column: ColumnDesign, shaft_start: ColumnDesign, ctx: _Ctx) -> ColumnDesign:
    """Size shaft longitudinal + transverse steel for capacity protection.

    Flexure: shaft Mn >= column overstrength moment demand (per basis).
    Shear:   phi*Vn_shaft >= column overstrength shear Vo = Mo/Hcol.
    """
    spec = ctx.spec
    P_col = ctx.effective_axial(column.D)
    Mo = ctx.provisions.overstrength_factor * _plastic_moment(column, P_col)
    Vo = Mo / ctx.geometry.Hcol
    if ctx.shaft_moment_basis == "fixity":
        Df = max(ctx.geometry.fixity_depth(m) for m in ctx.mults)
        m_demand = Mo * (ctx.geometry.Hcol + Df) / ctx.geometry.Hcol
    else:
        m_demand = Mo

    shaft = replace(shaft_start)

    # --- longitudinal: smallest that develops the required flexural capacity ---
    long_ladder = _longitudinal_ladder(shaft, spec)
    chosen = None
    for n, b, bundle in long_ladder:
        cand = replace(shaft, n_bars=n, long_bar_no=b, long_bundle=bundle)
        if _plastic_moment(cand, P_col) >= m_demand:
            chosen = cand
            break
    if chosen is None and long_ladder:                # exhausted -> use the heaviest
        n, b, bundle = long_ladder[-1]
        chosen = replace(shaft, n_bars=n, long_bar_no=b, long_bundle=bundle)
    if chosen is not None:
        shaft = chosen

    # --- transverse: smallest that develops the required shear capacity ---
    for b, s, bundle in _confinement_ladder(spec):
        cand = replace(shaft, spiral_bar_no=b, spiral_spacing=s, spiral_bundle=bundle)
        phiVn, _, _ = shear_capacity(cand.section(), P_col, mu_d=1.0,
                                     inside_hinge=False,
                                     vs_max_coeff=ctx.provisions.vs_max_coeff)
        if phiVn >= Vo:
            shaft = cand
            break
    else:
        b, s, bundle = _confinement_ladder(spec)[-1]
        shaft = replace(shaft, spiral_bar_no=b, spiral_spacing=s, spiral_bundle=bundle)

    return shaft


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------
def optimize_column(
    start: ColumnDesign,
    shaft_start: ColumnDesign,
    geometry: Geometry,
    spectrum: DesignSpectrum,
    axial: float,
    weight: float,
    spec: OptimizeSpec | None = None,
    fixity_multipliers: tuple[float, ...] = (3.0, 6.0),
    shaft_moment_basis: str = "interface",
    lle_spectrum=None,
    lle_mu_limit: float = 1.0,
    concrete_unit_weight: float = UNIT_WEIGHT_DEFAULT,
    self_weight_mass_factor: float = 1.0 / 3.0,
    self_weight_in_axial: bool = True,
    provisions: CodeProvisions = SDC_2_0,
    max_iterations: int = 400,
) -> OptimizeResult:
    """Greedy priority-ordered search for a feasible column + shaft design."""
    spec = spec or OptimizeSpec()
    ctx = _Ctx(geometry, spectrum, axial, weight, fixity_multipliers,
               shaft_moment_basis, lle_spectrum, lle_mu_limit, spec,
               concrete_unit_weight, self_weight_mass_factor, self_weight_in_axial,
               provisions)
    design = replace(start)
    log: list[str] = []
    state = {"iters": 0, "shaft": replace(shaft_start)}

    def assess(d: ColumnDesign) -> ColumnAssessment:
        shaft = size_shaft(d, shaft_start, ctx)
        state["shaft"] = shaft
        state["iters"] += 1
        return evaluate_column(
            d.section(), shaft.section(), geometry, spectrum, axial, weight,
            fixity_multipliers=fixity_multipliers,
            rho_l_min=spec.rho_l_min, rho_l_max=spec.rho_l_max,
            shaft_moment_basis=shaft_moment_basis,
            lle_spectrum=lle_spectrum, lle_mu_limit=lle_mu_limit,
            concrete_unit_weight=concrete_unit_weight,
            self_weight_mass_factor=self_weight_mass_factor,
            self_weight_in_axial=self_weight_in_axial,
            provisions=provisions,
        )

    assessment = assess(design)
    if assessment.passed:
        log.append("Starting design already satisfies all checks.")
        return OptimizeResult(design, state["shaft"], assessment, True,
                              state["iters"], log)

    for param in spec.priority:
        if param not in spec.variable:
            continue
        if state["iters"] >= max_iterations:
            break
        design, assessment = _sweep_parameter(param, design, spec, assess, log,
                                               state, max_iterations)
        if assessment.passed:
            log.append(f"Feasible after adjusting '{param}'.")
            return OptimizeResult(design, state["shaft"], assessment, True,
                                  state["iters"], log)

    feasible = assessment.passed
    log.append("Exhausted variable parameters." if not feasible else "Feasible.")
    return OptimizeResult(design, state["shaft"], assessment, feasible,
                          state["iters"], log)


def _sweep_parameter(param, design, spec, assess, log, state, max_iterations):
    """Step one parameter along its ladder until the design passes or exhausts."""
    if param == "longitudinal":
        for n, b, bundle in _longitudinal_ladder(design, spec):
            if state["iters"] >= max_iterations:
                break
            cand = replace(design, n_bars=n, long_bar_no=b, long_bundle=bundle)
            a = assess(cand)
            design = cand
            if a.passed:
                log.append(f"longitudinal -> {cand.long_label()} (rho_l={cand.rho_l():.4f})")
                return design, a
        log.append(f"longitudinal maxed at {design.long_label()}")
        return design, assess(design)

    if param == "confinement":
        for b, s, bundle in _confinement_ladder(spec):
            if state["iters"] >= max_iterations:
                break
            cand = replace(design, spiral_bar_no=b, spiral_spacing=s, spiral_bundle=bundle)
            a = assess(cand)
            design = cand
            if a.passed:
                log.append(f"confinement -> {cand.spiral_label()}")
                return design, a
        log.append("confinement maxed")
        return design, assess(design)

    if param == "diameter":
        for D in _ascending_from(spec.diameters, design.D):
            if state["iters"] >= max_iterations:
                break
            cand = replace(design, D=D)
            a = assess(cand)
            design = cand
            if a.passed:
                log.append(f"diameter -> {D} in")
                return design, a
        log.append("diameter maxed")
        return design, assess(design)

    if param == "fc":
        for fc in _ascending_from(spec.fc_values, design.fc):
            if state["iters"] >= max_iterations:
                break
            cand = replace(design, fc=fc)
            a = assess(cand)
            design = cand
            if a.passed:
                log.append(f"f'c -> {fc} ksi")
                return design, a
        log.append("f'c maxed")
        return design, assess(design)

    return design, assess(design)
