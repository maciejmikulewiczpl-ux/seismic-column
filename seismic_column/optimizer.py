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
    inground_demand,
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
    fce_factor: float = 1.3     # f'ce = factor * f'c for the section response
    fce_floor: float | None = None   # Caltrans floors f'ce at 5.0 ksi

    def section(self) -> CircularSection:
        return CircularSection(
            D=self.D, fc=self.fc, fce_factor=self.fce_factor,
            fce_floor=self.fce_floor,
            cover=self.cover, n_bars=self.n_bars,
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
    # oversized (Type II) shaft: grow the shaft so it stays this many inches
    # larger than the column; the entered shaft is the floor.  Standard shaft
    # sizes the growth snaps up to.
    min_shaft_oversize: float = 24.0
    shaft_diameters: tuple[float, ...] = tuple(range(36, 181, 6))
    # objective for the diameter search (starts at the min diameter and grows):
    #   "min_diameter" -> smallest column that works (steel as needed)
    #   "target_steel" -> smallest column that works with ~target_rho longitudinal
    #   "min_steel"    -> smallest column that works at the ~rho_l_min floor
    objective: str = "min_diameter"
    target_rho: float = 0.02


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
    fixity_source: str = "multiplier"
    soil_profile: object = None
    shaft_embed_length: float | None = None
    soil_bounds: tuple[float, float] = (2.0, 0.5)

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


def _min_steel_design(design: ColumnDesign, spec: OptimizeSpec) -> ColumnDesign:
    """The lightest longitudinal cage (smallest ladder rung >= rho_l_min)."""
    ladder = _longitudinal_ladder(design, spec)
    if not ladder:
        return design
    n, b, bundle = ladder[0]
    return replace(design, n_bars=n, long_bar_no=b, long_bundle=bundle)


def _cage_at_ratio(design: ColumnDesign, spec: OptimizeSpec,
                   target: float) -> ColumnDesign:
    """The lightest longitudinal cage whose ratio is >= ``target`` (heaviest if
    the target exceeds the ladder).  Used by the "target steel %" objective:
    hold the longitudinal ratio at ~target and search the diameter."""
    ladder = _longitudinal_ladder(design, spec)
    if not ladder:
        return design
    Ag = math.pi * design.D ** 2 / 4.0
    for n, b, bundle in ladder:                       # ascending by steel area
        rho = n * bundle * bar_area(b) / Ag
        if rho >= target - 1e-9:
            return replace(design, n_bars=n, long_bar_no=b, long_bundle=bundle)
    n, b, bundle = ladder[-1]
    return replace(design, n_bars=n, long_bar_no=b, long_bundle=bundle)


def required_shaft_diameter(col_D: float, oversize: float,
                            shaft_sizes: tuple[float, ...]) -> float:
    """Shaft diameter derived from the column: the smallest standard shaft size
    that is at least ``oversize`` inches larger than the column.

    The shaft ALWAYS tracks the column by the oversize — it is not floored by any
    entered shaft size, so shrinking the column (e.g. the "smallest column"
    objective) shrinks the shaft too.  The entered shaft diameter is therefore a
    starting value only; the optimiser overwrites it.
    """
    required = col_D + oversize
    for s in sorted(shaft_sizes):
        if s >= required:
            return s
    return required


# ---------------------------------------------------------------------------
# Cached plastic moment (for fast shaft sizing across many column candidates)
# ---------------------------------------------------------------------------
_MC_CACHE: dict[tuple, object] = {}


def _mc_cached(design: ColumnDesign, axial: float):
    key = (design.D, design.fc, design.cover, design.n_bars, design.long_bar_no,
           design.long_bundle, design.spiral_bar_no, design.spiral_spacing,
           design.spiral_bundle, design.fye, design.fue, design.fyh,
           design.fce_factor, design.fce_floor, round(axial, 3))
    if key not in _MC_CACHE:
        _MC_CACHE[key] = moment_curvature(design.section(), axial)
    return _MC_CACHE[key]


def _plastic_moment(design: ColumnDesign, axial: float) -> float:
    return _mc_cached(design, axial).Mp


def _eff_EI(design: ColumnDesign, axial: float) -> float:
    """Cracked effective EI (Mp/φy) — the shaft stiffness fed to the p-y solve."""
    return _mc_cached(design, axial).EI_eff


# ---------------------------------------------------------------------------
# Shaft capacity-protection sizing
# ---------------------------------------------------------------------------
def _bsearch_ladder(shaft: ColumnDesign, ladder, P_col: float,
                    m_demand: float) -> ColumnDesign | None:
    """Smallest rung on an Ast-ascending ladder whose Mp >= ``m_demand``.

    Mp is monotonic in steel area, so a binary search finds the boundary in
    ~log2(n) moment-curvature evaluations instead of scanning every rung — the
    dominant cost when sizing a large shaft.  Always returns a *passing* rung
    (or None), so at worst it is one rung heavier than the true minimum.
    """
    lo, hi, best = 0, len(ladder), None
    while lo < hi:
        mid = (lo + hi) // 2
        n, b, bundle = ladder[mid]
        cand = replace(shaft, n_bars=n, long_bar_no=b, long_bundle=bundle)
        if _plastic_moment(cand, P_col) >= m_demand:
            best, hi = cand, mid
        else:
            lo = mid + 1
    return best


def _size_shaft_longitudinal(shaft: ColumnDesign, m_demand: float,
                             P_col: float, spec: OptimizeSpec) -> ColumnDesign:
    """Smallest longitudinal cage whose Mn >= ``m_demand`` (heaviest if none)."""
    # The Type II shaft is capacity-protected (not ductility-limited).  Escalate
    # only as needed: conventional single-layer bars first, then add #18, then
    # allow bundling — up to the 0.04 "compression member" max (SGS 8.8.1 /
    # SDC 5.3.9.1).  This keeps a conventional cage unless heavy steel is forced.
    augmented = tuple(sorted(set(spec.bar_numbers) | {14, 18}))
    tiers = ((spec.bar_numbers, False), (augmented, False), (augmented, True))
    for bars, bundle_ok in tiers:
        ladder = _longitudinal_ladder(
            shaft, replace(spec, bar_numbers=bars, allow_bundling=bundle_ok))
        cand = _bsearch_ladder(shaft, ladder, P_col, m_demand)
        if cand is not None:
            return cand
    ladder = _longitudinal_ladder(                    # exhausted -> heaviest
        shaft, replace(spec, bar_numbers=augmented, allow_bundling=True))
    if ladder:
        n, b, bundle = ladder[-1]
        return replace(shaft, n_bars=n, long_bar_no=b, long_bundle=bundle)
    return shaft


def _size_shaft_transverse(shaft: ColumnDesign, v_demand: float, rho_s_req: float,
                           P_col: float, provisions: CodeProvisions,
                           spec: OptimizeSpec) -> ColumnDesign:
    """Smallest transverse steel giving phi*Vn >= ``v_demand`` and rho_s >= req."""
    for b, s, bundle in _confinement_ladder(spec):
        cand = replace(shaft, spiral_bar_no=b, spiral_spacing=s, spiral_bundle=bundle)
        sec = cand.section()
        phiVn, _, _ = shear_capacity(sec, P_col, mu_d=1.0, inside_hinge=False,
                                     provisions=provisions)
        if phiVn >= v_demand and sec.rho_s >= rho_s_req:
            return cand
    b, s, bundle = _confinement_ladder(spec)[-1]        # exhausted -> heaviest
    return replace(shaft, spiral_bar_no=b, spiral_spacing=s, spiral_bundle=bundle)


def _same_shaft(a: ColumnDesign, b: ColumnDesign) -> bool:
    return (a.n_bars, a.long_bar_no, a.long_bundle, a.spiral_bar_no,
            a.spiral_spacing, a.spiral_bundle) == \
           (b.n_bars, b.long_bar_no, b.long_bundle, b.spiral_bar_no,
            b.spiral_spacing, b.spiral_bundle)


def size_shaft(column: ColumnDesign, shaft_start: ColumnDesign, ctx: _Ctx) -> ColumnDesign:
    """Size shaft longitudinal + transverse steel for capacity protection.

    Flexure: shaft Mn >= gamma * (column overstrength moment demand).
    Shear:   phi*Vn_shaft >= column overstrength shear Vo = Mo/Hcol.

    In p-y (soil) mode the demand includes the IN-GROUND moment/shear the shaft
    carries below grade at column overstrength — which usually peaks below the
    interface and exceeds the interface value.  That demand depends on the shaft
    stiffness, so we iterate (size -> re-solve p-y -> resize) to a fixed point,
    the shaft-size iteration Caltrans SDC 2.1 §C6.2.5.3 describes.
    """
    spec = ctx.spec
    P_col = ctx.effective_axial(column.D)
    gamma = ctx.provisions.shaft_demand_factor
    Mo = ctx.provisions.overstrength_factor * _plastic_moment(column, P_col)
    Vo = Mo / ctx.geometry.Hcol
    if ctx.shaft_moment_basis == "fixity":
        Df = max(ctx.geometry.fixity_depth(m) for m in ctx.mults)
        m_interface = Mo * (ctx.geometry.Hcol + Df) / ctx.geometry.Hcol
    else:
        m_interface = Mo
    # SGS 8.9 capacity-protection amplification must be applied here too, or the
    # optimiser sizes for Mo and the check then demands gamma*Mo and fails.
    m_interface *= gamma

    frac = ctx.provisions.shaft_confinement_fraction
    rho_s_req = frac * column.section().rho_s if frac is not None else 0.0

    # 1) size for the INTERFACE demand first (no p-y solve needed).  This alone
    # covers the multiplier path and gives soil mode a near-final starting cage.
    shaft = _size_shaft_longitudinal(replace(shaft_start), m_interface, P_col, spec)
    shaft = _size_shaft_transverse(shaft, Vo, rho_s_req, P_col, ctx.provisions, spec)

    soil_mode = ctx.fixity_source == "soil" and ctx.soil_profile is not None
    if not soil_mode:
        return shaft

    # 2) add the p-y IN-GROUND demand and iterate to a fixed point — the demand
    # depends on the shaft's own EI (Caltrans SDC 2.1 §C6.2.5.3 iteration).  Steel
    # raises capacity faster than it stiffens the shaft, and step 1 starts us
    # close, so this settles in 1-2 passes.  The p-y solves at the converged EI
    # are shared (cached) with evaluate_column's overstrength solves.
    EI_col = _eff_EI(column, P_col)
    L_embed = (ctx.shaft_embed_length or
               (ctx.soil_profile.depth if ctx.soil_profile else 0.0))
    for _ in range(4):
        EI_shaft = _eff_EI(shaft, P_col)
        M_ig, V_ig, _ = inground_demand(
            ctx.geometry.Hcol, L_embed, EI_col, EI_shaft, ctx.geometry.D_shaft,
            P_col, ctx.soil_profile, Vo, ctx.soil_bounds)
        m_demand = max(m_interface, gamma * M_ig)
        v_demand = max(Vo, V_ig)
        new_shaft = _size_shaft_longitudinal(shaft, m_demand, P_col, spec)
        new_shaft = _size_shaft_transverse(new_shaft, v_demand, rho_s_req, P_col,
                                           ctx.provisions, spec)
        if _same_shaft(new_shaft, shaft):
            return new_shaft
        shaft = new_shaft
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
    fixity_source: str = "multiplier",
    soil_profile: object = None,
    shaft_embed_length: float | None = None,
    soil_bounds: tuple[float, float] = (2.0, 0.5),
    max_iterations: int = 400,
    on_candidate=None,
) -> OptimizeResult:
    """Greedy priority-ordered search for a feasible column + shaft design.

    ``on_candidate(iters)`` (optional) is called once per trial design — used by
    the GUI to show live progress within a single (possibly slow, soil p-y)
    column so a long run doesn't look like a crash.
    """
    spec = spec or OptimizeSpec()
    ctx = _Ctx(geometry, spectrum, axial, weight, fixity_multipliers,
               shaft_moment_basis, lle_spectrum, lle_mu_limit, spec,
               concrete_unit_weight, self_weight_mass_factor, self_weight_in_axial,
               provisions, fixity_source, soil_profile, shaft_embed_length,
               soil_bounds)
    design = replace(start)
    log: list[str] = []
    state = {"iters": 0, "shaft": replace(shaft_start)}
    # the shaft ALWAYS tracks the column by this oversize (Type II); the entered
    # shaft is only a starting value.  The code minimum is a hard floor, the
    # user's value can be more.
    min_oversize = max(spec.min_shaft_oversize, provisions.min_shaft_oversize)

    def assess(d: ColumnDesign) -> ColumnAssessment:
        # derive the shaft diameter from the column + oversize (tracks up & down)
        shaft_D = required_shaft_diameter(d.D, min_oversize, spec.shaft_diameters)
        geom_d = (geometry if shaft_D == geometry.D_shaft
                  else replace(geometry, D_shaft=shaft_D))
        ctx_d = ctx if geom_d is geometry else replace(ctx, geometry=geom_d)
        seed = (shaft_start if shaft_D == shaft_start.D
                else replace(shaft_start, D=shaft_D))
        shaft = size_shaft(d, seed, ctx_d)
        state["shaft"] = shaft
        state["iters"] += 1
        if on_candidate is not None:
            on_candidate(state["iters"])
        return evaluate_column(
            d.section(), shaft.section(), geom_d, spectrum, axial, weight,
            fixity_multipliers=fixity_multipliers,
            rho_l_min=spec.rho_l_min, rho_l_max=spec.rho_l_max,
            shaft_moment_basis=shaft_moment_basis,
            lle_spectrum=lle_spectrum, lle_mu_limit=lle_mu_limit,
            concrete_unit_weight=concrete_unit_weight,
            self_weight_mass_factor=self_weight_mass_factor,
            self_weight_in_axial=self_weight_in_axial,
            provisions=provisions,
            fixity_source=fixity_source, soil_profile=soil_profile,
            shaft_embed_length=shaft_embed_length, soil_bounds=soil_bounds,
        )

    def greedy_fixed(seed: ColumnDesign, inner: tuple):
        """First feasible design at a fixed diameter, escalating the ``inner``
        parameters only (or the last tried if none pass)."""
        d, a = seed, assess(seed)
        if a.passed:
            return d, a, state["shaft"]
        for param in inner:
            if param not in spec.variable:
                continue
            if state["iters"] >= max_iterations:
                break
            d, a = _sweep_parameter(param, d, spec, assess, log, state,
                                    max_iterations, ctx)
            if a.passed:
                return d, a, state["shaft"]
        return d, a, state["shaft"]

    all_inner = tuple(p for p in spec.priority if p != "diameter")

    # Fixed column diameter — either the user excluded it from the search, or the
    # "fixed_diameter" objective (pick the diameter, minimise the reinforcement).
    # Seed the MIN steel so the greedy escalation returns the smallest feasible
    # longitudinal ratio at the entered diameter.
    if "diameter" not in spec.variable or spec.objective == "fixed_diameter":
        seed = (_min_steel_design(design, spec)
                if "longitudinal" in spec.variable else design)
        d, a, sh = greedy_fixed(seed, all_inner)
        log.append(f"Fixed D={d.D:g} in: "
                   + (f"feasible at rho_l={d.rho_l():.4f}" if a.passed
                      else "no feasible reinforcement"))
        return OptimizeResult(d, sh, a, a.passed, state["iters"], log)

    # Objective-driven diameter search.  The optimiser starts at the MINIMUM
    # column diameter and grows, seeking the SMALLEST diameter that is feasible:
    #   min_diameter -> feasible with steel escalated as needed (min-feasible).
    #   target_steel -> feasible with longitudinal held at ~target_rho (only
    #                   confinement/f'c escalate); one probe per diameter.
    #   min_steel    -> target_steel at the rho_l_min floor.
    # Feasibility is monotonic in diameter (a bigger column has more capacity),
    # so the smallest feasible diameter is found by BINARY SEARCH of the ladder
    # (~log2(N) trial columns), not a linear march.
    diams = sorted(spec.diameters)
    fixed_steel = spec.objective in ("target_steel", "min_steel")
    target = spec.rho_l_min if spec.objective == "min_steel" else spec.target_rho
    probed: dict[int, tuple] = {}

    def probe(i: int) -> tuple:
        if i not in probed:
            base = replace(start, D=diams[i])
            if fixed_steel:                       # hold longitudinal at ~target
                seed = _cage_at_ratio(base, spec, target)
                d, a, sh = greedy_fixed(seed, ("confinement", "fc"))
            else:                                 # min_diameter: escalate steel
                # NB: a cheap min/max-steel probe does NOT work here — for a
                # capacity-protected column MORE longitudinal steel raises Mp and
                # so RAISES the overstrength shear/shaft demand, i.e. both too
                # little steel (low capacity) AND too much (high demand) fail, so
                # the feasible band is in the middle.  The steel must be searched.
                d, a, sh = greedy_fixed(_min_steel_design(base, spec), all_inner)
            probed[i] = (d, sh, a)
            log.append(f"D={diams[i]:g} in: "
                       + (f"feasible at rho_l={d.rho_l():.4f}" if a.passed
                          else "infeasible"))
        return probed[i]

    lo, hi, found = 0, len(diams), None
    while lo < hi:
        if state["iters"] >= max_iterations:
            break
        mid = (lo + hi) // 2
        if probe(mid)[2].passed:
            found, hi = mid, mid                 # feasible — try smaller
        else:
            lo = mid + 1                         # need a larger column

    if found is not None:
        d, sh, a = probed[found]
        log.append(f"Objective '{spec.objective}': D={d.D:g} in, "
                   f"rho_l={d.rho_l():.4f}.")
        return OptimizeResult(d, sh, a, True, state["iters"], log)

    # nothing feasible even at the largest diameter
    d, sh, a = probe(len(diams) - 1)
    log.append("No feasible design across the diameter ladder.")
    return OptimizeResult(d, sh, a, False, state["iters"], log)


def _sweep_parameter(param, design, spec, assess, log, state, max_iterations, ctx):
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

    # NB: "diameter" is handled by the objective-driven outer sweep in
    # optimize_column (which grows the shaft to keep the Type II oversize), not
    # here — inner_priority excludes it.

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
