"""Finite-element lateral pile (beam-on-nonlinear-Winkler) solver.

Models the **column and shaft as one continuous Euler–Bernoulli beam-column**:
the column segment above ground (rigidity ``EI_col``, no soil) and the embedded
shaft below (``EI_shaft``, nonlinear p-y soil springs). A lateral point load
``V`` is applied at the **column top** — the moment at the top of shaft,
``M = V·Hcol``, and the resulting shaft deflection therefore include *both* the
shear and the column overturning moment (the beam is continuous).

The nonlinear p-y springs (:mod:`seismic_column.soil`) are handled by **secant
iteration**: each pass updates the nodal lateral spring from the p-y secant
modulus ``Es = p/y`` at the current deflection, until the deflection converges.
Axial load enters through the consistent **geometric stiffness** (P-Δ).

Units: kip, inch throughout (EI kip-in², loads kip, deflection in, moment
kip-in). The solver returns the pile-head flexibility/stiffness, the deflection
and moment profiles, the max in-ground shaft moment, and an equivalent
depth-to-fixity ``Df_eq`` matching the two-segment cantilever.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from .soil import SoilProfile

_BW = 3   # half-bandwidth of the 2-DOF Euler-Bernoulli beam global matrix


@dataclass
class PileSolution:
    x: np.ndarray               # distance from column top, in (0 = top)
    y: np.ndarray               # lateral deflection, in
    moment: np.ndarray          # bending moment, kip-in
    shear: np.ndarray           # shear V = dM/dx, kip
    ground_index: int           # first node at/below ground (top of shaft)
    head_deflection: float      # y at the column top, in
    head_stiffness: float       # V / y_top, kip/in
    head_flexibility: float     # y_top / V, in/kip
    Df_eq: float                # equivalent depth to fixity below top of shaft, in
    max_inground_moment: float  # max |M| below ground, kip-in
    max_moment_depth: float     # its depth below top of shaft, in
    converged: bool
    iterations: int
    stable: bool = True         # physical solution (positive, bounded deflection)
    max_inground_shear: float = 0.0   # max |V| below ground, kip


def _ke_elastic(EI: float, L: float) -> np.ndarray:
    """Euler–Bernoulli elastic 4x4 stiffness, DOF [v_i, θ_i, v_j, θ_j]."""
    L2, L3 = L * L, L * L * L
    return (EI / L3) * np.array([
        [12.0,   6*L,  -12.0,   6*L],
        [6*L,  4*L2,   -6*L,  2*L2],
        [-12.0, -6*L,   12.0,  -6*L],
        [6*L,  2*L2,   -6*L,  4*L2],
    ])


def _element_matrices(EI: float, L: float, P: float) -> np.ndarray:
    """Elastic + geometric (P-Δ) stiffness; compression (P>0) softens laterally."""
    L2 = L * L
    kg = (P / (30.0 * L)) * np.array([
        [36.0,   3*L,  -36.0,   3*L],
        [3*L,  4*L2,   -3*L,  -L2],
        [-36.0, -3*L,   36.0,  -3*L],
        [3*L,  -L2,   -3*L,  4*L2],
    ])
    return _ke_elastic(EI, L) - kg


def solve_lateral(
    Hcol: float,
    L_embed: float,
    EI_col: float,
    EI_shaft: float,
    D_shaft: float,
    axial: float,
    soil: SoilProfile,
    V_head: float,
    *,
    target_h: float | None = None,
    max_nodes: int = 241,
    max_iter: int = 100,
    tol: float = 1e-6,
    relax: float = 1.0,
) -> PileSolution:
    """Solve the laterally-loaded column+shaft and return a :class:`PileSolution`.

    ``V_head`` is the lateral load at the column top (use ``Mp/Hcol`` for the
    effective/secant head stiffness at yield). ``target_h`` sets the element
    length (default ≈ D_shaft/4, capped by ``max_nodes``).
    """
    L_total = Hcol + L_embed
    h_t = target_h if target_h else max(D_shaft / 4.0, L_total / (max_nodes - 1))
    # mesh with a node EXACTLY at the ground line (Hcol): separate column & shaft
    # meshes sharing that node, so the diagrams and the interface moment are clean
    n_col = max(int(round(Hcol / h_t)), 2) if Hcol > 0 else 0
    budget = max_nodes - 1 - n_col
    n_shaft = max(min(int(round(L_embed / h_t)), budget), 10)
    x_col = np.linspace(0.0, Hcol, n_col + 1) if n_col else np.array([0.0])
    x_shaft = np.linspace(Hcol, L_total, n_shaft + 1)[1:]
    x = np.concatenate([x_col, x_shaft])
    n_node = x.size
    n_elem = n_node - 1
    ndof = 2 * n_node

    ground_index = n_col                             # the node at Hcol
    depth = np.clip(x - Hcol, 0.0, None)             # depth below top of shaft
    embedded = x >= Hcol - 1e-9
    he = np.diff(x)                                   # per-element length, in
    # nodal tributary length = half of each adjacent element
    trib = np.zeros(n_node)
    trib[:-1] += 0.5 * he
    trib[1:] += 0.5 * he

    EI_elem = np.where(np.arange(n_elem) < max(n_col, 0), EI_col, EI_shaft)

    # pre-assemble the constant elastic + geometric global matrix in banded form
    # (ab[_BW + i - j, j] = K[i, j]); only the soil-spring diagonal changes per pass
    ab0 = np.zeros((2 * _BW + 1, ndof))
    for e in range(n_elem):
        ke = _element_matrices(EI_elem[e], he[e], axial)
        d = (2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3)
        for a, ia in enumerate(d):
            for b, jb in enumerate(d):
                ab0[_BW + ia - jb, jb] += ke[a, b]
    F = np.zeros(ndof)
    F[0] = V_head                                     # lateral load at column top
    spring_dofs = [2 * i for i in range(n_node) if embedded[i]]
    spring_depth = [depth[i] for i in range(n_node) if embedded[i]]
    spring_trib = [trib[i] for i in range(n_node) if embedded[i]]

    v = np.zeros(n_node)
    u = np.zeros(ndof)
    converged = False
    prev_dv = np.inf
    runaway = 20.0 * L_total          # deflection beyond this = clearly unstable
    stall = 0                         # consecutive non-improving iterations
    it = 0
    for it in range(1, max_iter + 1):
        ab = ab0.copy()
        for dof, zc, tb in zip(spring_dofs, spring_depth, spring_trib):
            Es = soil.secant_modulus(zc, v[dof // 2], D_shaft)      # kip/in^2
            ab[_BW, dof] += Es * tb                                  # lateral spring
        # A P-Δ / very-soft-soil candidate can drive Ke - Kg to (near-)singular.
        # LAPACK then raises LinAlgError (or returns inf/nan). Treat that as an
        # UNSTABLE analysis — the physicality guard below yields the flexible
        # sentinel — rather than letting it crash the whole batch row.
        try:
            u_new = solve_banded((_BW, _BW), ab, F)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(u_new)):
            break
        u = u_new
        v_new = u[0::2]
        dv = np.max(np.abs(v_new - v))
        # Early bail-outs for the (common, expensive) UNSTABLE cases — a soft
        # bound that P-Δ-buckles or oscillates would otherwise burn all max_iter
        # secant passes only to be flagged unstable anyway.  Both leave
        # converged=False, so the physicality guard returns the flexible sentinel.
        if np.max(np.abs(v_new)) > runaway:            # deflection running away
            break
        stall = stall + 1 if dv >= prev_dv else 0
        if stall >= 12:                                # not converging (oscillating)
            break
        # divergence guard: back off the step if the update grows
        r = relax if dv <= prev_dv else max(relax * 0.5, 0.2)
        v = r * v_new + (1.0 - r) * v
        prev_dv = dv
        if dv / max(np.max(np.abs(v)), 1e-9) < tol:
            v = v_new
            converged = True
            break

    theta = u[1::2]
    # moment & shear from the FE element end-forces (EI-consistent, so the
    # column/shaft rigidity change at the ground line is handled correctly —
    # the internal moment stays continuous there).
    moment = np.zeros(n_node)
    shear = np.zeros(n_node)
    wt = np.zeros(n_node)
    for e in range(n_elem):
        fe = _ke_elastic(EI_elem[e], he[e]) @ np.array(
            [v[e], theta[e], v[e + 1], theta[e + 1]])
        # internal diagram sign: M(top)=0, M(ground)=+V*Hcol ; V(top)=+V_head
        moment[e] += -fe[1]
        moment[e + 1] += fe[3]
        shear[e] += fe[0]
        shear[e + 1] += -fe[2]
        wt[e] += 1
        wt[e + 1] += 1
    moment /= np.maximum(wt, 1)
    shear /= np.maximum(wt, 1)

    y_top = float(v[0])
    below = depth > 0.0
    if np.any(below):
        idx = np.argmax(np.abs(moment) * below)
        max_m = float(abs(moment[idx]))
        max_m_depth = float(depth[idx])
        max_v = float(np.max(np.abs(shear[below])))
    else:
        max_m, max_m_depth, max_v = 0.0, 0.0, 0.0

    # Physicality guard: a positive load must give a positive, bounded head
    # deflection.  Soft soil (or P-Δ buckling) under a very high demand can
    # produce a negative or runaway deflection — treat that as an UNSTABLE
    # analysis (soil too soft / pile too short for the force), NOT as a stiff
    # base.  A stiff base (Df_eq -> 0) would be dangerously unconservative.
    same_sign = (V_head == 0.0) or (y_top * V_head > 0.0)
    bounded = abs(y_top) < 5.0 * L_total
    stable = bool(converged and same_sign and bounded)
    if stable and V_head != 0.0:
        f_soil = y_top / V_head
        Df_eq = equivalent_fixity_depth(f_soil, Hcol, EI_col, EI_shaft)
        head_stiff = V_head / y_top
    else:                                            # failed / unstable
        f_soil = float("inf")
        Df_eq = 50.0 * D_shaft                        # sentinel: very flexible
        head_stiff = 0.0
        stable = False

    return PileSolution(
        x=x, y=v, moment=moment, shear=shear, ground_index=ground_index,
        head_deflection=y_top, head_stiffness=head_stiff,
        head_flexibility=f_soil, Df_eq=Df_eq,
        max_inground_moment=max_m, max_moment_depth=max_m_depth,
        converged=converged, iterations=it, stable=stable,
        max_inground_shear=max_v,
    )


def equivalent_fixity_depth(f_soil: float, Hcol: float, EI_col: float,
                            EI_shaft: float) -> float:
    """Depth to fixity Df below top of shaft matching the head flexibility.

    Inverts the two-segment cantilever flexibility
    ``f = Hcol³/(3·EI_col) + ((Hcol+Df)³ − Hcol³)/(3·EI_shaft)`` for ``Df`` so
    the existing :class:`~seismic_column.geometry.Geometry` machinery reproduces
    the soil-derived head displacement. Returns 0 if the soil is so stiff that
    the column flexure alone already exceeds ``f_soil``.
    """
    f_col = Hcol ** 3 / (3.0 * EI_col)
    if not np.isfinite(f_soil) or f_soil <= f_col:
        return 0.0
    # solve (Hcol+Df)^3 = Hcol^3 + 3*EI_shaft*(f_soil - f_col)
    le_cubed = Hcol ** 3 + 3.0 * EI_shaft * (f_soil - f_col)
    Le = le_cubed ** (1.0 / 3.0)
    return max(Le - Hcol, 0.0)
