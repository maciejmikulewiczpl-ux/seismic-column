"""Soil strata and p-y curves for laterally-loaded pile analysis.

This module supplies the geotechnical side of the in-house lateral-pile
(soil–structure-interaction) analysis that replaces the assumed point of fixity.
It defines a layered :class:`SoilProfile` and the nonlinear **p-y curves** that
give the soil reaction ``p`` (force per unit length) versus pile deflection
``y`` at any depth, following the same public-domain formulations LPile uses:

* **Matlock (1970) soft clay** — cyclic.
* **API / O'Neill–Murchison sand** — cyclic (A = 0.9).
* **Welch & Reese stiff clay (no free water)** — static envelope (see note).

Seismic loading is cyclic, so the cyclic branches are used where defined.

Units
-----
All internal quantities are **kip and inch** (consistent with the structural
side: EI in kip-in^2, deflection in in, ``p`` in kip/in, soil modulus
``Es = p/y`` in kip/in^2).  :meth:`SoilLayer.from_engineering` converts the
usual geotechnical inputs (ft, pcf, ksf, degrees, pci) — the same values an
engineer types into LPile.

References
----------
Matlock, H. (1970). "Correlations for Design of Laterally Loaded Piles in Soft
    Clay." OTC 1204.
API RP 2GEO / RP 2A-WSD; O'Neill, M.W. & Murchison, J.M. (1983).
Reese, L.C., Cox, W.R., Koop, F.D. (1974). "Analysis of Laterally Loaded Piles
    in Sand." OTC 2080.
Welch, R.C. & Reese, L.C. (1972). Laterally loaded behavior of drilled shafts.
Davisson, M.T. & Robinson, K.E. (1965). "Bending and buckling of partially
    embedded piles."
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Unit conversions to internal (kip, in)
# ---------------------------------------------------------------------------
FT_TO_IN = 12.0
PCF_TO_KCI = 1.0 / (1000.0 * 1728.0)   # lb/ft^3 -> kip/in^3
KSF_TO_KSI = 1.0 / 144.0               # kip/ft^2 -> kip/in^2
PCI_TO_KCI = 1.0 / 1000.0              # lb/in^3 -> kip/in^3

CLAY_MODELS = ("matlock_soft_clay", "welch_stiff_clay")
SAND_MODELS = ("api_sand",)
ELASTIC_MODELS = ("elastic_subgrade",)   # constant linear modulus (k_py = Es, kip/in^2)
PY_MODELS = CLAY_MODELS + SAND_MODELS + ELASTIC_MODELS


@dataclass
class SoilLayer:
    """One soil stratum, in internal kip/inch units.

    Parameters
    ----------
    thickness:
        Layer thickness, in.
    py_model:
        p-y curve model key (see :data:`PY_MODELS`).
    gamma_eff:
        Effective (buoyant below the water table) unit weight, kip/in^3.
    su_top, su_bot:
        Undrained shear strength at the top / bottom of the layer, ksi
        (clay models). Linearly interpolated within the layer.
    eps50:
        Strain at 50% strength (clay models), dimensionless.
    phi:
        Effective friction angle, degrees (sand models).
    k_py:
        Initial modulus of subgrade reaction, kip/in^3 (sand & stiff clay).
    """

    thickness: float
    py_model: str
    gamma_eff: float
    su_top: float = 0.0
    su_bot: float = 0.0
    eps50: float = 0.01
    phi: float = 0.0
    k_py: float = 0.0

    def __post_init__(self) -> None:
        if self.py_model not in PY_MODELS:
            raise ValueError(f"unknown p-y model {self.py_model!r}; "
                             f"choose from {PY_MODELS}")
        if self.su_bot == 0.0 and self.su_top != 0.0:
            self.su_bot = self.su_top

    @property
    def is_clay(self) -> bool:
        return self.py_model in CLAY_MODELS

    def su_at(self, z_in_layer: float) -> float:
        """Undrained strength at depth ``z_in_layer`` below the layer top, ksi."""
        if self.thickness <= 0.0:
            return self.su_top
        f = min(max(z_in_layer / self.thickness, 0.0), 1.0)
        return self.su_top + (self.su_bot - self.su_top) * f

    @classmethod
    def from_engineering(
        cls,
        thickness_ft: float,
        py_model: str,
        gamma_pcf: float,
        *,
        su_top_ksf: float = 0.0,
        su_bot_ksf: float | None = None,
        eps50: float = 0.01,
        phi_deg: float = 0.0,
        k_pci: float = 0.0,
        submerged: bool = False,
        gamma_water_pcf: float = 62.4,
    ) -> "SoilLayer":
        """Build a layer from the usual geotechnical (LPile) input units.

        ``gamma_pcf`` is the *total* unit weight; when ``submerged`` the buoyant
        (effective) weight ``gamma - gamma_water`` is stored.
        """
        g = gamma_pcf - (gamma_water_pcf if submerged else 0.0)
        su_bot = su_top_ksf if su_bot_ksf is None else su_bot_ksf
        return cls(
            thickness=thickness_ft * FT_TO_IN,
            py_model=py_model,
            gamma_eff=g * PCF_TO_KCI,
            su_top=su_top_ksf * KSF_TO_KSI,
            su_bot=su_bot * KSF_TO_KSI,
            eps50=eps50,
            phi=phi_deg,
            k_py=k_pci * PCI_TO_KCI,
        )


@dataclass
class SoilProfile:
    """A stack of :class:`SoilLayer` from the top of shaft (= ground surface)."""

    layers: tuple[SoilLayer, ...]
    J: float = 0.5                         # Matlock/stiff-clay bearing factor
    cyclic: bool = True                    # seismic -> cyclic p-y branches
    stiffness_factor: float = 1.0          # scales p (and Es) for upper/lower bounds
    # cumulative layer-top depths (in), filled on init
    _tops: tuple[float, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        tops, z = [], 0.0
        for lyr in self.layers:
            tops.append(z)
            z += lyr.thickness
        object.__setattr__(self, "_tops", tuple(tops))

    @property
    def depth(self) -> float:
        """Total profiled depth, in."""
        return sum(lyr.thickness for lyr in self.layers)

    def signature(self) -> tuple:
        """Hashable summary of the profile (for caching pile solves)."""
        return (self.J, self.cyclic, round(self.stiffness_factor, 6)) + tuple(
            (l.py_model, round(l.thickness, 4), round(l.gamma_eff, 12),
             round(l.su_top, 8), round(l.su_bot, 8), round(l.eps50, 6),
             round(l.phi, 4), round(l.k_py, 10)) for l in self.layers)

    def layer_at(self, z: float) -> tuple[SoilLayer, float]:
        """Return (layer, depth-below-its-top) for global depth ``z`` (in)."""
        for lyr, top in zip(self.layers, self._tops):
            if z < top + lyr.thickness or lyr is self.layers[-1]:
                return lyr, z - top
        return self.layers[-1], z - self._tops[-1]

    def sigma_v_eff(self, z: float) -> float:
        """Effective vertical (overburden) stress at depth ``z`` (in), ksi."""
        s, remaining = 0.0, z
        for lyr in self.layers:
            dz = min(remaining, lyr.thickness)
            if dz <= 0.0:
                break
            s += lyr.gamma_eff * dz
            remaining -= dz
        if remaining > 0.0 and self.layers:            # below the profile
            s += self.layers[-1].gamma_eff * remaining
        return s

    # ------------------------------------------------------------------
    def p_ult(self, z: float, D: float) -> float:
        """Ultimate soil resistance per unit length at depth ``z``, kip/in."""
        lyr, zl = self.layer_at(z)
        if lyr.py_model in ELASTIC_MODELS:
            return float("inf")                        # linear, no cap
        sv = self.sigma_v_eff(z)
        if lyr.is_clay:
            su = max(lyr.su_at(zl), 1e-9)
            Np = min(3.0 + sv / su + self.J * z / D, 9.0)
            return Np * su * D
        return _sand_pu(lyr.phi, D, z, sv)

    def p_of_y(self, z: float, y: float, D: float) -> float:
        """Soil reaction ``p`` (kip/in) at depth ``z`` for deflection ``y`` (in)."""
        y = abs(y)
        lyr, zl = self.layer_at(z)
        if lyr.py_model == "elastic_subgrade":
            base = lyr.k_py * y                         # k_py = constant Es
        else:
            pu = self.p_ult(z, D)
            if lyr.py_model == "matlock_soft_clay":
                base = _matlock(pu, y, D, lyr.eps50, z,
                                self._matlock_XR(lyr, zl, D), self.cyclic)
            elif lyr.py_model == "welch_stiff_clay":
                base = _welch_stiff(pu, y, D, lyr.eps50)
            elif lyr.py_model == "api_sand":
                A = 0.9 if self.cyclic else max(3.0 - 0.8 * z / D, 0.9)
                base = _api_sand(pu, y, lyr.k_py, z, A)
            else:
                raise ValueError(lyr.py_model)
        return self.stiffness_factor * base

    def secant_modulus(self, z: float, y: float, D: float) -> float:
        """Secant soil modulus Es = p/y (kip/in^2) for the FD solver.

        As ``y`` -> 0 returns the initial tangent so the assembled system stays
        well-conditioned.
        """
        y = abs(y)
        if y < 1e-9:
            return self._initial_modulus(z, D)
        return self.p_of_y(z, y, D) / y

    def _initial_modulus(self, z: float, D: float) -> float:
        lyr, zl = self.layer_at(z)
        if lyr.py_model == "elastic_subgrade":
            return max(self.stiffness_factor * lyr.k_py, 1e-6)   # constant Es
        if lyr.py_model == "api_sand":
            return max(self.stiffness_factor * lyr.k_py * z, 1e-6)  # k*z
        # clay: initial slope of the (y/yc)^(1/n) curve is infinite at y=0;
        # use the secant at a small reference deflection (0.1*yc) instead.
        y50 = 2.5 * lyr.eps50 * D
        y_ref = 0.1 * y50
        return self.p_of_y(z, y_ref, D) / y_ref        # already scaled

    def _matlock_XR(self, lyr: SoilLayer, zl: float, D: float) -> float:
        su = max(lyr.su_at(zl), 1e-9)
        denom = lyr.gamma_eff * D / su + self.J
        return 6.0 * D / denom if denom > 0 else 1e9

    # ------------------------------------------------------------------
    # p-y curve export (for use as springs in a global structural model)
    # ------------------------------------------------------------------
    def py_curve(self, z: float, D: float, y_max: float | None = None,
                 n: int = 41) -> tuple[np.ndarray, np.ndarray]:
        """Return (y, p) arrays for the p-y curve at depth ``z`` (in).

        ``p`` is soil reaction per unit length, kip/in; ``y`` is deflection, in.
        Default ``y_max`` = 0.15·D spans well past mobilisation of most curves.
        """
        y_max = y_max if y_max is not None else 0.15 * D
        ys = np.linspace(0.0, y_max, n)
        ps = np.array([self.p_of_y(z, float(y), D) for y in ys])
        return ys, ps

    def representative_depths(self, D: float, embed: float) -> list[float]:
        """A handful of depths (in) that characterise the lateral response.

        One per layer mid-height within the embedment, plus a fine set through
        the upper 8·D where the p-y springs dominate the pile-head stiffness.
        """
        deep = [z for z in np.arange(0.5 * D, min(8.0 * D, embed) + 1e-9, D)]
        mids = []
        for lyr, top in zip(self.layers, self._tops):
            zc = top + 0.5 * lyr.thickness
            if zc <= embed:
                mids.append(zc)
        depths = sorted({round(z, 2) for z in deep + mids if 0 < z <= embed})
        return depths


# ---------------------------------------------------------------------------
# p-y curve kernels (all kip/in units)
# ---------------------------------------------------------------------------
def _matlock(pu: float, y: float, D: float, eps50: float, z: float,
             XR: float, cyclic: bool) -> float:
    """Matlock (1970) soft-clay p-y (static or cyclic)."""
    yc = 2.5 * eps50 * D
    if yc <= 0.0:
        return pu
    if not cyclic:
        if y <= 8.0 * yc:
            return 0.5 * pu * (y / yc) ** (1.0 / 3.0)
        return pu
    # cyclic
    if y <= 3.0 * yc:
        return 0.5 * pu * (y / yc) ** (1.0 / 3.0)
    if z >= XR:                                        # deep: flat at 0.72 pu
        return 0.72 * pu
    # shallow: degrade toward 0.72 pu * (z/XR)
    if y <= 15.0 * yc:
        return 0.72 * pu * (1.0 - (1.0 - z / XR) * (y - 3.0 * yc) / (12.0 * yc))
    return 0.72 * pu * (z / XR)


def _welch_stiff(pu: float, y: float, D: float, eps50: float) -> float:
    """Welch & Reese stiff-clay (no free water), static envelope.

    Note: stiff-clay *cyclic* degradation is N-cycle dependent and is not
    applied here; the static curve is used as a conservative envelope. Prefer
    Matlock soft clay where a proper cyclic curve matters.
    """
    y50 = 2.5 * eps50 * D
    if y50 <= 0.0:
        return pu
    if y <= 16.0 * y50:
        return 0.5 * pu * (y / y50) ** 0.25
    return pu


def _sand_pu(phi_deg: float, D: float, z: float, sv: float) -> float:
    """API/Reese sand ultimate resistance per unit length, kip/in."""
    phi = math.radians(max(phi_deg, 1.0))
    a = phi / 2.0
    b = math.pi / 4.0 + phi / 2.0
    K0 = 0.4
    Ka = math.tan(math.pi / 4.0 - phi / 2.0) ** 2
    tb, tphi, ta = math.tan(b), math.tan(phi), math.tan(a)
    tbmphi = math.tan(b - phi)
    C1 = (tb ** 2 * ta) / tbmphi + K0 * (
        (tphi * math.sin(b)) / (math.cos(a) * tbmphi)
        + tb * (tphi * math.sin(b) - ta))
    C2 = tb / tbmphi - Ka
    C3 = Ka * (tb ** 8 - 1.0) + K0 * tphi * tb ** 4
    # sv = effective overburden = gamma'*z (generalised for layering)
    pus = (C1 * z + C2 * D) * sv
    pud = C3 * D * sv
    return max(min(pus, pud), 1e-9)


def _api_sand(pu: float, y: float, k_py: float, z: float, A: float) -> float:
    """O'Neill–Murchison / API sand p-y: p = A*pu*tanh(k*z*y/(A*pu))."""
    if pu <= 0.0 or A <= 0.0:
        return 0.0
    arg = (k_py * z * y) / (A * pu)
    return A * pu * math.tanh(arg)


# ---------------------------------------------------------------------------
# Davisson–Robinson closed-form equivalent depth-to-fixity (cross-check)
# ---------------------------------------------------------------------------
def davisson_fixity_depth(EI: float, profile: SoilProfile, D: float,
                          ref_depth: float | None = None) -> float:
    """Approximate depth to fixity below ground (Davisson & Robinson 1965), in.

    A linear-limit cross-check / initial guess only — it collapses the layered,
    nonlinear profile to a single equivalent modulus taken over the upper
    ``ref_depth`` (default 5*D, the zone that dominates lateral response):

    * cohesive-dominated:  R = (EI/k_h)^(1/4),  Lf ~= 1.4*R
    * cohesionless:        T = (EI/n_h)^(1/5),  Lf ~= 1.8*T

    where ``k_h`` (constant modulus) and ``n_h`` (modulus gradient) are averaged
    from the initial p-y tangents in the reference zone.
    """
    ref = ref_depth if ref_depth is not None else 5.0 * D
    n = 20
    dz = ref / n
    # sample initial modulus Es0(z) = p-tangent at small y over the zone
    sand_like = 0
    k_sum = 0.0       # constant-modulus average (clay)
    nh_sum = 0.0      # gradient average nh = Es0/z (sand)
    cnt = 0
    for i in range(1, n + 1):
        z = i * dz
        lyr, _ = profile.layer_at(z)
        Es0 = profile._initial_modulus(z, D)          # kip/in^2
        kh = Es0 / D                                   # modulus per unit length
        k_sum += kh
        nh_sum += Es0 / z / D if z > 0 else 0.0
        if lyr.py_model in SAND_MODELS:
            sand_like += 1
        cnt += 1
    if cnt == 0:
        return 1.8 * D
    if sand_like >= cnt / 2:                            # sand-dominated
        nh = max(nh_sum / cnt, 1e-9)
        T = (EI / nh) ** 0.2
        return 1.8 * T
    kh = max(k_sum / cnt, 1e-9)
    R = (EI / kh) ** 0.25
    return 1.4 * R
