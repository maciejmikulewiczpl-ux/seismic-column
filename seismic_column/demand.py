"""Seismic displacement demand via the Equivalent Static Analysis (ESA) method.

The design acceleration response spectrum is defined in the AASHTO / Caltrans
form from two spectral ordinates (short-period ``Sds`` and one-second ``Sd1``,
both in g).  For a single-degree-of-freedom cantilever the effective period is
computed from the cracked lateral stiffness and the tributary mass, the spectral
acceleration is read from the spectrum, and the displacement demand follows from
the equal-displacement assumption:

    T   = 2*pi*sqrt(m/k)
    Sa  = spectrum(T)                    [g]
    Dd  = Sa*g*(T/(2*pi))^2              [in]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from . import G_IN_S2


@dataclass
class DesignSpectrum:
    """AASHTO/Caltrans-form design acceleration response spectrum (g vs T)."""

    Sds: float  # short-period spectral acceleration, g
    Sd1: float  # 1-second spectral acceleration, g

    @property
    def Ts(self) -> float:
        return self.Sd1 / self.Sds

    @property
    def T0(self) -> float:
        return 0.2 * self.Ts

    def Sa(self, T: float) -> float:
        """Spectral acceleration (g) at period ``T`` (s)."""
        if T <= 0.0:
            return 0.4 * self.Sds
        if T < self.T0:
            return self.Sds * (0.4 + 0.6 * T / self.T0)
        if T <= self.Ts:
            return self.Sds
        return self.Sd1 / T


@dataclass
class TabularSpectrum:
    """User-supplied ARS curve given as (period, Sa) points, linearly interpolated."""

    periods: tuple[float, ...]
    accels: tuple[float, ...]

    @property
    def Ts(self) -> float:
        """Effective corner period Ts for the short-period magnification Rd.

        Derived from the ARS curve using the code's own definitions
        (AASHTO SGS §4.3.3 / Article 3.5):

            SDS = 0.9 · max(Sa)            (90% of the peak Sa)
            SD1 = 0.9 · max(T·Sa)          (90% of the peak spectral velocity,
                                            over the 1–5 s window)
            Ts  = SD1 / SDS = max(T·Sa | 1..5 s) / max(Sa)

        The 0.9 factors cancel, so no site-class input is needed.  The 1–5 s
        window is the softer-site (more conservative) range of §3.5; it captures
        long-period velocity peaks that a 1–2 s window would miss.
        """
        peak_sa = max(self.accels)
        if peak_sa <= 0.0:
            return 0.0
        # sample T·Sa over the 1–5 s window at the nodes and a fine grid so an
        # interpolated peak between nodes is not missed.
        t_hi = min(5.0, self.periods[-1])
        if t_hi < 1.0:
            return 0.0
        grid = [1.0 + 0.05 * i for i in range(int((t_hi - 1.0) / 0.05) + 1)]
        grid += [t for t in self.periods if 1.0 <= t <= t_hi]
        max_tsa = max(t * self.Sa(t) for t in grid)
        return max_tsa / peak_sa

    def Sa(self, T: float) -> float:
        """Spectral acceleration (g) at period ``T`` (s) by linear interpolation."""
        p = self.periods
        a = self.accels
        if T <= p[0]:
            return a[0]
        if T >= p[-1]:
            return a[-1]
        # linear interpolation between bracketing points
        for i in range(1, len(p)):
            if T <= p[i]:
                t0, t1 = p[i - 1], p[i]
                a0, a1 = a[i - 1], a[i]
                return a0 + (a1 - a0) * (T - t0) / (t1 - t0)
        return a[-1]


@dataclass
class SpectrumSpec:
    """Serialisable spectrum specification (parametric or tabular ARS curve)."""

    kind: str = "parametric"          # "parametric" | "tabular"
    Sds: float = 1.0
    Sd1: float = 0.6
    periods: tuple[float, ...] = ()
    accels: tuple[float, ...] = ()

    def build(self):
        """Return a spectrum object exposing ``Sa(T)``."""
        if self.kind == "tabular" and len(self.periods) >= 2:
            pts = sorted(zip(self.periods, self.accels))
            p = tuple(float(x) for x, _ in pts)
            a = tuple(float(y) for _, y in pts)
            return TabularSpectrum(p, a)
        return DesignSpectrum(self.Sds, self.Sd1)



@dataclass
class DemandResult:
    period: float          # effective period, s
    Sa: float             # spectral acceleration, g
    disp_demand: float    # displacement demand Dd, in (incl. Rd if applied)
    stiffness: float      # lateral stiffness used, kip/in
    mass: float           # tributary mass, kip*s^2/in
    disp_elastic: float = 0.0   # Dd before short-period magnification, in
    Rd: float = 1.0             # SGS 4.3.3 short-period magnification applied
    Ts: float = 0.0             # corner period used for Rd (0 if not applied)
    mu_for_Rd: float = 0.0      # displacement ductility used in the Rd fixed point


def short_period_magnification(T: float, Ts: float, mu_d: float) -> float:
    """Short-period displacement magnification Rd (SGS 4.3.3), dimensionless.

    Rd = (1 - 1/mu_D)*(T*/T) + 1/mu_D >= 1.0  for T*/T > 1.0, else 1.0,
    with T* = 1.25*Ts (Eq. 4.3.3-3).  The equal-displacement assumption breaks
    down for short-period structures; Rd corrects the elastic estimate.
    """
    if T <= 0.0 or Ts <= 0.0 or mu_d <= 0.0:
        return 1.0
    T_star = 1.25 * Ts
    if T_star / T <= 1.0:
        return 1.0
    Rd = (1.0 - 1.0 / mu_d) * (T_star / T) + 1.0 / mu_d
    return max(Rd, 1.0)


def magnified_demand(demand: DemandResult, spectrum, delta_y: float,
                     iterations: int = 25) -> DemandResult:
    """Apply SGS 4.3.3 to ``demand``, solving the Rd/mu_D circularity.

    Rd depends on mu_D = Dd/delta_y, which itself depends on Rd, so this
    iterates to a fixed point.  Returns the original result unchanged when the
    spectrum exposes no ``Ts`` (e.g. a tabular ARS curve) or the structure is
    not short-period.
    """
    Ts = getattr(spectrum, "Ts", None)
    if Ts is None or delta_y <= 0.0 or demand.period <= 0.0:
        return demand
    dd_elastic = demand.disp_demand
    dd = dd_elastic
    Rd = 1.0
    for _ in range(iterations):
        Rd_new = short_period_magnification(demand.period, Ts, dd / delta_y)
        dd_new = Rd_new * dd_elastic
        if abs(dd_new - dd) < 1e-9:
            dd, Rd = dd_new, Rd_new
            break
        dd, Rd = dd_new, Rd_new
    return replace(demand, disp_demand=dd, disp_elastic=dd_elastic, Rd=Rd,
                   Ts=Ts, mu_for_Rd=dd / delta_y)


def displacement_demand(
    spectrum: DesignSpectrum,
    stiffness: float,
    weight: float,
) -> DemandResult:
    """Compute the ESA displacement demand.

    Parameters
    ----------
    spectrum:
        The design response spectrum.
    stiffness:
        Effective (cracked) lateral stiffness of the cantilever, kip/in.
    weight:
        Tributary weight carried by the column, kip.
    """
    mass = weight / G_IN_S2
    period = 2.0 * math.pi * math.sqrt(mass / stiffness)
    sa = spectrum.Sa(period)
    dd = sa * G_IN_S2 * (period / (2.0 * math.pi)) ** 2
    return DemandResult(period=period, Sa=sa, disp_demand=dd, stiffness=stiffness, mass=mass)
