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
from dataclasses import dataclass, field

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
    disp_demand: float    # displacement demand Dd, in
    stiffness: float      # lateral stiffness used, kip/in
    mass: float           # tributary mass, kip*s^2/in


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
