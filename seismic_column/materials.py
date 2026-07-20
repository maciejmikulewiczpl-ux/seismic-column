"""Material models for confined/unconfined concrete and reinforcing steel.

All models return stress in ksi for a given strain (compression positive for
concrete, tension positive for steel). Implementations follow:

* Concrete : Mander, Priestley & Park (1988) confined-concrete model, specialised
  for circular sections with spiral or circular-hoop transverse reinforcement.
* Steel    : Caltrans SDC 2.0 reinforcing-steel model for ASTM A706 bars
  (elastic / yield-plateau / strain-hardening parabola) using *expected*
  material properties.

References
---------
Mander, J.B., Priestley, M.J.N., Park, R. (1988). "Theoretical Stress-Strain
    Model for Confined Concrete." ASCE J. Struct. Eng. 114(8).
Caltrans Seismic Design Criteria, Version 2.0 (2019), Section 3.2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


# ---------------------------------------------------------------------------
# Reinforcing steel bar catalogue (US #-bars)
# ---------------------------------------------------------------------------
# nominal diameter (in) and area (in^2) keyed by bar designation number.
BAR_PROPERTIES: dict[int, dict[str, float]] = {
    3: {"db": 0.375, "area": 0.11},
    4: {"db": 0.500, "area": 0.20},
    5: {"db": 0.625, "area": 0.31},
    6: {"db": 0.750, "area": 0.44},
    7: {"db": 0.875, "area": 0.60},
    8: {"db": 1.000, "area": 0.79},
    9: {"db": 1.128, "area": 1.00},
    10: {"db": 1.270, "area": 1.27},
    11: {"db": 1.410, "area": 1.56},
    14: {"db": 1.693, "area": 2.25},
    18: {"db": 2.257, "area": 4.00},
}


def bar_diameter(bar_no: int) -> float:
    """Nominal bar diameter, in."""
    return BAR_PROPERTIES[bar_no]["db"]


def bar_area(bar_no: int) -> float:
    """Nominal bar area, in^2."""
    return BAR_PROPERTIES[bar_no]["area"]


# ---------------------------------------------------------------------------
# Reinforcing steel model (Caltrans SDC 2.0, ASTM A706 expected properties)
# ---------------------------------------------------------------------------
@dataclass
class ReinforcingSteel:
    """ASTM A706 reinforcing-steel stress-strain model (expected properties).

    Parameters
    ----------
    bar_no:
        Bar designation used to select onset-of-hardening and ultimate strains
        from the SDC tables.
    fye, fue:
        Expected yield / ultimate tensile stress, ksi.
    Es:
        Elastic modulus, ksi.
    """

    bar_no: int = 10
    fye: float = 68.0
    fue: float = 95.0
    Es: float = 29000.0
    eps_sh: float = field(init=False)
    eps_su: float = field(init=False)
    eps_su_r: float = field(init=False)  # reduced ultimate tensile strain
    eps_ye: float = field(init=False)

    def __post_init__(self) -> None:
        self.eps_ye = self.fye / self.Es
        self.eps_sh = self._onset_hardening_strain(self.bar_no)
        self.eps_su, self.eps_su_r = self._ultimate_strains(self.bar_no)

    @staticmethod
    def _onset_hardening_strain(bar_no: int) -> float:
        """Onset of strain hardening eps_sh per SDC 2.0 Table (by bar size)."""
        if bar_no <= 8:
            return 0.0150
        if bar_no == 9:
            return 0.0125
        if bar_no in (10, 11):
            return 0.0115
        if bar_no == 14:
            return 0.0075
        return 0.0050  # #18

    @staticmethod
    def _ultimate_strains(bar_no: int) -> tuple[float, float]:
        """Return (eps_su, reduced eps_su^R) per SDC 2.0 (by bar size)."""
        if bar_no <= 10:
            return 0.120, 0.090
        return 0.090, 0.060  # #11 and larger

    def stress(self, eps: float) -> float:
        """Steel stress (ksi) for tensile-positive strain ``eps``.

        Uses the reduced ultimate tensile strain as the fracture limit; beyond
        it the stress is returned as 0 (bar fractured).
        """
        sign = 1.0 if eps >= 0.0 else -1.0
        e = abs(eps)
        if e <= self.eps_ye:
            return sign * self.Es * e
        if e <= self.eps_sh:
            return sign * self.fye
        if e <= self.eps_su:
            # strain-hardening parabola (King/Caltrans form)
            ratio = (self.eps_su - e) / (self.eps_su - self.eps_sh)
            return sign * (self.fue - (self.fue - self.fye) * ratio ** 2)
        return 0.0  # fractured


# ---------------------------------------------------------------------------
# Concrete modulus
# ---------------------------------------------------------------------------
def concrete_modulus(fc: float) -> float:
    """Concrete elastic modulus Ec (ksi) for ``fc`` in ksi.

    Ec = 57000 * sqrt(f'c[psi]) psi  ==  57 * sqrt(1000 * f'c[ksi]) ksi
    """
    return 57.0 * math.sqrt(1000.0 * fc)


# ---------------------------------------------------------------------------
# Unconfined concrete (cover) - Mander model with linear descending branch
# ---------------------------------------------------------------------------
@dataclass
class UnconfinedConcrete:
    """Unconfined concrete stress-strain (compression positive)."""

    fc: float = 4.0
    eps_c0: float = 0.002
    eps_spall: float = 0.005
    Ec: float = field(init=False)
    Esec: float = field(init=False)
    r: float = field(init=False)

    def __post_init__(self) -> None:
        self.Ec = concrete_modulus(self.fc)
        self.Esec = self.fc / self.eps_c0
        self.r = self.Ec / (self.Ec - self.Esec)

    def stress(self, eps: float) -> float:
        """Compressive stress (ksi) for compression-positive strain ``eps``."""
        if eps <= 0.0:
            return 0.0  # no tension capacity
        if eps <= 2.0 * self.eps_c0:
            x = eps / self.eps_c0
            return self.fc * x * self.r / (self.r - 1.0 + x ** self.r)
        if eps <= self.eps_spall:
            # linear descent from stress at 2*eps_c0 to zero at spalling strain
            x = 2.0
            f_2e = self.fc * x * self.r / (self.r - 1.0 + x ** self.r)
            return f_2e * (self.eps_spall - eps) / (self.eps_spall - 2.0 * self.eps_c0)
        return 0.0  # spalled


# ---------------------------------------------------------------------------
# Confined concrete (Mander) for circular sections
# ---------------------------------------------------------------------------
@dataclass
class ConfinedConcrete:
    """Mander confined-concrete model for a circular core.

    Parameters
    ----------
    fc:
        Unconfined concrete strength f'c, ksi.
    D:
        Overall section diameter, in.
    cover:
        Clear cover to the transverse (spiral/hoop) steel, in.
    spiral_bar_no:
        Transverse bar designation.
    spacing:
        Centre-to-centre pitch/spacing of transverse steel, in.
    fyh:
        Expected yield stress of transverse steel, ksi.
    eps_su_h:
        Ultimate strain of transverse steel (for eps_cu), typ. 0.09-0.12.
    rho_long:
        Longitudinal steel ratio relative to the *core* area (Ast/Acore).
    hoops:
        If ``True`` treat as discrete circular hoops, else continuous spiral.
    spiral_bundle:
        Number of transverse bars bundled together at each layer (e.g. 2 for a
        bundled #4 spiral).  Multiplies the effective transverse steel area.
    """

    fc: float = 4.0
    D: float = 48.0
    cover: float = 2.0
    spiral_bar_no: int = 5
    spacing: float = 3.0
    fyh: float = 68.0
    eps_su_h: float = 0.09
    rho_long: float = 0.02
    hoops: bool = False
    spiral_bundle: int = 1
    eps_c0: float = 0.002

    # computed
    ds: float = field(init=False)          # dia. of confined core to spiral centreline
    rho_s: float = field(init=False)       # volumetric transverse steel ratio
    ke: float = field(init=False)          # confinement effectiveness coefficient
    fl_eff: float = field(init=False)      # effective lateral confining pressure
    fcc: float = field(init=False)         # confined strength
    eps_cc: float = field(init=False)      # strain at peak confined stress
    eps_cu: float = field(init=False)      # ultimate confined compressive strain
    Ec: float = field(init=False)
    Esec: float = field(init=False)
    r: float = field(init=False)

    def __post_init__(self) -> None:
        dsp = bar_diameter(self.spiral_bar_no)
        asp = bar_area(self.spiral_bar_no) * self.spiral_bundle
        # core diameter measured to centreline of transverse steel
        self.ds = self.D - 2.0 * self.cover - dsp
        # clear spacing between successive transverse bars
        s_clear = self.spacing - dsp
        # volumetric ratio of transverse steel: rho_s = 4*Asp/(ds*s)
        self.rho_s = 4.0 * asp / (self.ds * self.spacing)
        # confinement effectiveness coefficient
        denom = 1.0 - self.rho_long
        base = 1.0 - s_clear / (2.0 * self.ds)
        self.ke = (base ** 2 if self.hoops else base) / denom
        # effective lateral confining pressure (circular): fl' = 0.5 ke rho_s fyh
        self.fl_eff = 0.5 * self.ke * self.rho_s * self.fyh
        # confined strength (Mander)
        ratio = self.fl_eff / self.fc
        self.fcc = self.fc * (-1.254 + 2.254 * math.sqrt(1.0 + 7.94 * ratio) - 2.0 * ratio)
        # strain at peak confined stress
        self.eps_cc = self.eps_c0 * (1.0 + 5.0 * (self.fcc / self.fc - 1.0))
        # ultimate confined compressive strain (Priestley/Mander energy balance form)
        self.eps_cu = 0.004 + 1.4 * self.rho_s * self.fyh * self.eps_su_h / self.fcc
        # stress-strain shape parameters
        self.Ec = concrete_modulus(self.fc)
        self.Esec = self.fcc / self.eps_cc
        self.r = self.Ec / (self.Ec - self.Esec)

    def stress(self, eps: float) -> float:
        """Confined compressive stress (ksi) for compression-positive strain."""
        if eps <= 0.0:
            return 0.0
        x = eps / self.eps_cc
        f = self.fcc * x * self.r / (self.r - 1.0 + x ** self.r)
        return f
