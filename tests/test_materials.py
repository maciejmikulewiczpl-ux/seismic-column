import math

from seismic_column.materials import (
    ConfinedConcrete,
    ReinforcingSteel,
    UnconfinedConcrete,
    concrete_modulus,
)


def test_concrete_modulus_positive():
    assert concrete_modulus(4.0) > 3000.0


def test_confined_strength_exceeds_unconfined():
    c = ConfinedConcrete(fc=4.0, D=48.0, cover=2.0, spiral_bar_no=5, spacing=3.0)
    assert c.fcc > c.fc
    assert 0.0 < c.ke <= 1.0
    assert c.eps_cu > 0.004
    assert c.eps_cc > c.eps_c0


def test_confinement_increases_with_more_steel():
    loose = ConfinedConcrete(fc=4.0, spacing=6.0, spiral_bar_no=4)
    tight = ConfinedConcrete(fc=4.0, spacing=2.0, spiral_bar_no=6)
    assert tight.fcc > loose.fcc
    assert tight.eps_cu > loose.eps_cu


def test_unconfined_peak_and_spalling():
    u = UnconfinedConcrete(fc=4.0)
    assert abs(u.stress(u.eps_c0) - u.fc) < 0.2
    assert u.stress(0.0) == 0.0
    assert u.stress(0.01) == 0.0  # beyond spalling strain


def test_steel_elastic_and_yield():
    s = ReinforcingSteel(bar_no=10, fye=68.0)
    assert abs(s.stress(0.5 * s.eps_ye) - s.Es * 0.5 * s.eps_ye) < 1e-6
    assert abs(s.stress(s.eps_ye) - s.fye) < 1e-6
    # yield plateau
    assert abs(s.stress(s.eps_sh) - s.fye) < 1e-6
    # curve extends to eps_su (reduced strain is only the M-phi limit state)
    assert s.stress(0.5 * (s.eps_sh + s.eps_su)) > s.fye
    # fracture beyond ultimate tensile strain
    assert s.stress(s.eps_su + 0.01) == 0.0
    # odd function
    assert abs(s.stress(-0.001) + s.stress(0.001)) < 1e-9
