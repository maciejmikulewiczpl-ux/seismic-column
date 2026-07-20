from seismic_column.moment_curvature import moment_curvature
from seismic_column.section import CircularSection


def make_section():
    return CircularSection(
        D=48, fc=4, cover=2, n_bars=20, long_bar_no=10,
        spiral_bar_no=5, spiral_spacing=3,
    )


def test_moment_curvature_monotonic_points():
    s = make_section()
    mc = moment_curvature(s, axial=800)
    assert mc.phi_u > mc.phi_y > 0
    assert mc.Mp > 0
    assert mc.phi_u / mc.phi_y > 3.0  # ductile
    assert mc.failure_mode in ("concrete", "steel")


def test_effective_stiffness_in_cracked_range():
    s = make_section()
    mc = moment_curvature(s, axial=800)
    Ec = s.confined.Ec
    Ig = s.gross_inertia()
    ratio = (mc.EI_eff / Ec) / Ig
    assert 0.15 < ratio < 0.7


def test_higher_axial_increases_moment():
    s = make_section()
    low = moment_curvature(s, axial=400)
    high = moment_curvature(s, axial=1200)
    assert high.Mp > low.Mp


def test_more_longitudinal_steel_increases_capacity():
    light = CircularSection(D=48, fc=4, cover=2, n_bars=12, long_bar_no=8,
                            spiral_bar_no=5, spiral_spacing=3)
    heavy = CircularSection(D=48, fc=4, cover=2, n_bars=28, long_bar_no=10,
                            spiral_bar_no=5, spiral_spacing=3)
    assert moment_curvature(heavy, 800).Mp > moment_curvature(light, 800).Mp
