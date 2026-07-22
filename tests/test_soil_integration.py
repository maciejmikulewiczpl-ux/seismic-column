"""evaluate_column with fixity_source='soil' — the p-y integration."""
import pytest

from seismic_column.demand import DesignSpectrum
from seismic_column.geometry import Geometry
from seismic_column.sdc_capacity import evaluate_column
from seismic_column.section import CircularSection
from seismic_column.soil import SoilLayer, SoilProfile


def _sections():
    col = CircularSection(D=48, fc=4, cover=2, n_bars=24, long_bar_no=11,
                          spiral_bar_no=6, spiral_spacing=3)
    shaft = CircularSection(D=84, fc=4, cover=3, n_bars=40, long_bar_no=11,
                            spiral_bar_no=6, spiral_spacing=3.5)
    return col, shaft


def _profile(model="matlock_soft_clay", **kw):
    kw.setdefault("su_top_ksf", 1.5)
    kw.setdefault("eps50", 0.01)
    kw.setdefault("phi_deg", 36)
    kw.setdefault("k_pci", 90)
    return SoilProfile((SoilLayer.from_engineering(
        80, model, 120, submerged=True, **kw),))


def test_multiplier_path_unchanged():
    col, shaft = _sections()
    geom = Geometry(Hcol=22 * 12, D_shaft=84)
    spec = DesignSpectrum(Sds=1.0, Sd1=0.6)
    a = evaluate_column(col, shaft, geom, spec, 800, 800)   # default multiplier
    assert [round(b.fixity_depth) for b in a.bounds] == [252, 504]   # 3D, 6D
    assert all(b.soil_solution is None for b in a.bounds)


def test_soil_source_builds_soil_bounds():
    col, shaft = _sections()
    geom = Geometry(Hcol=22 * 12, D_shaft=84)
    spec = DesignSpectrum(Sds=1.0, Sd1=0.6)
    a = evaluate_column(col, shaft, geom, spec, 800, 800,
                        fixity_source="soil", soil_profile=_profile())
    assert len(a.bounds) == 2
    for b in a.bounds:
        assert b.soil_solution is not None and b.soil_solution.converged
        assert b.fixity_depth > 0                      # soil-derived Df_eq
        assert b.soil_solution.max_inground_moment > 0
        assert b.soil_label                            # "upper (stiff soil)" etc.
    # stiffer-soil bound gives shallower fixity than the soft-soil bound
    assert a.bounds[0].fixity_depth < a.bounds[1].fixity_depth
    # the equivalent multiplier lands in a physically reasonable range
    assert all(1.0 < b.multiplier < 12.0 for b in a.bounds)


def test_inground_shaft_design_present_and_deeper_than_interface():
    """Soil mode adds the overstrength in-ground shaft demand: an in-ground
    solution, the two design checks, and a max moment that can exceed (and sits
    below) the interface Mo."""
    col, shaft = _sections()
    geom = Geometry(Hcol=25 * 12, D_shaft=84)
    spec = DesignSpectrum(Sds=1.2, Sd1=0.7)
    # soft-over-dense profile → peak moment well below the interface
    prof = SoilProfile((
        SoilLayer.from_engineering(15, "matlock_soft_clay", 115, su_top_ksf=1.0,
                                   eps50=0.015, submerged=True),
        SoilLayer.from_engineering(85, "api_sand", 130, phi_deg=38, k_pci=120,
                                   submerged=True)))
    a = evaluate_column(col, shaft, geom, spec, 1500, 1500,
                        fixity_source="soil", soil_profile=prof,
                        shaft_embed_length=100 * 12)
    assert a.inground_solution is not None
    assert a.inground_moment > 0 and a.inground_shear > 0
    names = {c.name for c in a.checks}
    assert "Shaft flexure in-ground (p-y)" in names
    assert "Shaft shear in-ground (p-y)" in names
    # the max moment is below the interface (deeper), not at the top of shaft
    assert a.inground_solution.max_moment_depth > 0.0


def test_soil_report_prescribes_forces_and_py_detail():
    """The p-y report section is detailed like the others: prescribes the applied
    pile-head forces, shows the p-y development, and the in-ground shaft design."""
    from seismic_column.optimizer import ColumnDesign
    from seismic_column.batch import RowResult
    from seismic_column.report import column_report
    col, shaft = _sections()
    geom = Geometry(Hcol=25 * 12, D_shaft=84)
    a = evaluate_column(col, shaft, geom, DesignSpectrum(Sds=1.2, Sd1=0.7),
                        1500, 1500, fixity_source="soil",
                        soil_profile=_profile("api_sand"),
                        shaft_embed_length=100 * 12)
    cd = ColumnDesign(D=48, fc=4, cover=2, n_bars=24, long_bar_no=11,
                      spiral_bar_no=6, spiral_spacing=3)
    sd = ColumnDesign(D=84, fc=4, cover=3, n_bars=40, long_bar_no=11,
                      spiral_bar_no=6, spiral_spacing=3.5)
    t = column_report(RowResult("P1", cd, sd, a, a.passed, False, []))
    assert "Point-of-fixity source: nonlinear p-y" in t
    assert "Applied pile-head forces" in t
    assert "F_y" in t and "Mp / Hcol" in t             # stiffness solve force
    assert "Vo" in t and "Mo / Hcol" in t              # in-ground design force
    assert "p-y curve development" in t
    assert "In-ground shaft design" in t
    assert "Shaft flexure in-ground" in t
    # p-y is a code-sanctioned approach; default code here is Caltrans SDC 2.1
    assert "C6.2.5.3" in t
    # closed-form linear cross-check (Davisson / LRFD 10.7.3.13.4) is shown
    assert "Closed-form cross-check" in t and "10.7.3.13.4" in t
    assert "Why they differ" in t


def test_multiplier_report_states_assumed_source():
    from seismic_column.optimizer import ColumnDesign
    from seismic_column.batch import RowResult
    from seismic_column.report import column_report
    col, shaft = _sections()
    a = evaluate_column(col, shaft, Geometry(Hcol=22 * 12, D_shaft=84),
                        DesignSpectrum(Sds=1.0, Sd1=0.6), 800, 800)
    cd = ColumnDesign(D=48, fc=4, cover=2, n_bars=24, long_bar_no=11,
                      spiral_bar_no=6, spiral_spacing=3)
    sd = ColumnDesign(D=84, fc=4, cover=3, n_bars=40, long_bar_no=11,
                      spiral_bar_no=6, spiral_spacing=3.5)
    t = column_report(RowResult("P1", cd, sd, a, a.passed, False, []))
    assert "Point-of-fixity source: assumed multipliers" in t
    assert "mult · D_shaft" in t
    # estimated depth to fixity is code-accepted; note its linear-elastic caveat
    # (default code here is Caltrans SDC 2.1 → §6.2.6 + AASHTO-CA BDS 10.7.3.13.4)
    assert "6.2.6" in t and "10.7.3.13.4" in t


def test_aashto_report_cites_sgs_fmm_table():
    """Under AASHTO SGS provisions the fixity note cites the SGS FMM table, not
    the Caltrans clauses (code-specific references, no leakage)."""
    from seismic_column.optimizer import ColumnDesign
    from seismic_column.batch import RowResult
    from seismic_column.provisions import get_provisions
    from seismic_column.report import column_report
    col, shaft = _sections()
    a = evaluate_column(col, shaft, Geometry(Hcol=25 * 12, D_shaft=84),
                        DesignSpectrum(Sds=1.2, Sd1=0.7), 1500, 1500,
                        fixity_source="soil", soil_profile=_profile("api_sand"),
                        shaft_embed_length=100 * 12,
                        provisions=get_provisions("AASHTO SGS 3rd Ed."))
    cd = ColumnDesign(D=48, fc=4, cover=2, n_bars=24, long_bar_no=11,
                      spiral_bar_no=6, spiral_spacing=3)
    sd = ColumnDesign(D=84, fc=4, cover=3, n_bars=40, long_bar_no=11,
                      spiral_bar_no=6, spiral_spacing=3.5)
    t = column_report(RowResult("P1", cd, sd, a, a.passed, False, []))
    assert "Table 5.3.1-1" in t and "Foundation Modeling Method II" in t
    assert "C6.2.5.3" not in t          # Caltrans clause must not leak in


def test_multiplier_mode_has_no_inground():
    col, shaft = _sections()
    geom = Geometry(Hcol=22 * 12, D_shaft=84)
    spec = DesignSpectrum(Sds=1.0, Sd1=0.6)
    a = evaluate_column(col, shaft, geom, spec, 800, 800)
    assert a.inground_solution is None
    assert not any("in-ground" in c.name for c in a.checks)


def test_soil_source_requires_profile():
    col, shaft = _sections()
    geom = Geometry(Hcol=22 * 12, D_shaft=84)
    spec = DesignSpectrum(Sds=1.0, Sd1=0.6)
    with pytest.raises(ValueError, match="soil_profile"):
        evaluate_column(col, shaft, geom, spec, 800, 800, fixity_source="soil")


def test_invalid_fixity_source():
    col, shaft = _sections()
    geom = Geometry(Hcol=22 * 12, D_shaft=84)
    spec = DesignSpectrum(Sds=1.0, Sd1=0.6)
    with pytest.raises(ValueError, match="fixity_source"):
        evaluate_column(col, shaft, geom, spec, 800, 800, fixity_source="nope")


def test_full_soil_assessment_has_all_checks():
    col, shaft = _sections()
    geom = Geometry(Hcol=22 * 12, D_shaft=84)
    spec = DesignSpectrum(Sds=1.0, Sd1=0.6)
    a = evaluate_column(col, shaft, geom, spec, 800, 800,
                        fixity_source="soil", soil_profile=_profile("api_sand"))
    names = {c.name for c in a.checks}
    assert "Displacement capacity" in names
    assert "Shaft flexure (capacity protection)" in names
    assert a.governing_bound in a.bounds


def test_soil_presets_build_and_solve():
    """The bundled LPile-style strata presets build valid profiles, the 'Ignore'
    top layer carries zero p-y resistance, and evaluate_column solves stably."""
    from seismic_column.io_schema import (
        SOIL_PROFILE_PRESETS, build_soil_profile, load_soil_preset, GlobalConfig)
    col, shaft = _sections()
    df_eq = {}
    for name in SOIL_PROFILE_PRESETS:
        wt, layers = load_soil_preset(name)
        cfg = GlobalConfig(fixity_source="soil", water_table_ft=wt,
                           soil_layers=tuple(layers))
        prof = build_soil_profile(cfg)
        assert prof is not None and len(prof.layers) == len(layers)
        # 'Ignore' top layer (elastic k=0) gives zero lateral resistance,
        # but its weight still adds overburden to the layers below it.
        assert prof.p_of_y(2.5 * 12, 1.0, 84.0) == 0.0
        assert prof.sigma_v_eff(20 * 12) > 0.0
        a = evaluate_column(col, shaft, Geometry(Hcol=25 * 12, D_shaft=84),
                            DesignSpectrum(Sds=1.2, Sd1=0.7), 1500, 1500,
                            fixity_source="soil", soil_profile=prof,
                            shaft_embed_length=int(prof.depth))
        for b in a.bounds:
            assert b.soil_solution.stable and b.soil_solution.converged
        df_eq[name] = max(b.fixity_depth for b in a.bounds)
    # the high-water, liquefied-to-20ft B profile is softer → deeper fixity
    assert df_eq["SeaTac Piers B2–B18 (GWT 5 ft)"] > \
        df_eq["SeaTac Piers A8–A11 (GWT 10 ft)"]


def test_optimizer_sizes_shaft_for_inground_py_demand():
    """The optimiser escalates shaft steel until the p-y IN-GROUND flexure and
    shear checks pass — the demand depends on the shaft's own stiffness, so this
    exercises the size->re-solve->resize fixed-point iteration."""
    from seismic_column.optimizer import (
        ColumnDesign, OptimizeSpec, optimize_column)
    from seismic_column.io_schema import (
        load_soil_preset, build_soil_profile, GlobalConfig)
    from seismic_column.provisions import get_provisions
    prov = get_provisions("AASHTO SGS 3rd Ed.")
    wt, layers = load_soil_preset("SeaTac Piers A8–A11 (GWT 10 ft)")
    prof = build_soil_profile(GlobalConfig(fixity_source="soil",
                                           water_table_ft=wt,
                                           soil_layers=tuple(layers)))
    col = ColumnDesign(D=60, fc=5, cover=2, n_bars=30, long_bar_no=10,
                       spiral_bar_no=6, spiral_spacing=4, fce_floor=5.0)
    # deliberately light starting shaft so the in-ground checks start failing
    shaft = ColumnDesign(D=96, fc=5, cover=6, n_bars=30, long_bar_no=8,
                         spiral_bar_no=4, spiral_spacing=6, fce_floor=5.0)
    geom = Geometry(Hcol=20 * 12, D_shaft=96)
    spec = DesignSpectrum(Sds=1.0, Sd1=0.6)
    soil_kw = dict(fixity_source="soil", soil_profile=prof,
                   shaft_embed_length=int(prof.depth), soil_bounds=(2.0, 0.5))

    a0 = evaluate_column(col.section(), shaft.section(), geom, spec, 1500, 1500,
                         provisions=prov, **soil_kw)
    before = {c.name: c.passed for c in a0.checks if "in-ground" in c.name}
    assert before and not all(before.values())          # starts failing

    res = optimize_column(col, shaft, geom, spec, 1500, 1500,
                          spec=OptimizeSpec(variable={"longitudinal", "confinement"},
                                            priority=("longitudinal", "confinement")),
                          provisions=prov, **soil_kw)
    ig = [c for c in res.assessment.checks if "in-ground" in c.name]
    assert ig and all(c.passed for c in ig)             # sized to pass
    # the shaft steel was escalated above the (light) starting cage
    assert res.shaft.rho_l() > shaft.rho_l()


def test_pile_cache_reuse():
    """A repeated identical evaluation reuses the cached pile solve."""
    from seismic_column import sdc_capacity as sc
    sc._PILE_CACHE.clear()
    col, shaft = _sections()
    geom = Geometry(Hcol=22 * 12, D_shaft=84)
    spec = DesignSpectrum(Sds=1.0, Sd1=0.6)
    prof = _profile()
    evaluate_column(col, shaft, geom, spec, 800, 800,
                    fixity_source="soil", soil_profile=prof)
    n1 = len(sc._PILE_CACHE)
    assert n1 == 4          # 2 bounds × (F_y stiffness + Vo in-ground) solves
    evaluate_column(col, shaft, geom, spec, 800, 800,
                    fixity_source="soil", soil_profile=prof)
    assert len(sc._PILE_CACHE) == n1                   # no new solves
