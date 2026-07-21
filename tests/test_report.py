"""Tests for the Markdown calculation report.

These guard structure and internal consistency rather than exact wording:
every section renders, every check is shown, the equations carry code
references, and the moment-curvature section substitutes the *expected*
strength f'ce (the bug that previously showed nominal f'c in the arithmetic
but f'ce in the result)."""
import pytest

from seismic_column.batch import run_batch
from seismic_column.io_schema import GlobalConfig, default_dataframe
from seismic_column.report import column_report

CODES = ["SDC 2.1", "AASHTO SGS 3rd Ed."]
SECTION_HEADERS = [f"### {i} ·" for i in range(1, 10)]  # sections 1..9


def _reports(code, optimize=False, n=3):
    _, results = run_batch(default_dataframe(n), GlobalConfig(code=code,
                                                              optimize=optimize))
    return [(rr, column_report(rr)) for rr in results]


@pytest.mark.parametrize("code", CODES)
@pytest.mark.parametrize("optimize", [False, True])
def test_report_renders_every_section(code, optimize):
    for rr, text in _reports(code, optimize):
        assert text.startswith("# Seismic Column Report")
        assert "## Detailed calculations" in text
        for header in SECTION_HEADERS:
            assert header in text, f"missing {header} for {code}"


@pytest.mark.parametrize("code", CODES)
def test_report_has_no_nan_or_none(code):
    for _, text in _reports(code):
        low = text.lower()
        assert "nan" not in low
        assert "none" not in low
        assert "{" not in text and "}" not in text  # no unfilled f-string braces


@pytest.mark.parametrize("code", CODES)
def test_report_lists_every_check(code):
    for rr, text in _reports(code):
        for c in rr.assessment.checks:
            assert c.name in text, f"check '{c.name}' not shown in report"


@pytest.mark.parametrize("code", CODES)
def test_report_uses_expected_strength_in_mphi(code):
    """Section 1 Ec/f'cc must substitute f'ce, not nominal f'c (regression)."""
    for rr, text in _reports(code):
        fce = rr.design.section().fce
        # the Ec substitution line must carry the f'ce value under the radical
        assert f"57000·√{fce * 1000:.0f}" in text
        # and f'ce itself is derived as factor*f'c
        assert "f'ce" in text


def test_report_references_are_code_specific():
    # Caltrans SDC 2.1 clause numbers
    _, ct = _reports("SDC 2.1")[0]
    assert "SDC 5.3.7.2" in ct           # concrete shear
    assert "Table 5.3.8.2-1" in ct       # min transverse lookup
    assert "5.3.3" in ct                 # axial load ratio
    assert "SGS" not in ct               # no AASHTO refs leaking in
    # AASHTO SGS clause numbers
    _, aa = _reports("AASHTO SGS 3rd Ed.")[0]
    assert "SGS 8.6.2" in aa             # concrete shear
    assert "8.6.5" in aa                 # min transverse floor
    assert "4.3.3" in aa                 # short-period magnification
    assert "SDC 5.3" not in aa           # no Caltrans refs leaking in


@pytest.mark.parametrize("code", CODES)
def test_report_shows_symbolic_then_numeric(code):
    """Detailed calcs should present bolded results (the '= **...**' pattern)
    and clause references in brackets."""
    for _, text in _reports(code):
        detail = text[text.index("## Detailed calculations"):]
        assert detail.count("= **") >= 15   # many substituted → result lines
        assert detail.count("*[") >= 10      # many bracketed code references


@pytest.mark.parametrize("code", CODES)
def test_report_pass_fail_matches_assessment(code):
    """Every failing check must be flagged NG somewhere in the report, and a
    fully-passing column must not contain an NG marker."""
    for rr, text in _reports(code, optimize=True):
        if rr.feasible:
            assert "NG" not in text
        else:
            assert "NG" in text
