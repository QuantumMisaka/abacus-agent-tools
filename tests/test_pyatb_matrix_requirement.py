"""Matrix-presence gating for PyATB property post-processing."""

from __future__ import annotations

from pathlib import Path

import pytest

from abacusagent.modules.util.comm import has_pyatb_matrix_files
from abacusagent.modules.util import pyatb as pyatb_module


def _scf_dir(tmp_path: Path, *, hs: bool, rR: bool) -> Path:
    work = tmp_path / "scf"
    out = work / "OUT.ABACUS"
    out.mkdir(parents=True)
    (work / "INPUT").write_text(
        "INPUT_PARAMETERS\nbasis_type lcao\ncalculation scf\nsuffix ABACUS\n",
        encoding="utf-8",
    )
    if hs:
        (out / "data-HR-sparse_SPIN0.csr").write_text("hr", encoding="utf-8")
        (out / "data-SR-sparse_SPIN0.csr").write_text("sr", encoding="utf-8")
    if rR:
        (out / "data-rR-sparse.csr").write_text("rr", encoding="utf-8")
    return work


def test_hs_requirement_accepts_hamiltonian_only(tmp_path):
    work = _scf_dir(tmp_path, hs=True, rR=False)
    assert has_pyatb_matrix_files(work, required="hs") is True
    assert has_pyatb_matrix_files(work) is False          # historical default
    assert has_pyatb_matrix_files(work, required="hsr") is False


def test_hsr_requirement_needs_position_matrix(tmp_path):
    work = _scf_dir(tmp_path, hs=True, rR=True)
    assert has_pyatb_matrix_files(work, required="hs") is True
    assert has_pyatb_matrix_files(work, required="hsr") is True


def test_missing_hs_is_never_reusable(tmp_path):
    work = _scf_dir(tmp_path, hs=False, rR=True)
    assert has_pyatb_matrix_files(work, required="hs") is False
    assert has_pyatb_matrix_files(work, required="hsr") is False


def test_invalid_requirement_is_rejected(tmp_path):
    work = _scf_dir(tmp_path, hs=True, rR=True)
    with pytest.raises(ValueError):
        has_pyatb_matrix_files(work, required="h")


def test_property_scf_skips_rerun_when_only_hs_is_required(tmp_path, monkeypatch):
    """Regression: band post-processing re-ran an 8h SCF for a missing rR file."""
    work = _scf_dir(tmp_path, hs=True, rR=False)
    (work / "STRU").write_text("ATOMIC_SPECIES\nSb 121.76 Sb.upf\n", encoding="utf-8")

    calls: list[str] = []
    monkeypatch.setattr(
        pyatb_module, "run_abacus", lambda *a, **k: calls.append("run_abacus")
    )

    result = pyatb_module.property_calculation_scf(
        work, mode="pyatb", required_matrix="hs"
    )

    assert calls == []
    assert Path(result["work_path"]) == work.resolve()
