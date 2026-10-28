"""Explicit work-parent allocation for purpose-named property stages."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest


def _write_inputs(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "INPUT").write_text(
        "INPUT_PARAMETERS\nnspin 1\nbasis_type lcao\ncalculation scf\n"
        "pseudo_dir ./\ngamma_only 1\n",
        encoding="utf-8",
    )
    (path / "STRU").write_text(
        "ATOMIC_SPECIES\nH 1.008 H.upf\n\n"
        "LATTICE_CONSTANT\n1.0\n\n"
        "LATTICE_VECTORS\n10 0 0\n0 10 0\n0 0 10\n\n"
        "ATOMIC_POSITIONS\nDirect\nH\n0.0\n1\n0.1 0.1 0.1\n",
        encoding="utf-8",
    )
    (path / "KPT").write_text(
        "K_POINTS\n0\nGamma\n1 1 1 0 0 0\n", encoding="utf-8"
    )
    (path / "H.upf").write_text("fixture pseudopotential\n", encoding="utf-8")
    (path / "H.orb").write_text("fixture orbital\n", encoding="utf-8")
    return path


def test_generate_stage_path_creates_purpose_name_under_parent(tmp_path: Path):
    from abacusagent.modules.util.work_path import generate_stage_path

    parent = tmp_path / "GaN" / "PBE_dos_run"
    stage = generate_stage_path("scf", parent=parent)

    assert Path(stage) == parent / "scf"
    assert (parent / "scf").is_dir()


def test_dos_scf_stage_uses_work_parent(tmp_path: Path, monkeypatch):
    from abacusagent.modules.submodules import dos

    inputs = _write_inputs(tmp_path / "inputs")
    parent = tmp_path / "GaN" / "PBE_dos_run"
    monkeypatch.setattr(dos, "has_chgfile", lambda _path: False)
    monkeypatch.setattr(
        dos,
        "link_abacusjob",
        lambda *, src, dst, **_kwargs: shutil.copytree(src, dst, dirs_exist_ok=True),
    )
    monkeypatch.setattr(dos, "run_abacus", lambda *_a, **_k: None)
    monkeypatch.setattr(
        dos,
        "RESULT",
        lambda **_kwargs: {
            "normal_end": True,
            "scf_steps": 1,
            "converge": True,
            "energy": -1.0,
        },
    )

    result = dos.abacus_dos_run_scf(inputs, note="legacy", work_parent=parent)

    assert result["scf_work_path"] == parent.resolve() / "scf"
    assert (parent / "scf" / "INPUT").is_file()


def test_band_stage_inputs_use_work_parent(tmp_path: Path):
    from abacusagent.modules.submodules import band

    inputs = _write_inputs(tmp_path / "inputs")
    parent = tmp_path / "GaN" / "PBE_band_run"

    staged = band._stage_band_inputs(
        inputs,
        band.ReadInput(inputs / "INPUT"),
        note="legacy",
        work_parent=parent,
    )

    assert Path(staged) == (parent / "inputs").resolve()
    assert (parent / "inputs" / "INPUT").is_file()


def test_property_scf_uses_work_parent(tmp_path: Path, monkeypatch):
    from abacusagent.modules.util import pyatb

    inputs = _write_inputs(tmp_path / "inputs")
    parent = tmp_path / "GaN" / "PBE_band_run"
    monkeypatch.setattr(
        pyatb,
        "link_abacusjob",
        lambda *, src, dst, **_kwargs: shutil.copytree(src, dst, dirs_exist_ok=True),
    )
    monkeypatch.setattr(pyatb, "run_abacus", lambda *_a, **_k: None)
    monkeypatch.setattr(
        pyatb,
        "RESULT",
        lambda **_kwargs: {
            "normal_end": True,
            "scf_steps": 1,
            "converge": True,
            "energies": [-1.0],
        },
    )

    result = pyatb.property_calculation_scf(
        inputs, mode="pyatb", always_run=True, note="band", work_parent=parent
    )

    assert result["work_path"] == parent.resolve() / "band"
    assert (parent / "band" / "INPUT").is_file()
