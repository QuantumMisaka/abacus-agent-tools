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


class _StageSentinel(RuntimeError):
    pass


def _run_stage_under_parent(monkeypatch, module, call, parent) -> Path:
    """Run ``call`` with a recording link step and stop right after allocation."""

    recorded: dict[str, Path] = {}

    def fake_link(*, src, dst, copy_files=(), **_kwargs):
        recorded.setdefault("dst", Path(dst))
        Path(dst).mkdir(parents=True, exist_ok=True)
        for name in copy_files:
            source = Path(src) / name
            if source.is_file():
                shutil.copy2(source, Path(dst) / name)

    def fake_run(*_args, **_kwargs):
        raise _StageSentinel("allocated")

    monkeypatch.setattr(module, "link_abacusjob", fake_link)
    monkeypatch.setattr(module, "run_abacus", fake_run)
    # These property backends convert failures into a ``message`` payload, so
    # the recorded allocation is the observable proof of placement.
    try:
        call()
    except _StageSentinel:
        pass
    return recorded["dst"]


@pytest.mark.parametrize(
    ("function", "extra_kwargs"),
    [
        ("abacus_cal_elf", {}),
        ("abacus_cal_charge_density_difference", {"subsys1_atom_index": [1]}),
        ("abacus_cal_spin_density", {}),
    ],
)
def test_cube_property_stages_use_work_parent(tmp_path, monkeypatch, function, extra_kwargs):
    from abacusagent.modules.submodules import cube

    inputs = _write_inputs(tmp_path / "inputs")
    parent = tmp_path / "GaN" / "PBE_elf_run"
    monkeypatch.setattr(cube, "check_abacus_inputs", lambda _p: (True, ""))
    call = lambda: getattr(cube, function)(
        inputs, note="stage", work_parent=parent, **extra_kwargs
    )

    _run_stage_under_parent(monkeypatch, cube, call, parent)

    stages = [path for path in parent.iterdir() if path.is_dir()]
    assert [path.name for path in stages] == ["stage"]


def test_work_function_stage_uses_work_parent(tmp_path, monkeypatch):
    from abacusagent.modules.submodules import work_function

    inputs = _write_inputs(tmp_path / "inputs")
    parent = tmp_path / "GaN" / "PBE_work_function_run"
    monkeypatch.setattr(work_function, "check_abacus_inputs", lambda _p: (True, ""))
    call = lambda: work_function.abacus_cal_work_function(
        inputs, vacuum_direction="c", note="stage", work_parent=parent
    )

    allocated = _run_stage_under_parent(monkeypatch, work_function, call, parent)

    assert allocated.parent == parent.resolve()
    assert allocated.name.startswith("stage")


def test_bader_charge_density_stage_uses_work_parent(tmp_path, monkeypatch):
    from abacusagent.modules.submodules import bader

    inputs = _write_inputs(tmp_path / "inputs")
    parent = tmp_path / "GaN" / "PBE_bader_run"
    call = lambda: bader.calculate_charge_densities_with_abacus(
        inputs, note="stage", work_parent=parent
    )

    allocated = _run_stage_under_parent(monkeypatch, bader, call, parent)

    assert allocated.parent == parent.resolve()
    assert allocated.name.startswith("stage")


def test_jdos_forwards_work_parent_to_property_scf(tmp_path, monkeypatch):
    from abacusagent.modules.submodules import jdos_pyatb

    inputs = _write_inputs(tmp_path / "inputs")
    parent = tmp_path / "GaN" / "PBE_jdos_run"
    captured: dict[str, object] = {}

    def fake_property_scf(path, *, mode, note, work_parent):
        captured.update(
            {"path": path, "mode": mode, "note": note, "work_parent": work_parent}
        )
        raise _StageSentinel("allocated")

    monkeypatch.setattr(jdos_pyatb, "property_calculation_scf", fake_property_scf)

    with pytest.raises(_StageSentinel):
        jdos_pyatb.pyatb_calculate_jdos(inputs, note="stage", work_parent=parent)

    assert captured == {
        "path": inputs,
        "mode": "pyatb",
        "note": "stage",
        "work_parent": parent,
    }


def test_allocate_work_path_without_note_keeps_backend_fallback_name(tmp_path, monkeypatch):
    """note=None 时保持历史命名：目录名来自后端函数，而不是分配器。"""

    from abacusagent.modules.util.work_path import allocate_work_path

    monkeypatch.chdir(tmp_path)

    def abacus_cal_elf_stage():
        return allocate_work_path()

    def calculate_charge_density_stage():
        return allocate_work_path()

    first = abacus_cal_elf_stage()
    second = calculate_charge_density_stage()

    assert first.name == "abacus_cal_elf_stage"
    assert second.name == "calculate_charge_density_stage"
    assert first.parent == second.parent == tmp_path.resolve()


def test_allocate_work_path_without_note_still_uses_parent(tmp_path):
    from abacusagent.modules.util.work_path import allocate_work_path

    parent = tmp_path / "GaN"

    def work_function_stage():
        return allocate_work_path(work_parent=parent)

    allocated = work_function_stage()

    assert allocated.parent == parent.resolve()
    assert allocated.name == "work_function_stage"
