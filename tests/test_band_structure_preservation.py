"""Regression tests for band-cell and magnetic-seed preservation."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import seekpath
from ase import Atoms

from abacustest.lib_prepare.abacus import AbacusStru, ReadInput
from abacusagent.modules.submodules import band
from abacusagent.modules.util import pyatb as pyatb_module


NI_O_STRU = (
    Path(__file__).resolve().parents[1] / "tests" / "abacus" / "STRU_NiO_fixatom"
)
SI_INPUTS = Path(__file__).resolve().parents[1] / "tests" / "abacus" / "Si-sp"


def _write_inputs(path: Path) -> Path:
    path.mkdir()
    (path / "INPUT").write_text(
        "INPUT_PARAMETERS\n"
        "basis_type pw\n"
        "calculation scf\n"
        "suffix ABACUS\n",
        encoding="utf-8",
    )
    shutil.copy2(NI_O_STRU, path / "STRU")
    (path / "KPT").write_text(
        "K_POINTS\n0\nGamma\n1 1 1 0 0 0\n", encoding="utf-8"
    )
    return path


def _write_custom_inputs(path: Path) -> Path:
    path.mkdir()
    (path / "INPUT").write_text(
        "INPUT_PARAMETERS\n"
        "basis_type pw\n"
        "calculation scf\n"
        "suffix ABACUS\n"
        "stru_file magnetic.stru\n"
        "kpoint_file initial.kpt\n",
        encoding="utf-8",
    )
    shutil.copy2(NI_O_STRU, path / "magnetic.stru")
    (path / "initial.kpt").write_text(
        "K_POINTS\n0\nGamma\n1 1 1 0 0 0\n", encoding="utf-8"
    )
    return path


def _seekpath_orig_cell_points(stru: AbacusStru) -> dict[str, list[float]]:
    labels = stru.get_label()
    label_types = {label: idx for idx, label in enumerate(dict.fromkeys(labels))}
    result = seekpath.get_path_orig_cell(
        (
            stru.get_cell(bohr=True),
            stru.get_coord(bohr=True, direct=True),
            [label_types[label] for label in labels],
        )
    )
    points = dict(result["point_coords"])
    if "GAMMA" in points and "G" not in points:
        points["G"] = points.pop("GAMMA")
    return points


def test_band_auto_path_stages_magnetic_cell_and_keeps_upstream_stru(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path / "inputs")
    source_stru = (inputs / "STRU").read_bytes()
    work_path = tmp_path / "band-work"
    work_path.mkdir()
    seen = {}

    def fake_scf(abacus_inputs_dir, mode, always_run=False, required_matrix="hsr", note=None):
        seen.update(
            inputs_dir=Path(abacus_inputs_dir),
            mode=mode,
            always_run=always_run,
            required_matrix=required_matrix,
            note=note,
        )
        return {"work_path": work_path, "mode": "nscf"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(band, "property_calculation_scf", fake_scf)
    monkeypatch.setattr(band, "run_abacus", lambda _path: None)
    monkeypatch.setattr(
        band,
        "abacus_plot_band_nscf",
        lambda *_args: {"band_gap": 0.1, "band_picture": work_path / "band.png"},
    )

    result = band.abacus_cal_band(inputs, mode="nscf", note="NiO-band")

    assert result["band_calc_dir"] == work_path.absolute()
    assert seen["inputs_dir"] != inputs
    assert (inputs / "STRU").read_bytes() == source_stru
    assert not (inputs / "KPT_band").exists()

    staged = AbacusStru.ReadStru(seen["inputs_dir"] / "STRU")
    source = AbacusStru.ReadStru(inputs / "STRU")
    assert staged.get_natoms() == source.get_natoms() == 4
    assert staged.get_atommag() == [2.0, -2.0, 0.0, 0.0]
    assert np.allclose(np.asarray(staged.get_cell()), np.asarray(source.get_cell()))

    # The generated labels/coordinates must use the reciprocal basis of the
    # preserved user cell, as returned by SeekPath's orig-cell route.
    atoms = Atoms(
        symbols=staged.get_label(),
        cell=staged.get_cell(),
        scaled_positions=staged.get_coord(direct=True),
        pbc=True,
    )
    assert len(atoms) == 4
    assert np.isclose(atoms.get_volume(), source.get_volume())
    kpt_band = (seen["inputs_dir"] / "KPT_band").read_text(encoding="utf-8")
    expected_points = _seekpath_orig_cell_points(source)
    for line in kpt_band.splitlines():
        fields = line.split()
        if "#" not in line or len(fields) < 4:
            continue
        label = line.split("#", 1)[1].strip()
        assert label in expected_points
        assert np.allclose(
            np.asarray(fields[:3], dtype=float), np.asarray(expected_points[label])
        )

    assert "K_POINTS" in kpt_band
    assert "#G" in kpt_band


def test_band_explicit_path_bypasses_seekpath_and_normalizes_custom_inputs(
    tmp_path, monkeypatch
):
    inputs = _write_custom_inputs(tmp_path / "inputs")
    source_stru = (inputs / "magnetic.stru").read_bytes()
    source_kpt = (inputs / "initial.kpt").read_bytes()
    work_path = tmp_path / "band-work"
    work_path.mkdir()
    seen = {}

    def fail_if_seekpath_is_used(*_args, **_kwargs):
        raise AssertionError("explicit kpath must not read/generate a SeekPath route")

    monkeypatch.setattr(band.AbacusStru, "ReadStru", fail_if_seekpath_is_used)

    def fake_scf(abacus_inputs_dir, mode, always_run=False, required_matrix="hsr", note=None):
        seen.update(
            inputs_dir=Path(abacus_inputs_dir),
            mode=mode,
            always_run=always_run,
            required_matrix=required_matrix,
            note=note,
        )
        return {"work_path": work_path, "mode": "nscf"}

    monkeypatch.setattr(band, "property_calculation_scf", fake_scf)
    monkeypatch.setattr(band, "run_abacus", lambda _path: None)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        band,
        "abacus_plot_band_nscf",
        lambda *_args: {"band_gap": 0.2, "band_picture": work_path / "band.png"},
    )

    result = band.abacus_cal_band(
        inputs,
        mode="nscf",
        kpath=["G", "X"],
        high_symm_points={"G": [0.0, 0.0, 0.0], "X": [0.5, 0.0, 0.0]},
        note="NiO-explicit",
    )

    assert result["band_calc_dir"] == work_path.absolute()
    assert seen["inputs_dir"] != inputs
    assert seen["mode"] == "nscf"
    assert (inputs / "magnetic.stru").read_bytes() == source_stru
    assert (inputs / "initial.kpt").read_bytes() == source_kpt
    assert not (inputs / "KPT_band").exists()

    staged_input = ReadInput(seen["inputs_dir"] / "INPUT")
    assert staged_input["stru_file"] == "STRU"
    assert staged_input["kpoint_file"] == "KPT"
    kpt_band = (seen["inputs_dir"] / "KPT_band").read_text(encoding="utf-8")
    assert "0.50000000000" in kpt_band
    runtime_input = ReadInput(work_path / "INPUT")
    assert runtime_input["stru_file"] == "STRU"
    assert runtime_input["kpoint_file"] == "KPT"
    assert (work_path / "KPT").read_text(encoding="utf-8") == kpt_band


def test_band_auto_path_reads_custom_stru_from_staging(tmp_path, monkeypatch):
    inputs = _write_custom_inputs(tmp_path / "inputs")
    source_stru = (inputs / "magnetic.stru").read_bytes()
    work_path = tmp_path / "band-work"
    work_path.mkdir()
    seen = {}

    def fake_scf(abacus_inputs_dir, mode, always_run=False, required_matrix="hsr", note=None):
        seen.update(inputs_dir=Path(abacus_inputs_dir), mode=mode)
        return {"work_path": work_path, "mode": "nscf"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(band, "property_calculation_scf", fake_scf)
    monkeypatch.setattr(band, "run_abacus", lambda _path: None)
    monkeypatch.setattr(
        band,
        "abacus_plot_band_nscf",
        lambda *_args: {"band_gap": 0.25, "band_picture": work_path / "band.png"},
    )

    result = band.abacus_cal_band(inputs, mode="nscf", note="NiO-custom-auto")

    assert result["band_calc_dir"] == work_path.absolute()
    assert seen["inputs_dir"] != inputs
    assert seen["mode"] == "nscf"
    assert (inputs / "magnetic.stru").read_bytes() == source_stru
    staged = AbacusStru.ReadStru(seen["inputs_dir"] / "STRU")
    assert staged.get_natoms() == 4
    assert staged.get_atommag() == [2.0, -2.0, 0.0, 0.0]
    assert (seen["inputs_dir"] / "KPT_band").is_file()


def test_band_auto_path_selects_reusable_matrix_outputs_in_private_copy(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path / "inputs")
    output_dir = inputs / "OUT.ABACUS"
    output_dir.mkdir()
    matrix_files = {
        "data-HR-sparse_SPIN0.csr": b"hr",
        "data-SR-sparse_SPIN0.csr": b"sr",
        "data-rR-sparse.csr": b"rr",
    }
    for name, content in matrix_files.items():
        (output_dir / name).write_bytes(content)
    source_output = {
        name: (output_dir / name).read_bytes() for name in matrix_files
    }
    work_path = tmp_path / "band-work"
    work_path.mkdir()
    seen = {}

    def fake_scf(abacus_inputs_dir, mode, always_run=False, required_matrix="hsr", note=None):
        seen.update(
            inputs_dir=Path(abacus_inputs_dir),
            mode=mode,
            always_run=always_run,
            required_matrix=required_matrix,
            note=note,
        )
        return {"work_path": Path(abacus_inputs_dir), "mode": mode}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(band, "property_calculation_scf", fake_scf)
    monkeypatch.setattr(
        band,
        "abacus_plot_band_pyatb",
        lambda *_args: {"band_gap": 0.3, "band_picture": work_path / "band.png"},
    )

    result = band.abacus_cal_band(inputs, mode="auto", note="NiO-reuse")

    staged = seen["inputs_dir"]
    assert result["band_calc_dir"] == staged.absolute()
    assert seen["mode"] == "pyatb"
    assert seen["always_run"] is False
    # BAND_STRUCTURE 只读 H/S：要求 rR 会让一次已完成的 SCF 被整体重算。
    assert seen["required_matrix"] == "hs"
    assert (staged / "OUT.ABACUS").is_dir()
    assert not (staged / "OUT.ABACUS").is_symlink()
    for name, content in matrix_files.items():
        assert (staged / "OUT.ABACUS" / name).read_bytes() == content
        assert (inputs / "OUT.ABACUS" / name).read_bytes() == source_output[name]


def test_band_auto_path_keeps_nonmagnetic_input_compatible(tmp_path, monkeypatch):
    inputs = tmp_path / "si-inputs"
    inputs.mkdir()
    for name in ("INPUT", "KPT", "STRU"):
        shutil.copy2(SI_INPUTS / name, inputs / name)
    work_path = tmp_path / "si-band-work"
    work_path.mkdir()
    seen = {}

    def fake_scf(abacus_inputs_dir, mode, always_run=False, required_matrix="hsr", note=None):
        seen.update(inputs_dir=Path(abacus_inputs_dir), mode=mode)
        return {"work_path": work_path, "mode": "pyatb"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(band, "property_calculation_scf", fake_scf)
    monkeypatch.setattr(
        band,
        "abacus_plot_band_pyatb",
        lambda *_args: {"band_gap": 0.4, "band_picture": work_path / "band.png"},
    )

    result = band.abacus_cal_band(inputs, mode="auto", note="Si-auto")

    assert result["band_calc_dir"] == work_path.absolute()
    assert seen["mode"] == "auto"
    staged = AbacusStru.ReadStru(seen["inputs_dir"] / "STRU")
    assert staged.get_natoms() == 1
    assert staged.get_atommag() == [0.0]


def test_band_charge_reuse_runs_nscf_in_staging_with_real_property_helper(
    tmp_path, monkeypatch
):
    inputs = _write_inputs(tmp_path / "inputs")
    output_dir = inputs / "OUT.ABACUS"
    output_dir.mkdir()
    charge_bytes = b"source-charge"
    (output_dir / "SPIN1_CHG.cube").write_bytes(charge_bytes)
    source_stru = (inputs / "STRU").read_bytes()
    source_input = (inputs / "INPUT").read_bytes()
    source_kpt = (inputs / "KPT").read_bytes()
    seen = {}

    class FakeResult:
        def __getitem__(self, key):
            return {
                "normal_end": True,
                "scf_steps": 1,
                "converge": True,
                "energies": [-1.0],
            }[key]

    def fake_run_abacus(work_path):
        seen["run_path"] = Path(work_path)
        (Path(work_path) / "OUT.ABACUS" / "BANDS_1.dat").write_text(
            "stub band output", encoding="utf-8"
        )

    def fake_plot(work_path, *_args):
        seen["plot_path"] = Path(work_path)
        return {"band_gap": 0.15, "band_picture": Path(work_path) / "band.png"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(band, "property_calculation_scf", pyatb_module.property_calculation_scf)
    monkeypatch.setattr(pyatb_module, "RESULT", lambda **_kwargs: FakeResult())
    monkeypatch.setattr(
        pyatb_module,
        "run_abacus",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("charge reuse must not launch a fresh SCF")
        ),
    )
    monkeypatch.setattr(band, "run_abacus", fake_run_abacus)
    monkeypatch.setattr(band, "abacus_plot_band_nscf", fake_plot)

    result = band.abacus_cal_band(inputs, mode="auto", note="NiO-charge-reuse")

    staged = seen["run_path"]
    assert result["band_calc_dir"] == staged.absolute()
    assert seen["plot_path"] == staged
    assert staged != inputs
    assert (staged / "OUT.ABACUS" / "SPIN1_CHG.cube").read_bytes() == charge_bytes
    assert (staged / "OUT.ABACUS" / "BANDS_1.dat").is_file()
    assert (inputs / "STRU").read_bytes() == source_stru
    assert (inputs / "INPUT").read_bytes() == source_input
    assert (inputs / "KPT").read_bytes() == source_kpt
    assert not (inputs / "OUT.ABACUS" / "BANDS_1.dat").exists()


def test_band_staging_writes_kpoint_file_not_kpt_file(tmp_path, monkeypatch):
    """Regression: _stage_band_inputs must write ABACUS-native 'kpoint_file',
    not the non-standard 'kpt_file' that ABACUS rejects with
    'THE PARAMETER NAME kpt_file IS NOT USED'.

    See: task 45077, 45082 (ABACUS v3.10.1 band failure).
    """
    inputs = _write_custom_inputs(tmp_path / "inputs")
    work_path = tmp_path / "band-work"
    work_path.mkdir()
    seen = {}

    def fake_scf(abacus_inputs_dir, mode, always_run=False, required_matrix="hsr", note=None):
        seen.update(inputs_dir=Path(abacus_inputs_dir), mode=mode)
        return {"work_path": work_path, "mode": "nscf"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(band, "property_calculation_scf", fake_scf)
    monkeypatch.setattr(band, "run_abacus", lambda *a, **kw: None)
    monkeypatch.setattr(band, "abacus_plot_band_nscf", lambda *a, **kw: {
        "band_gap": 0.0, "band_picture": str(work_path / "band.png"),
    })

    band.abacus_cal_band(
        inputs,
        mode="nscf",
        kpath=["G", "X"],
        high_symm_points={"G": [0.0, 0.0, 0.0], "X": [0.5, 0.0, 0.0]},
        note="regression",
    )

    staged_input = ReadInput(seen["inputs_dir"] / "INPUT")
    # The correct ABACUS parameter name is 'kpoint_file', not 'kpt_file'
    assert "kpoint_file" in staged_input, (
        "staged INPUT must contain ABACUS-native 'kpoint_file'"
    )
    assert "kpt_file" not in staged_input, (
        "staged INPUT must NOT contain non-standard 'kpt_file' "
        "(ABACUS rejects it: THE PARAMETER NAME 'kpt_file' IS NOT USED)"
    )
    assert staged_input["kpoint_file"] == "KPT"

    # Also verify the raw file content does not contain 'kpt_file'
    raw_text = (seen["inputs_dir"] / "INPUT").read_text(encoding="utf-8")
    for line in raw_text.splitlines():
        stripped = line.split("#")[0].strip()
        if stripped:
            key = stripped.split()[0]
            assert key != "kpt_file", (
                f"INPUT file contains 'kpt_file' key: {line!r}"
            )

    runtime_input = ReadInput(work_path / "INPUT")
    assert "kpoint_file" in runtime_input
    assert "kpt_file" not in runtime_input
