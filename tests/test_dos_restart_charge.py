"""DOS reuse of ABACUS binary and cube charge outputs."""

from pathlib import Path
import shutil

import pytest


@pytest.mark.parametrize("suffix,read_dir", [
    ("ABACUS", None), ("CALC", None), ("CALC", "checkpoint"),
])
def test_binary_checkpoint_is_a_reusable_dos_source(tmp_path, suffix, read_dir):
    from abacusagent.modules.submodules.dos import _dos_charge_source

    params = {"suffix": suffix, "nspin": 1}
    if read_dir:
        params["read_file_dir"] = read_dir
    output = tmp_path / (read_dir or f"OUT.{suffix}")
    output.mkdir()
    (output / f"{suffix}-CHARGE-DENSITY.restart").write_bytes(b"charge")
    assert _dos_charge_source(tmp_path, params) == output


def test_charge_source_requires_matching_suffix_and_complete_spin_set(tmp_path):
    from abacusagent.modules.submodules.dos import _dos_charge_source

    output = tmp_path / "OUT.CALC"
    output.mkdir()
    (output / "ABACUS-CHARGE-DENSITY.restart").write_bytes(b"wrong")
    (output / "SPIN1_CHG.cube").write_text("up")
    params = {"suffix": "CALC", "nspin": 2}
    assert _dos_charge_source(tmp_path, params) is None
    (output / "SPIN2_CHG.cube").write_text("down")
    assert _dos_charge_source(tmp_path, params) == output
    assert _dos_charge_source(tmp_path, {"suffix": "CALC", "nspin": 4}) is None


@pytest.mark.parametrize("output_charge,old_charge", [
    (True, True), (True, False), (False, True),
])
def test_nscf_reads_actual_charge_source_after_new_cwd(
    tmp_path, monkeypatch, output_charge, old_charge,
):
    from abacusagent.modules.submodules import dos
    from abacustest.lib_prepare.abacus import ReadInput

    source = tmp_path / "scf"
    source.mkdir()
    checkpoint = source / "SOURCE_CHECKPOINT"
    checkpoint.mkdir()
    if old_charge:
        (checkpoint / "CALC-CHARGE-DENSITY.restart").write_bytes(b"old")
    (source / "INPUT").write_text(
        "INPUT_PARAMETERS\nsuffix CALC\nread_file_dir SOURCE_CHECKPOINT\nnspin 1\n"
    )
    if output_charge:
        output = source / "OUT.CALC"
        output.mkdir()
        (output / "CALC-CHARGE-DENSITY.restart").write_bytes(b"new")
    monkeypatch.setattr(dos, "link_abacusjob", lambda *, src, dst, **_kw:
                        shutil.copytree(src, dst, dirs_exist_ok=True))
    monkeypatch.setattr(dos, "run_abacus", lambda _path: None)
    monkeypatch.setattr(dos, "RESULT", lambda **_kw: {"normal_end": True})
    work = tmp_path / "nscf"
    monkeypatch.setattr(dos, "generate_work_path", lambda **_kw: work)
    dos.abacus_dos_run_nscf(source)
    actual = ReadInput(str(work / "INPUT"))["read_file_dir"]
    expected = ("OUT.CALC" if output_charge else str(checkpoint.resolve()))
    assert actual == expected
