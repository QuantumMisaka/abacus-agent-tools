from pathlib import Path

import pytest
from abacusagent.modules.submodules import relax


@pytest.mark.parametrize(
    ("normal_end", "relax_converge", "expected_prepare_note"),
    [
        (True, True, "H2O_relax_inputs"),
        (True, False, None),
        (False, True, None),
    ],
)
def test_relax_handoff_directory_inherits_parent_note_only_when_converged(
    monkeypatch, tmp_path, normal_end, relax_converge, expected_prepare_note
):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    handoff = tmp_path / "H2O_relax_inputs-8379"
    generated_notes = []
    prepared_notes = []

    def fake_generate_work_path(*, note, work_parent=None):
        generated_notes.append(note)
        work = tmp_path / "H2O_relax-1731"
        work.mkdir(exist_ok=True)
        return work

    def fake_prepare(relax_jobpath, note=None, *, work_parent=None):
        prepared_notes.append((Path(relax_jobpath), note))
        handoff.mkdir()
        return {"job_path": handoff}

    monkeypatch.setattr(relax, "check_abacus_inputs", lambda _path: (True, ""))
    monkeypatch.setattr(relax, "allocate_work_path", fake_generate_work_path)
    monkeypatch.setattr(relax, "link_abacusjob", lambda **_kwargs: None)
    monkeypatch.setattr(relax, "prepare_relax_inputs", lambda **_kwargs: None)
    monkeypatch.setattr(relax, "run_abacus", lambda _path: None)
    monkeypatch.setattr(
        relax,
        "relax_postprocess",
        lambda _path: {
            "normal_end": normal_end,
            "relax_converge": relax_converge,
        },
    )
    monkeypatch.setattr(relax, "abacus_prepare_inputs_from_relax_results", fake_prepare)

    outputs = relax.abacus_do_relax(inputs, note="H2O_relax")

    assert generated_notes == ["H2O_relax"]
    if expected_prepare_note is None:
        assert prepared_notes == []
        assert "new_abacus_inputs_dir" not in outputs
    else:
        assert prepared_notes == [
            (tmp_path / "H2O_relax-1731", expected_prepare_note)
        ]
        assert outputs["new_abacus_inputs_dir"] == handoff
