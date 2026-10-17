"""The internal execution hook observes the actual directory handed to ABACUS."""

from pathlib import Path

import pytest

from abacusagent.modules.submodules import md, relax, scf


@pytest.fixture(params=["md", "relax", "scf"])
def backend_case(request, monkeypatch, tmp_path):
    mode = request.param
    backend = {"md": md, "relax": relax, "scf": scf}[mode]
    source = tmp_path / "source"
    source.mkdir()
    work = tmp_path / "work"
    events = []

    def link_abacusjob(*, src, dst, **_kwargs):
        assert Path(src) == source
        Path(dst).mkdir()
        (Path(dst) / "INPUT").write_text("calculation scf\n")
        events.append("link")

    def read_input(_path):
        return {"calculation": "scf", "suffix": "ABACUS"}

    def write_input(params, path):
        Path(path).write_text(f"calculation {params['calculation']}\n")
        events.append("write")

    monkeypatch.setattr(backend, "generate_work_path", lambda **_kwargs: work)
    monkeypatch.setattr(backend, "link_abacusjob", link_abacusjob)
    monkeypatch.setattr(backend, "ReadInput", read_input)
    monkeypatch.setattr(backend, "WriteInput", write_input)
    if mode != "md":
        monkeypatch.setattr(backend, "check_abacus_inputs", lambda _path: (True, ""))
    if mode == "md":
        monkeypatch.setattr(backend, "collect_metrics", lambda _path: {"normal_end": True})
        monkeypatch.setattr(
            backend, "convert_md_dump_to_ase_traj", lambda _path: (work / "traj", 1)
        )
        invoke = lambda **kwargs: md.abacus_run_md(source, **kwargs)
        calculation = "md"
    elif mode == "relax":
        monkeypatch.setattr(
            backend, "relax_postprocess", lambda _path: {"normal_end": False}
        )
        invoke = lambda **kwargs: relax.abacus_do_relax(
            source, relax_cell=True, **kwargs
        )
        calculation = "cell-relax"
    else:
        monkeypatch.setattr(backend, "collect_metrics", lambda _path, **_kwargs: {})
        invoke = lambda **kwargs: scf.abacus_calculation_scf(source, **kwargs)
        calculation = "scf"

    def run_abacus(path):
        assert Path(path) == work
        assert (work / "INPUT").read_text() == f"calculation {calculation}\n"
        events.append("run")

    monkeypatch.setattr(backend, "run_abacus", run_abacus)
    return backend, invoke, work, events, calculation


def test_before_run_sees_final_input_once_before_launch(backend_case):
    _backend, invoke, work, events, calculation = backend_case

    def before_run(path):
        assert path == work.absolute()
        assert (path / "INPUT").read_text() == f"calculation {calculation}\n"
        events.append("check")

    result = invoke(before_run=before_run)

    assert "message" not in result
    assert events == ["link", "write", "check", "run"]


def test_before_run_rejection_propagates_and_does_not_launch(backend_case):
    _backend, invoke, work, events, calculation = backend_case
    rejection = ValueError("preflight rejected final input")

    def before_run(path):
        assert path == work
        assert (path / "INPUT").read_text() == f"calculation {calculation}\n"
        events.append("check")
        raise rejection

    with pytest.raises(ValueError) as raised:
        invoke(before_run=before_run)

    assert raised.value is rejection
    assert events == ["link", "write", "check"]


def test_before_run_default_preserves_existing_launch(backend_case):
    _backend, invoke, _work, events, _calculation = backend_case

    result = invoke()

    assert "message" not in result
    assert events == ["link", "write", "run"]


def test_ordinary_execution_error_keeps_existing_message_result(backend_case, monkeypatch):
    backend, invoke, _work, events, _calculation = backend_case

    def fail_run(_path):
        events.append("run")
        raise RuntimeError("execution failed")

    monkeypatch.setattr(backend, "run_abacus", fail_run)

    result = invoke()

    assert "execution failed" in result["message"]
    assert events == ["link", "write", "run"]


@pytest.mark.parametrize("backend_case", ["relax", "scf"], indirect=True)
def test_final_callback_supersedes_legacy_source_checker(backend_case, monkeypatch):
    backend, invoke, work, events, _calculation = backend_case

    def legacy_checker(_path):
        raise SystemExit("legacy source checker must not run")

    monkeypatch.setattr(backend, "check_abacus_inputs", legacy_checker)

    result = invoke(before_run=lambda path: events.append("check"))

    assert "message" not in result
    assert events == ["link", "write", "check", "run"]


@pytest.mark.parametrize("backend_case", ["relax", "scf"], indirect=True)
def test_without_callback_legacy_source_checker_remains(backend_case, monkeypatch):
    backend, invoke, _work, events, _calculation = backend_case
    checked = []

    def legacy_checker(path):
        checked.append(path)
        return False, "legacy input failure"

    monkeypatch.setattr(backend, "check_abacus_inputs", legacy_checker)

    result = invoke()

    assert checked
    assert "legacy input failure" in result["message"]
    assert events == []
