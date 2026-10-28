import errno
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST_ROOT / "src"))

from abacusagent.modules.util import work_path


def _allocate_in_process(parent):
    return work_path.generate_work_path(note="shared scf", parent=parent)


def test_note_slug_is_public_and_shared_with_candidate_names():
    assert work_path.slugify_work_note("  NiO 300K/nvt  ") == "NiO-300K-nvt"
    assert work_path.slugify_work_note("  中文 计算  ") == "中文-计算"
    assert work_path.slugify_work_note("! / ") == ""
    assert work_path.slugify_work_note("a" * 65) == "a" * 64
    assert work_path.slugify_work_note("a" * 63 + "/long") == "a" * 63
    names = work_path.iter_work_path_names(" NiO 300K/nvt ")
    assert [next(names) for _ in range(3)] == [
        "NiO-300K-nvt", "NiO-300K-nvt-002", "NiO-300K-nvt-003"
    ]
    with pytest.raises(ValueError, match="slug is empty"):
        work_path.iter_work_path_names(" / ")


def test_comm_reexports_public_naming_contract():
    from abacusagent.modules.util import comm

    assert comm.slugify_work_note is work_path.slugify_work_note
    assert comm.iter_work_path_names is work_path.iter_work_path_names
    assert comm.generate_work_path is work_path.generate_work_path
    assert comm.allocate_work_path is work_path.allocate_work_path


def test_allocate_work_path_uses_explicit_parent_and_avoids_collision(tmp_path):
    parent = tmp_path / "project" / "GaN"
    parent.mkdir(parents=True)
    (parent / "PBE_run").mkdir()
    cwd_before = Path.cwd()

    path = work_path.allocate_work_path("PBE_run", work_parent=parent)

    assert path == (parent / "PBE_run-002").resolve()
    assert path.is_dir()
    assert Path.cwd() == cwd_before
    assert not (tmp_path / "PBE_run").exists()


def test_allocate_work_path_keeps_legacy_cwd_when_parent_omitted(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    path = work_path.allocate_work_path("PBE_run")

    assert path == (tmp_path / "PBE_run").resolve()
    assert path.is_dir()


def test_first_free_name_is_created_and_existing_results_are_untouched(tmp_path):
    bare = tmp_path / "Si_scf"
    bare.mkdir()
    (bare / "sentinel").write_text("old result")
    (tmp_path / "Si_scf-002").mkdir()
    (tmp_path / "Si_scf-004").mkdir()

    assert work_path.generate_work_path("Si_scf", parent=tmp_path) == str(tmp_path / "Si_scf-003")
    assert work_path.generate_work_path("Si_scf", parent=tmp_path) == str(tmp_path / "Si_scf-005")
    assert work_path.generate_work_path("Si_relax", parent=tmp_path) == str(tmp_path / "Si_relax")
    assert (bare / "sentinel").read_text() == "old result"


def test_numeric_note_tail_and_old_random_name_are_plain_labels(tmp_path):
    (tmp_path / "Si_scf-002").mkdir()
    (tmp_path / "Si_scf-0007").mkdir()

    assert work_path.generate_work_path("Si_scf", parent=tmp_path) == str(tmp_path / "Si_scf")
    assert work_path.generate_work_path("Si_scf-002", parent=tmp_path) == str(tmp_path / "Si_scf-002-002")
    assert (tmp_path / "Si_scf-0007").is_dir()


def test_files_directories_and_both_symlink_types_occupy_names(tmp_path):
    (tmp_path / "occupied").write_text("keep")
    (tmp_path / "occupied-002").mkdir()
    (tmp_path / "occupied-003").symlink_to(tmp_path / "occupied-002", target_is_directory=True)
    (tmp_path / "occupied-004").symlink_to(tmp_path / "missing", target_is_directory=True)

    assert work_path.generate_work_path("occupied", parent=tmp_path) == str(tmp_path / "occupied-005")
    assert (tmp_path / "occupied").read_text() == "keep"
    assert (tmp_path / "occupied-003").is_symlink()
    assert (tmp_path / "occupied-004").is_symlink()


def test_create_false_only_queries_first_currently_free_name(tmp_path):
    (tmp_path / "Fe-scf").mkdir()
    (tmp_path / "Fe-scf-002").symlink_to(tmp_path / "missing")

    candidate = work_path.generate_work_path("Fe scf", create=False, parent=tmp_path)

    assert candidate == str(tmp_path / "Fe-scf-003")
    assert not Path(candidate).exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["Fe-scf", "Fe-scf-002"]


def test_default_relative_path_and_legacy_fallback_are_preserved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def legacy_caller():
        return work_path.generate_work_path(create=False)

    assert legacy_caller() == "legacy_caller"
    assert work_path.generate_work_path(None, "explicit", False) == "explicit"
    assert work_path.generate_work_path("中文 计算") == "中文-计算"
    assert (tmp_path / "中文-计算").is_dir()


def test_empty_note_and_normalization_collision(tmp_path):
    with pytest.raises(ValueError, match="slug is empty"):
        work_path.generate_work_path(note=" / ", parent=tmp_path)
    with pytest.raises(ValueError, match="slug is empty"):
        work_path.generate_work_path(note="", parent=tmp_path)
    assert work_path.generate_work_path("A/B", parent=tmp_path) == str(tmp_path / "A-B")
    assert work_path.generate_work_path("A B", parent=tmp_path) == str(tmp_path / "A-B-002")


def test_ordinal_extends_past_999(tmp_path):
    (tmp_path / "many").mkdir()
    for ordinal in range(2, 1000):
        (tmp_path / f"many-{ordinal:03d}").mkdir()

    assert work_path.generate_work_path("many", parent=tmp_path) == str(tmp_path / "many-1000")


def test_non_collision_mkdir_error_propagates(tmp_path, monkeypatch):
    original_mkdir = work_path.os.mkdir

    def denied_mkdir(path, *args, **kwargs):
        if Path(path) == tmp_path / "denied":
            raise PermissionError(errno.EACCES, "permission denied", str(path))
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(work_path.os, "mkdir", denied_mkdir)
    with pytest.raises(PermissionError) as exc_info:
        work_path.generate_work_path("denied", parent=tmp_path)
    assert exc_info.value.errno == errno.EACCES
    assert list(tmp_path.iterdir()) == []


def test_multiprocess_allocation_is_unique_and_preserves_existing_file(tmp_path):
    sentinel = tmp_path / "shared-scf"
    sentinel.write_text("old result")
    context = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=4, mp_context=context) as executor:
        paths = list(executor.map(_allocate_in_process, [tmp_path] * 12))

    assert len(paths) == len(set(paths)) == 12
    assert {Path(path).name for path in paths} == {f"shared-scf-{n:03d}" for n in range(2, 14)}
    assert all(Path(path).is_dir() for path in paths)
    assert sentinel.read_text() == "old result"
