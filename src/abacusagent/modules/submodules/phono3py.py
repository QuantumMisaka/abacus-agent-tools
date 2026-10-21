"""Phono3py thermal closure shared by split and one-shot ABACUS routes.

The module deliberately keeps the runtime package contract explicit: the
selected ``abacus:1.1.0`` SIF supplies the version/API evidence, while this
adapter never invents a phono3py version.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from abacusagent.modules.util.comm import run_abacus


def _dataset_count(dataset: Mapping[str, Any] | None) -> int:
    if not isinstance(dataset, Mapping):
        return 0
    displacements = dataset.get("displacements")
    if displacements is not None:
        return len(displacements)
    first_atoms = dataset.get("first_atoms")
    if first_atoms is not None:
        # Type-I FC3 datasets nest pair displacements under each first atom's
        # ``second_atoms``; every first- and second-atom entry maps to one
        # displacement job, so the flat job count is the flattened sum.
        return sum(1 + len(entry.get("second_atoms") or []) for entry in first_atoms)
    return 0


def _is_type_ii_dataset(dataset: Any) -> bool:
    """Type-II (random snapshot) datasets carry a flat ``displacements`` key.

    Real phono3py dataset mappings have no ``type`` key; the displacement
    layout is the only reliable discriminator across supported versions.
    """
    return isinstance(dataset, Mapping) and dataset.get("displacements") is not None


def _require_symfc() -> None:
    try:
        import symfc  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "random (Type-II) phono3py displacements require the symfc "
            "force-constants calculator, which is not installed in this runtime"
        ) from exc


_SCF_CONVERGED_TOKEN = "charge density convergence is achieved"


def _current_scf_log(path: Path) -> tuple[list[str], bool]:
    """Return the current native invocation and whether the log was appended."""
    log = path / "OUT.ABACUS" / "running_scf.log"
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").split("\n")
    except OSError:
        return [], False
    # Each native ABACUS invocation writes a Start Time line.  A prior
    # converged run in an appended log must not complete the current one.
    starts = [i for i, line in enumerate(lines) if re.match(r"\s*Start Time is\b", line)]
    current = lines[starts[-1]:] if starts else lines
    return current + [""], len(starts) > 1


def _scf_log_converged(path: Path) -> bool:
    """Check density convergence in the current native SCF invocation."""
    lines, _appended = _current_scf_log(path)
    return any(_SCF_CONVERGED_TOKEN in line for line in lines)


def _single_scf_result(
    path: Path, *, current_lines: list[str] | None = None, appended: bool | None = None
) -> tuple[Any, float]:
    """Load finite E/F from the same SCF invocation."""
    import dpdata
    import numpy as np

    if current_lines is None or appended is None:
        current_lines, appended = _current_scf_log(path)
    if appended:
        # dpdata's directory reader independently takes the last energy and
        # force in the whole file. Parse its current-run lines with the same
        # low-level reader so an old force cannot pair with a new energy.
        from dpdata.formats.abacus import scf

        energy, converged = scf.get_energy(current_lines)
        force = scf.get_force(current_lines, 0)
        if not converged or energy is None or force is None:
            raise ValueError("current ABACUS SCF run has no complete energy/force frame")
        input_lines = (path / "INPUT").read_text(encoding="utf-8").splitlines()
        stru_path = scf.get_geometry_in(str(path), input_lines)
        structure = scf.get_frame_from_stru(stru_path)
        natoms = sum(structure["atom_numbs"])
        forces = np.asarray(force, dtype=float)[np.newaxis, :, :]
        energies = np.asarray([energy], dtype=float)
        if forces.shape[1] != natoms:
            raise ValueError("current ABACUS SCF force table does not cover every atom")
    else:
        labelled = dpdata.LabeledSystem(str(path), fmt="abacus/scf")
        forces = np.asarray(labelled["forces"], dtype=float)
        energies = np.asarray(labelled["energies"], dtype=float)
    if forces.ndim != 3 or forces.shape[0] != 1 or forces.shape[1] < 1 or forces.shape[2] != 3:
        raise ValueError("each ABACUS displacement job must contain one (atom, xyz) force frame")
    if energies.size != 1 or not np.isfinite(forces).all() or not np.isfinite(energies).all():
        raise ValueError("each ABACUS displacement job must contain one finite energy and finite forces")
    return forces[0], float(energies.reshape(-1)[0])


def displacement_scf_ready(path: Path) -> bool:
    """Whether a displacement SCF has current-run convergence and finite E/F.

    The SAI array runner calls this same predicate in the computation image.
    """
    current_lines, appended = _current_scf_log(path)
    if not any(_SCF_CONVERGED_TOKEN in line for line in current_lines):
        return False
    try:
        _single_scf_result(path, current_lines=current_lines, appended=appended)
        return True
    except Exception:
        return False


# Preserve callers that imported the earlier private completion helper.
_job_is_completed = displacement_scf_ready


def collect_force_energy_records(
    ph3: Any,
    jobs: Sequence[tuple[Path, str]],
    *,
    work_dir: Path,
) -> dict[str, Any]:
    """Collect displacement records with one strict, shared contract.

    Both split collection and one-shot closure use this helper.  A missing,
    unreadable, or out-of-order job is a hard failure: silently dropping a
    force/energy row changes the displacement-to-force mapping and can make a
    formally successful BTE physically meaningless.
    """
    import numpy as np

    root = work_dir.resolve()
    grouped: dict[str, list[tuple[Path, Any, Any]]] = {"fc3": [], "fc2": []}
    for raw_path, raw_kind in jobs:
        path = Path(raw_path).resolve()
        kind = str(raw_kind or "fc3").lower()
        if kind not in grouped:
            raise ValueError(f"unsupported phono3py displacement kind: {kind}")
        if not path.is_dir() or root not in path.parents:
            raise ValueError("displacement jobs must be existing directories contained in work_dir")
        try:
            current_lines, appended = _current_scf_log(path)
            if not any(_SCF_CONVERGED_TOKEN in line for line in current_lines):
                raise ValueError("displacement SCF did not converge in its current run")
            forces, energy = _single_scf_result(path, current_lines=current_lines, appended=appended)
            grouped[kind].append((path, forces, energy))
        except Exception as exc:
            raise RuntimeError(f"failed to collect force/energy from displacement job {path}") from exc

    expected_fc3 = _dataset_count(getattr(ph3, "dataset", None))
    expected_fc2 = _dataset_count(getattr(ph3, "phonon_dataset", None))
    for kind, expected in (("fc3", expected_fc3), ("fc2", expected_fc2)):
        actual = len(grouped[kind])
        if expected and actual != expected:
            raise RuntimeError(f"{kind.upper()} displacement/force cardinality mismatch: {expected} != {actual}")
    if not grouped["fc3"]:
        raise RuntimeError("no FC3 displacement force/energy records were collected")

    return {
        "forces_fc3": np.asarray([row[1] for row in grouped["fc3"]]),
        "energies_fc3": np.asarray([row[2] for row in grouped["fc3"]]),
        "forces_fc2": np.asarray([row[1] for row in grouped["fc2"]]),
        "energies_fc2": np.asarray([row[2] for row in grouped["fc2"]]),
        "completed_jobs": [row[0].name for kind in ("fc3", "fc2") for row in grouped[kind]],
    }


def _set_compatibility_controls(ph3: Any, options: Mapping[str, Any]) -> None:
    """Configure controls through phono3py's actual object contract."""
    mesh = options["mesh"]
    if not hasattr(type(ph3), "mesh_numbers"):
        raise RuntimeError("phono3py object lacks the mesh_numbers property")
    ph3.mesh_numbers = mesh
    if list(ph3.mesh_numbers) != list(mesh):
        raise RuntimeError("phono3py mesh_numbers was not retained")
    cutoff = options["cutoff_frequency"]
    if cutoff is not None:
        # phono3py 4.x has no public setter; this is the constructor-owned
        # state used by _set_mesh_numbers and is deliberately verified.
        if not hasattr(ph3, "_cutoff_frequency"):
            raise RuntimeError("phono3py object lacks supported cutoff_frequency state")
        ph3._cutoff_frequency = float(cutoff)
        if float(ph3._cutoff_frequency) != float(cutoff):
            raise RuntimeError("phono3py cutoff_frequency was not retained")


def _init_phph_interaction_if_required(ph3: Any) -> None:
    initializer = getattr(ph3, "init_phph_interaction", None)
    if initializer is None:
        raise RuntimeError("installed phono3py API lacks init_phph_interaction")
    initializer()


def run_phono3py_thermal(
    *,
    work_dir: str | Path,
    displacement_yaml: str | Path,
    displacement_jobs: Sequence[Any] = (),
    mesh: Sequence[int] = (2, 2, 2),
    temperatures: Sequence[float] = (300.0,),
    cutoff_frequency: float | None = None,
    isotope: bool | None = None,
    use_N_U: bool | None = None,
    boundary_mfp: float | None = None,
    boundary_length: float | None = None,
    runtime_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run force jobs, close FC2/FC3 and BTE, and return canonical artifacts."""
    from toolkits.phono3py_thermal import validate_runtime_contract, validate_thermal_options

    options = validate_thermal_options(
        mesh=mesh, temperatures=temperatures, cutoff_frequency=cutoff_frequency,
        isotope=isotope, use_N_U=use_N_U, boundary_mfp=boundary_mfp,
        boundary_length=boundary_length,
    )
    # The expected version/API snapshot is a required input.  An empty
    # mapping must never make a live runtime self-validate against itself.
    contract = validate_runtime_contract(runtime_contract or {})
    root = Path(work_dir).resolve()
    yaml_path = Path(displacement_yaml).resolve()
    if not yaml_path.is_file() or yaml_path.parent != root:
        raise ValueError("displacement_yaml must be an existing canonical file in work_dir")
    typed_jobs = []
    for item in displacement_jobs:
        if isinstance(item, Mapping):
            typed_jobs.append((Path(item["path"]).resolve(), str(item.get("kind", "fc3")).lower()))
        else:
            typed_jobs.append((Path(item).resolve(), "fc3"))
    jobs = [path for path, _kind in typed_jobs]
    if jobs:
        for job in jobs:
            if not job.is_dir() or root not in job.parents:
                raise ValueError("displacement jobs must be existing directories contained in work_dir")
    import phono3py

    # Load before running any job so a runtime that cannot close the loaded
    # dataset (e.g. Type-II without symfc) fails before burning SCF compute.
    ph3 = phono3py.load(str(yaml_path))
    fc3_calculator = None
    if _is_type_ii_dataset(getattr(ph3, "dataset", None)):
        _require_symfc()
        fc3_calculator = "symfc"
    fc2_calculator = None
    if _is_type_ii_dataset(getattr(ph3, "phonon_dataset", None)):
        _require_symfc()
        fc2_calculator = "symfc"
    if jobs:
        pending_jobs = [job for job in jobs if not _job_is_completed(job)]
        if pending_jobs:
            run_abacus(pending_jobs)
        records = collect_force_energy_records(ph3, typed_jobs, work_dir=root)
        ph3.forces = records["forces_fc3"]
        ph3.supercell_energies = records["energies_fc3"]
        if len(records["forces_fc2"]):
            # Phono3py names the second-order force payload
            # ``phonon_forces`` (not ``forces_fc2``).
            ph3.phonon_forces = records["forces_fc2"]
            ph3.phonon_supercell_energies = records["energies_fc2"]
    if not hasattr(ph3, "produce_fc3") or not hasattr(ph3, "run_thermal_conductivity"):
        raise RuntimeError("installed phono3py API lacks FC3/BTE closure methods")
    if fc3_calculator is not None:
        ph3.produce_fc3(fc_calculator=fc3_calculator)
    else:
        ph3.produce_fc3()
    # ``supercell_fc2`` is serialized by phono3py as the independent
    # ``phonon_supercell_matrix``.  In phono3py 4.1.0 this path does not
    # close FC2 as a side effect of ``produce_fc3``; it must be produced
    # before initializing the ph-ph interaction used by the BTE.
    if getattr(ph3, "phonon_supercell_matrix", None) is not None:
        if not hasattr(ph3, "produce_fc2"):
            raise RuntimeError("installed phono3py API lacks produce_fc2 for independent FC2 supercell")
        if fc2_calculator is not None:
            ph3.produce_fc2(fc_calculator=fc2_calculator)
        else:
            ph3.produce_fc2()
    # Symmetrize the force constants before the BTE step.  Unsymmetrized
    # FC3/FC2 from finite differences carry numerical noise that inflates
    # scattering and collapses kappa (systematic-111 benchmark evidence:
    # 5.57 W/mK unsymmetrized vs 122.46 W/mK with symmetrize_fc3+fc2).
    # Fail closed on runtimes lacking the API — never run an unsymmetrized
    # closure silently.
    if not hasattr(ph3, "symmetrize_fc3"):
        raise RuntimeError("installed phono3py API lacks symmetrize_fc3 for BTE closure")
    ph3.symmetrize_fc3()
    if not hasattr(ph3, "symmetrize_fc2"):
        raise RuntimeError("installed phono3py API lacks symmetrize_fc2 for BTE closure")
    ph3.symmetrize_fc2()
    # Configure the object as well as the call signature.  This is required
    # even for releases that still accept legacy keywords, because mesh and
    # cutoff are consumed by the interaction initialization path.
    _set_compatibility_controls(ph3, options)
    _init_phph_interaction_if_required(ph3)
    old_cwd = Path.cwd()
    try:
        os.chdir(root)
        try:
            ph3.run_thermal_conductivity(
                mesh=options["mesh"], temperatures=options["temperatures"],
                cutoff_frequency=options["cutoff_frequency"], is_isotope=options["isotope"],
                is_N_U=options["use_N_U"], boundary_mfp=options["boundary_mfp_api_um"],
                write_kappa=True, write_gamma=True,
            )
        except TypeError:
            # Some supported releases consume mesh/cutoff through the loaded
            # object.  Set and verify every control before retrying; a retry
            # that drops a requested control is scientifically unsafe.
            _set_compatibility_controls(ph3, options)
            _init_phph_interaction_if_required(ph3)
            ph3.run_thermal_conductivity(
                temperatures=options["temperatures"],
                is_isotope=options["isotope"], is_N_U=options["use_N_U"],
                boundary_mfp=options["boundary_mfp_api_um"],
                write_kappa=True, write_gamma=True,
            )
    finally:
        os.chdir(old_cwd)
    all_kappa = sorted(root.glob("kappa*.hdf5")) + sorted(root.glob("kappa*.h5"))
    # ``write_gamma=True`` also emits per-grid-point ``kappa-m<mesh>-g<gp>.hdf5``
    # files that sort before the canonical ``kappa-m<mesh>.hdf5``.  The canonical
    # thermal-conductivity tensor must not be confused with a gamma component.
    canonical = [path for path in all_kappa if not re.search(r"-g\d+\.(?:hdf5|h5)$", path.name)]
    kappa_candidates = canonical or all_kappa
    if not kappa_candidates:
        raise RuntimeError("phono3py BTE completed without a canonical kappa HDF5 artifact")
    gamma_candidates = []
    try:
        import h5py
        with h5py.File(kappa_candidates[0], "r") as handle:
            names = []
            handle.visit(names.append)
            gamma_candidates = [kappa_candidates[0]] if any(name.rsplit("/", 1)[-1].startswith("gamma") for name in names) else []
    except Exception:
        gamma_candidates = []
    kappa_path = kappa_candidates[0]
    gamma_path = gamma_candidates[0] if gamma_candidates else None
    if hasattr(ph3, "save"):
        ph3.save(filename=str(root / "phono3py_thermal.yaml"))
    metadata = {"runtime": contract, "options": options, "displacement_yaml": yaml_path.name}
    (root / "phono3py_runtime_contract.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"work_dir": str(root), "kappa_hdf5": str(kappa_path), "gamma_hdf5": str(gamma_path) if gamma_path is not None else None, "runtime": contract, "options": options}
