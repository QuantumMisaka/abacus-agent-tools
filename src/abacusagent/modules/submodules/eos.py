import math
import numbers
import os
import shutil
from pathlib import Path
from typing import Literal, List, Optional
import copy
import numpy as np
from abacustest.lib_prepare.abacus import AbacusStru, ReadInput, WriteInput
from abacustest.lib_model.comm_eos import eos_fit
from abacustest.lib_model.comm import check_abacus_inputs

from abacusagent.modules.util.comm import run_abacus, link_abacusjob, allocate_work_path, collect_metrics


def _read_eos_stru(path):
    """Translate the pinned STRU parser's input rejection into a regular error."""
    try:
        return AbacusStru.ReadStru(path)
    except SystemExit as exc:
        if exc.code != 1:
            raise
        raise ValueError(f"EOS could not read STRU: {path}") from exc


def _resolve_eos_stru_path(inputs_dir: Path, stru_reference: str) -> Path:
    """Resolve the selected EOS structure before entering private staging."""
    reference = Path(str(stru_reference))
    if not reference.is_absolute():
        reference = Path(inputs_dir) / reference
    resolved = reference.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"EOS STRU file does not exist: {resolved}")
    return resolved


def _stage_eos_inputs(
    inputs_dir: Path,
    work_path: Path,
    input_stru_reference: str,
) -> tuple[Path, dict, object]:
    """Create an EOS-private input copy with a local structure reference."""
    input_stru_dir = work_path / "input_stru"
    source_stru_path = _resolve_eos_stru_path(inputs_dir, input_stru_reference)
    link_abacusjob(
        src=inputs_dir,
        dst=input_stru_dir,
        copy_files=["INPUT"],
        exclude=["OUT.*", "*.log", "*.out", "*.json", "log"],
        exclude_directories=True,
    )

    staged_stru_path = input_stru_dir / "STRU"
    if staged_stru_path.exists() or staged_stru_path.is_symlink():
        staged_stru_path.unlink()
    shutil.copy2(source_stru_path, staged_stru_path)

    staged_input = ReadInput(input_stru_dir / "INPUT")
    staged_input["stru_file"] = "STRU"
    return input_stru_dir, staged_input, _read_eos_stru(staged_stru_path)


def is_cubic(cell: List[List[float]]) -> bool:
    """
    Check if the cell is cubic.

    Args:
        cell (List[List[float]]): The cell vectors.

    Returns:
        bool: True if the cell is cubic, False otherwise.
    """
    a, b, c = np.array(cell[0]), np.array(cell[1]), np.array(cell[2])
    len_a, len_b, len_c = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
    alpha = np.arccos(np.dot(b, c) / (len_b * len_c))
    beta = np.arccos(np.dot(a, c) / (len_a * len_c))
    gamma = np.arccos(np.dot(a, b) / (len_a * len_b))
    if np.isclose(len_a, len_b) and np.isclose(len_b, len_c):
        if np.isclose(alpha, np.pi / 2) and np.isclose(beta, np.pi / 2) and np.isclose(gamma, np.pi / 2):
            return True
        else:
            return False
    else:
        return False


def _validate_eos_sampling(
    stru_scale_number: int,
    scale_stepsize: float,
) -> tuple[int, float]:
    """Validate and normalize the symmetric EOS sampling controls."""
    if (
        isinstance(stru_scale_number, bool)
        or not isinstance(stru_scale_number, numbers.Integral)
        or stru_scale_number < 2
    ):
        raise ValueError("stru_scale_number must be an integer of at least 2")

    if (
        isinstance(scale_stepsize, bool)
        or not isinstance(scale_stepsize, numbers.Real)
    ):
        raise ValueError("scale_stepsize must be a finite positive real number")
    step = float(scale_stepsize)
    if not math.isfinite(step) or step <= 0:
        raise ValueError("scale_stepsize must be a finite positive real number")

    number = int(stru_scale_number)
    try:
        lower_scale = 1.0 - float(number) * step
    except OverflowError:
        lower_scale = -math.inf
    if lower_scale <= 0:
        raise ValueError(
            "1 - stru_scale_number * scale_stepsize must be positive"
        )
    return number, step

def plot_eos(lat_params, fit_energy, scaled_lat_params, energies):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(lat_params, fit_energy, label='Fitted Birch-Murnaghan EOS', color='red')
    plt.scatter(scaled_lat_params, energies, label='Calculated Energies', color='blue')
    plt.xlabel('Lattice Parameter (Angstrom)')
    plt.ylabel('Energy (eV)')
    plt.title('Birch-Murnaghan EOS Fit')
    plt.legend()
    plt.grid()
    plt.savefig('birch_murnaghan_eos_fit.png', dpi=300)
    fig_path = Path('birch_murnaghan_eos_fit.png').absolute()

    return fig_path

def abacus_eos(
    abacus_inputs_dir: Path,
    stru_scale_number: int = 3,
    scale_stepsize: float = 0.02,
    note=None,
    work_parent: Optional[Path] = None,
):
    """
    Use Birch-Murnaghan equation of state (EOS) to calculate the EOS data. The
    fitted crystal must be cubic. ``stru_scale_number`` is the number of
    lattice-length steps on each side of the original cell, so the total number
    of generated points is ``2 * stru_scale_number + 1``. For example,
    ``stru_scale_number=5`` and ``scale_stepsize=0.01`` generate 11 points
    from 0.95 to 1.05 times the original lattice length.

    Args:
        abacus_inputs_dir (Path): Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
        stru_scale_number (int): Number of lattice-length steps on each side of
            the original cell. It must be at least 2; the default 3 generates 7
            symmetric points.
        scale_stepsize (float): Positive finite lattice-length step. The
            default 0.02 samples 2% increments on each side of the original
            cell.
        note: Optional task label used to name generated work directories. Agent-facing wrappers must pass a non-empty note; None is kept for backward-compatible internal calls.
        work_parent: Optional explicit parent for the generated work directory.
            When omitted, the historical cwd-relative allocation is kept.

    Returns:
        Dict[str, Any]: A dictionary containing EOS calculation results:
            - "eos_work_path" (Path): Working directory for the EOS calculation.
            - "eos_fig_path" (Path): Path to the EOS fitting plot (energy vs. volume).
            - "E0" (float): Minimum energy (in eV) from the EOS fit.
            - "V0" (float): Equilibrium volume (in Å³) corresponding to E0.
            - "B0" (float): Bulk modulus (in GPa) at equilibrium volume.
            - "B0_deriv" (float): Pressure derivative of the bulk modulus.
    """
    stru_scale_number, scale_stepsize = _validate_eos_sampling(
        stru_scale_number, scale_stepsize
    )

    failure_stage = "input validation"
    failure_code = None
    work_path = None
    try:
        is_valid, msg = check_abacus_inputs(abacus_inputs_dir)
        if not is_valid:
            raise RuntimeError(f"Invalid ABACUS input files: {msg}")

        work_path = allocate_work_path(note=note, work_parent=work_parent)
        failure_stage = "input preparation"

        input_params = ReadInput(os.path.join(abacus_inputs_dir, "INPUT"))
        input_stru_reference = input_params.get("stru_file", "STRU")
        input_stru_dir, input_params, input_stru = _stage_eos_inputs(
            Path(abacus_inputs_dir),
            work_path,
            input_stru_reference,
        )
        input_stru_file = "STRU"

        # Generated lattice parameters for EOS calculation
        original_cell = np.asarray(input_stru.get_cell(), dtype=float)
        if not is_cubic(original_cell.tolist()):
            failure_code = "unsupported_non_cubic_cell"
            raise ValueError("EOS calculation currently supports only cubic cells")

        scales = [
            1 + i * scale_stepsize
            for i in range(-stru_scale_number, stru_scale_number + 1)
        ]
        scaled_cells = [original_cell * scale for scale in scales]
        scaled_lat_params = [np.linalg.norm(cell[0]) for cell in scaled_cells]

        input_params["calculation"] = 'cell-relax'
        input_params['fixed_axes'] = 'volume'
        input_params['force_thr_ev'] = 0.01
        input_params['stress_thr'] = 1.0
        WriteInput(input_params, input_stru_dir / "INPUT")

        scale_cell_job_dirs = []
        for i, new_cell in enumerate(scaled_cells):
            dir_name = Path(os.path.join(work_path, f"scale_cell_{i}")).absolute()
            os.makedirs(dir_name, exist_ok=True)
            scale_cell_job_dirs.append(dir_name)

            link_abacusjob(
                src=input_stru_dir,
                dst=Path(dir_name).absolute(),
                copy_files=["INPUT", input_stru_file],
                exclude=["OUT.*", "*.log", "*.out", "*.json", "log"],
                exclude_directories=True
            )

            stru = copy.deepcopy(input_stru)
            stru.set_cell(new_cell, bohr=False, change_coord=True)
            stru.write(os.path.join(dir_name, input_stru_file))

        failure_stage = "scale-cell execution"
        run_abacus(scale_cell_job_dirs)

        energies = []
        volumes = []
        failure_stage = "scale-cell result collection"
        for i, job_dir in enumerate(scale_cell_job_dirs):
            metrics = collect_metrics(job_dir)
            if metrics['normal_end'] is not True or metrics['converge'] is not True:
                raise RuntimeError(f"Job {i} did not end normally or did not converge. Please check the job directory: {job_dir}")
            energies.append(metrics['energy'])
            point_input = ReadInput(job_dir / "INPUT")
            point_stru_file = point_input.get("stru_file", "STRU")
            point_stru = _read_eos_stru(job_dir / point_stru_file)
            volumes.append(
                abs(float(np.linalg.det(np.asarray(point_stru.get_cell(), dtype=float))))
            )

        failure_stage = "Birch-Murnaghan fit"
        V0, E0, fit_volume, fit_energy, B0, B0_deriv, residual0 = eos_fit(volumes, energies)
        lat_params = np.cbrt(np.array(fit_volume))
        observed_lat_params = np.cbrt(np.array(volumes))

        failure_stage = "EOS plot"
        fig_path = plot_eos(lat_params, fit_energy, observed_lat_params, energies)

        return {
            "eos_work_path": work_path.absolute(),
            "eos_fig_path": fig_path.absolute(),
            "E0": E0,
            "V0": V0,
            "B0": B0,
            "B0_deriv": B0_deriv,
            "fit_quality": {"residual": residual0},
        }
    except Exception as e:
        result = {
            "failure_stage": failure_stage,
            "message": (
                f"Fitting EOS failed at {failure_stage}: "
                f"{type(e).__name__}: {e}"
            ),
        }
        if failure_code is not None:
            result["failure_code"] = failure_code
        if work_path is not None:
            result["eos_work_path"] = work_path.absolute()
        return result
