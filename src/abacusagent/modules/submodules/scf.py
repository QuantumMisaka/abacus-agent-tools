import os
from pathlib import Path
from typing import Callable, Dict, Any, Optional

from abacustest.lib_prepare.abacus import ReadInput, WriteInput
from abacustest.lib_model.comm import check_abacus_inputs

from abacusagent.modules.util.comm import allocate_work_path, link_abacusjob, run_abacus, collect_metrics


def abacus_calculation_scf(
    abacus_inputs_dir: Path,
    note=None,
    *,
    work_parent: Optional[Path] = None,
    before_run: Optional[Callable[[Path], None]] = None,
) -> Dict[str, Any]:
    """
    Run ABACUS SCF calculation.

    Args:
        abacus_inputs_dir (Path): Path to the directory containing the ABACUS input files.
        note: Optional task label used to name generated work directories. Agent-facing wrappers must pass a non-empty note; None is kept for backward-compatible internal calls.
        work_parent: Optional explicit parent for the generated work directory.
            When omitted, the historical cwd-relative allocation is kept.
        before_run: Optional internal check of the final work directory immediately before ABACUS is launched. Its exceptions propagate unchanged.
    Returns:
        A dictionary containing the path to output file of ABACUS calculation, and a dictionary containing whether the SCF calculation
        finished normally, the SCF is converged or not, the converged SCF energy and total time used.
    """
    try:
        if before_run is None:
            is_valid, msg = check_abacus_inputs(abacus_inputs_dir)
            if not is_valid:
                raise RuntimeError(f"Invalid ABACUS input files: {msg}")
        
        work_path = allocate_work_path(note=note, work_parent=work_parent)
        link_abacusjob(src=abacus_inputs_dir, dst=work_path, copy_files=['INPUT', 'STRU'])
        input_params = ReadInput(os.path.join(work_path, "INPUT"))

        input_params['calculation'] = 'scf'
        WriteInput(input_params, os.path.join(work_path, "INPUT"))
    except Exception as e:
        return {"message": f"Performing SCF calculation failed: {e}"}

    if before_run is not None:
        before_run(work_path)

    try:
        run_abacus(work_path)

        return_dict = {'scf_work_dir': Path(work_path).absolute()}
        return_dict.update(collect_metrics(work_path, metrics_names=['normal_end', 'converge', 'energy', 'total_time']))

        return return_dict
    except Exception as e:
        return {"message": f"Performing SCF calculation failed: {e}"}
