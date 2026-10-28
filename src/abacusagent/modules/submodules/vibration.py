from typing import List, Optional
from pathlib import Path
from abacustest.lib_model.comm import check_abacus_inputs
from abacustest.lib_model.model_019_vibration import prepare_abacus_vibration_analysis, post_abacus_vibration_analysis_onejob

from abacusagent.modules.util.comm import allocate_work_path, link_abacusjob, run_abacus


def abacus_vibration_analysis(abacus_inputs_dir: Path,
                              selected_atoms: Optional[List[int]] = None,
                              stepsize: float = 0.01,
                              temperature: Optional[float] = 298.15,
                              note=None,
                              work_parent: Optional[Path] = None):
    """
    Performing vibrational analysis using finite displacement method.
    This tool function is usually followed by a relax calculation (`calculation` is set to `relax`).
    Args:
        abacus_inputs_dir (Path): Path to the ABACUS input files directory.
        selected_atoms (Optional[List[int]]): Indices (started from 1) of atoms included in the vibrational analysis. If this
            parameter are not given, all atoms in the structure will be included.
        stepsize (float): Step size to displace cartesian coordinates of atoms during the vibrational analysis.
            Units in Angstrom. The default value (0.01 Angstrom) is generally OK.
        temperature (float): Temperature used to calculate thermodynamic quantities. Units in Kelvin.
        note: Optional task label used to name generated work directories. Agent-facing wrappers must pass a non-empty note; None is kept for backward-compatible internal calls.
        work_parent: Optional explicit parent for the generated work directory.
            When omitted, the historical cwd-relative allocation is kept.
    Returns:
        A dictionary containing the following keys:
        - 'frequencies': List of real frequencies from vibrational analysis. Imaginary frequencies are represented by negative 
            values. Units in cm^{-1}.
        - 'zero_point_energy': Zero-point energy summed over all modes. Units in eV.
        - 'vib_analysis_work_path': Path to directory performing vibrational analysis. Containing animation of normal modes 
            with non-zero frequency in ASE traj format and `vib` directory containing collected forces.
        - 'thermo_corr': Corrections to entropy and free energy from vibrations using harmonic approximation. Keys are temperatures.
           For each temperature, contaning 2 quantities:
           - 'entropy':  Vibrational entropy using harmonic approximation. Units in eV/K.
           - 'free_energy': Free energy using harmonic approximation. Units in eV.
    """

    try:
        is_valid, msg = check_abacus_inputs(abacus_inputs_dir)
        if not is_valid:
            raise RuntimeError(f"Invalid ABACUS input files: {msg}")
        
        if stepsize <= 0:
            raise ValueError("stepsize should be positive.")
        
        work_path = allocate_work_path(note=note, work_parent=work_parent)
        link_abacusjob(src=abacus_inputs_dir,
                       dst=work_path,
                       copy_files=["INPUT", "STRU", "KPT"],
                       exclude=["OUT.*", "*.log", "*.out", "*.json", "log"],
                       exclude_directories=True)
        job_paths = prepare_abacus_vibration_analysis(job_path=work_path,
                                                      selected_atoms=selected_atoms,
                                                      stepsize=stepsize)
        
        # Run ABACUS calculation for all prepared jobs
        run_abacus(job_paths)
        
        vib_results = post_abacus_vibration_analysis_onejob(work_path, temperature=[temperature])
        vib_results['vib_analysis_work_path'] = Path(work_path).absolute()
        return vib_results
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'message': f"Doing vibration analysis failed: {e}"}
