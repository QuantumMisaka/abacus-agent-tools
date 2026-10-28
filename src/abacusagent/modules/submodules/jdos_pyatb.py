import os
from pathlib import Path
from typing import Dict, Any

from abacusagent.modules.util.comm import run_pyatb, run_command
from abacusagent.modules.util.pyatb import property_calculation_scf, PyatbInputGenerator

def plot_jdos_pyatb(
    jdos_dat_file: Path,
    plot_filename: str = "jdos.png"
) -> Path:
    """
    Plot the joint density of states (JDOS) from a given JDOS.dat file.
    Args:
        jdos_dat_file (Path): The path to the JDOS.dat file.
    Returns:
        Path: The path to the plotted JDOS.
    """
    energies, jdoses = [], []
    with open(jdos_dat_file, 'r') as f:
        for line in f:
            words = line.split()
            energies.append(float(words[0]))
            jdoses.append(float(words[1]))
    
    import matplotlib.pyplot as plt
    plt.plot(energies, jdoses, 'r-')
    plt.xlim(min(energies), max(energies))
    plt.xlabel(r"$\hbar \omega$ (eV)")
    plt.ylabel("JDOS (statas/eV)")
    plt.title("JDOS")
    plot_file = Path("./" + plot_filename).absolute()
    plt.savefig(plot_file, dpi=300)
    
    return plot_file

def pyatb_calculate_jdos(
    abacus_inputs_path: Path,
    note=None,
    work_parent=None,
) -> Dict[str, Any]:
    """
    Plot the joint density of states (JDOS) using pyatb after ABACUS SCF calculation.
    Args:
        abacus_inputs_path (Path): The path to the ABACUS input files.
        note: Optional task label used to name the generated SCF/JDOS work directory. Agent-facing wrappers must pass a non-empty note; None is kept for backward-compatible internal calls.
    Returns:
        Dict[str, Any]: A dictionary containing path to the plotted JDOS.
    """
    scf_results = property_calculation_scf(
        abacus_inputs_path, mode='pyatb', note=note, work_parent=work_parent
    )

    if not scf_results["normal_end"]:
        raise RuntimeError("SCF calculation did not finish successfully.")
    elif not scf_results["converge"]:
        raise RuntimeError("SCF calculation did not converge.")
    else:
        os.chdir(scf_results["work_path"])
        PyatbInputGenerator(jdos=True).run()

    run_pyatb(Path(os.path.join(scf_results["work_path"], "./pyatb")).absolute())

    jdos_dat_file = Path("./pyatb/Out/JDOS/JDOS.dat").absolute()
    jdos_plot_file = plot_jdos_pyatb(jdos_dat_file, "jdos.png")
    return {"work_path": scf_results["work_path"],
            "scf_converge": scf_results["converge"],
            "scf_normal_end": scf_results["normal_end"],
            "jdos_fig_path": Path(jdos_plot_file).absolute(),
            "jdos_data_path": jdos_dat_file.absolute(),}
