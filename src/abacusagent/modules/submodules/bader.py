'''
workflow of calculating Bader charges.
'''
import os
import re
import glob
import shutil
import json
import csv
import unittest
from typing import List, Dict, Optional, Any
from pathlib import Path

import numpy as np
from pymatgen.core.periodic_table import Element

from abacustest.lib_prepare.abacus import ReadInput, WriteInput, AbacusStru
from abacustest.lib_model.comm import check_abacus_inputs

from abacusagent.modules.util.comm import allocate_work_path, run_abacus, link_abacusjob, generate_work_path, run_command, has_chgfile,collect_metrics
from abacusagent.modules.util.cube_manipulator import read_gaussian_cube

BADER_EXE = os.environ.get("BADER_EXE", "bader") # use environment variable to specify the bader executable path


"""
Install bader:

apt install subversion
svn co https://theory.cm.utexas.edu/svn/bader
cd bader
make
cp bader /usr/local/bin/bader
"""

def parse_abacus_cmd(cmd: str) -> Dict[str, str|int]:
    '''
    parse the abacus command to get parallelization
    options, such as `mpirun`, `mpiexec`, `OMP_NUM_THREADS`, etc.
    A typical command looks like:
    ```bash
    OMP_NUM_THREADS=4 /path/to/mpirun -np 8 /path/to/abacus
    ```

    Parameters:
    cmd (str): The command string to parse.

    Returns:
    dict: A dictionary containing parsed options and their values.
    '''
    pat = r'^(?:OMP_NUM_THREADS=(\d+)\s+)?' \
          r'(?:([\w/.-]*mpirun|mpiexec)\s+-[np]\s+(\d+)\s+)?' \
          r'(.+abacus.*)$'
    match = re.match(pat, cmd)
    if not match:
        raise ValueError(f"Failed to parse command: {cmd}")
    return {
        'openmp': match.group(1) is not None,
        'nthreads': int(match.group(1)) if match.group(1) else 1,
        'mpi': match.group(2),
        'nproc': int(match.group(3)) if match.group(3) else 1,
        'abacus': match.group(4)
    }

def ver_cmp(v1: str|tuple[int], v2: str|tuple[int]) -> int:
    """
    Compare two version strings or tuples. For example,
    "1.0" < "1.0.1" returns -1, "1.0.1" > "1.0" returns 1,

    Parameters:
    v1 (str|tuple[int]): First version to compare.
    v2 (str|tuple[int]): Second version to compare.

    Returns:
    int: -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2.
    """
    if isinstance(v1, str):
        v1 = list(map(int, re.split(r'\D+', v1.replace('v', ''))))
    if isinstance(v2, str):
        v2 = list(map(int, re.split(r'\D+', v2.replace('v', ''))))
    # Ensure both versions are of the same length
    max_len = max(len(v1), len(v2))
    v1 = tuple(list(v1) + [0] * (max_len - len(v1)))
    v2 = tuple(list(v2) + [0] * (max_len - len(v2)))

    return (v1 > v2) - (v1 < v2)  # Returns 1, 0, or -1

def calculate_charge_densities_with_abacus(
    abacus_inputs_dir: Path,
    note=None,
    work_parent=None,
) -> Optional[List[str]]:
    """
    Calculate the charge density using ABACUS in the specified job directory.

    Parameters:
    abacus_inputs_dir (str): Directory where the job files are located.
    note: Optional task label used to name generated work directories. Agent-facing wrappers must pass a non-empty note; None is kept for backward-compatible internal calls.

    Returns:
    list: List of file names for the charge density cube files.
    """
    # get the abacus version with `abacus --version`
    work_path = str(allocate_work_path(note=note, work_parent=work_parent))
    link_abacusjob(src=abacus_inputs_dir,
                   dst=work_path,
                   copy_files=["INPUT"])
    input_param = ReadInput(os.path.join(work_path, 'INPUT'))

    if has_chgfile(work_path):
        print("Charge file already exists, skipping SCF calculation.")
    else:
        input_param["calculation"] = "scf"
        input_param["out_chg"] = 1
        WriteInput(input_param, os.path.join(work_path, 'INPUT'))

        run_abacus(job_paths=work_path)

    fcube = glob.glob(os.path.join(work_path, 'OUT.*', '*.cube'))
    fcube.sort()

    return {
        "work_path": Path(work_path).absolute(),
        "cube_files": [Path(f).absolute() for f in fcube]
    }

def merge_charge_densities_of_different_spin(
    fcube: List[str]
) -> str:
    """
    Run the cube manipulator to process cube files.

    Parameters:
    fcube (list): List of file names for the cube files to be manipulated.

    Returns:
    str: Output cube file path.
    """
    if len(fcube) == 1:
        return fcube[0]

    dir_ = os.path.dirname(fcube[0])
    prefix_ = os.path.basename(fcube[0]).replace('.cube', '')
    fout = os.path.join(dir_, f"{prefix_}_merged.cube")

    from abacusagent.modules.util.cube_manipulator import read_gaussian_cube, axpy, write_gaussian_cube

    data = read_gaussian_cube(fcube[0])
    data2 = read_gaussian_cube(fcube[1])
    data["data"] = axpy(data["data"], data2["data"], beta=1.0)
    write_gaussian_cube(data, fout)
    return fout

def read_bader_acf(
    fn: Path
) -> List[float]:
    """
    Read Bader charges from a file.

    Parameters:
    fn (str): Path to the file containing Bader charges.

    Returns:
    list: A list of Bader charges.
    """
    with open(fn) as f:
        data = f.readlines()[2:-4]  # Skip header and footer
    data = [l.strip().split() for l in data if l.strip()]
    data = np.array(data, dtype=float)
    return data[:, 4].tolist()  # Return the Bader charges

def calculate_bader_charges(
    fcube: Path
) -> List[Path]:
    """
    Calculate Bader charges using the bader executable.

    Parameters:
    fcube (str): Path to the cube file containing charge density.

    Returns:
    list: A list of file paths generated by the Bader analysis.
    """
    cmd = f'{BADER_EXE} {fcube}'
    print(f"Running command: {cmd}")
    return_code, out, err = run_command(cmd, shell=True)
    if return_code != 0:
        raise RuntimeError(f"Bader command failed with error: {err}")

    with open('bader.out', 'w') as f:
        f.write(out)

    files = [os.path.join(os.getcwd(), f)
             for f in ['ACF.dat', 'AVF.dat', 'BCF.dat', 'bader.out']]
    if not all(os.path.exists(f) for f in files):
        raise FileNotFoundError("Incomplete Bader analysis output files.")
    files = [Path(f).absolute() for f in files]

    return files

def calculate_bader_charge_from_cube(
    fcube: List[Path]|Path,
    note=None,
) -> Dict[str, Any]:
    """
    postprocess the charge density to obtain Bader charges.

    Parameters:
    fcube (list|str): List of cube files or a single cube file path.
    note: Optional task label used to name generated work directories. Agent-facing wrappers must pass a non-empty note; None is kept for backward-compatible internal calls.

    Returns:
    dict: A dictionary containing:
        - atom_labels: List of atom labels.
        - net_bader_charges: List of net charge for each atom. Core charge is included.
        - number_of_electrons: List of number of electrons around each atom. Core charge is not included.
        - core_charge: List of core charge for each atom.
        - work_path: Absolute path to the work directory.
        - cube_file: Absolute path to the cube file used in this tool.
        - charge_results_json: Absolute path to the JSON file containing detailed Bader charge results
    """
    if not isinstance(fcube, (list, tuple)):
        fcube = [fcube]

    merged_cube_file = merge_charge_densities_of_different_spin(fcube)
    merged_cube_file = Path(merged_cube_file).absolute()

    cwd = os.getcwd()
    work_path = generate_work_path(note=note)
    os.chdir(work_path)

    # Copy the merged cube file to the work path for Bader analysis to avoid too long path for `bader` executable
    merged_cube_file_copy = Path("./merged_cube.cube")
    shutil.copy(merged_cube_file, merged_cube_file_copy)
    try:
        _ = calculate_bader_charges(merged_cube_file_copy)
        merged_cube_file_copy = merged_cube_file_copy.absolute()
        merged_cube_file_copy.unlink(missing_ok=True)  # Clean up the copied file
    except Exception as e:
        os.chdir(cwd)
        raise RuntimeError(f"Failed to calculate Bader charges: {e}")
    os.chdir(cwd)

    bader_charges = read_bader_acf(Path(work_path) / 'ACF.dat')
    cube_data = read_gaussian_cube(merged_cube_file)
    net_bader_charges = (np.array(cube_data["chg"]) - np.array(bader_charges)).tolist()
    atom_labels = [Element.from_Z(i).symbol for i in cube_data['atomz']]

    bader_charge_json = Path("./bader_charge_results.json").absolute()
    with open(bader_charge_json, "w") as fout:
        json.dump({
            "number_of_electrons": bader_charges,
            "core_charge": cube_data["chg"],
            "net_bader_charges": net_bader_charges,
            "atom_labels": atom_labels
        }, fout)

    return {
        "atom_labels": atom_labels,
        "net_bader_charges": net_bader_charges,
        "number_of_electrons": bader_charges,
        "core_charge": cube_data["chg"],
        "work_path": Path(work_path).absolute(),
        "cube_file": Path(merged_cube_file).absolute(),
        "charge_results_json": bader_charge_json.absolute(),
    }


def abacus_badercharge_run(
    abacus_inputs_dir: Path,
    note=None,
) -> List[float]:
    """
    Calculate Bader charges for a given ABACUS input file directory, with ABACUS as
    the dft software to calculate the charge density, and then postprocess
    the charge density with the cube manipulator and Bader analysis.

    Parameters:
    abacus_inputs_dir (str): Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
    note: Optional task label used to name generated work directories. Agent-facing wrappers must pass a non-empty note; None is kept for backward-compatible internal calls.

    Returns:
    dict: A dictionary containing:
        - net_bader_charges: List of net Bader charge for each atom. Core charge is included.
        - number_of_electrons: List of number of electrons around each atom. Core charge is not included.
        - core_charges: List of core charge for each atom.
        - atom_labels: Labels of atoms in the structure.
        - abacus_workpath: Absolute path to the ABACUS work directory.
        - badercharge_run_workpath: Absolute path to the Bader analysis work directory.
        - bader_result_csv: Absolute path to the CSV file containing detailed Bader charge results
    """
    try:
        is_valid, msg = check_abacus_inputs(abacus_inputs_dir)
        if not is_valid:
            raise RuntimeError(f"Invalid ABACUS input files: {msg}")

        # Run ABACUS to calculate charge density
        results = calculate_charge_densities_with_abacus(abacus_inputs_dir, note=note)
        abacus_jobpath = results["work_path"]
        fcube = results["cube_files"]

        stru = AbacusStru.ReadStru(os.path.join(abacus_jobpath, "STRU"))
        if stru is not None:
            atom_labels = stru.get_label(total=True)
        else:
            atom_labels = None

        # Postprocess the charge density to obtain Bader charges
        bader_results = calculate_bader_charge_from_cube(fcube, note=note)

        # Write necessary results to csv file
        bader_result_csv = Path("./bader_charge_results.csv").absolute()

        with open(bader_result_csv, "w") as fout:
            writer = csv.writer(fout)
            writer.writerow(['atom_label', 'net_bader_charge'])
            for label, charge in zip(bader_results["atom_labels"], bader_results["net_bader_charges"]):
                writer.writerow([label, f"{charge: .6f}"])

        return {
            "net_bader_charges": bader_results["net_bader_charges"],
            "number_of_electrons": bader_results["number_of_electrons"],
            "core_charges": bader_results["core_charge"],
            "atom_labels": atom_labels, # Use atom labels from ABACUS STRU file, not from cube file
            "abacus_workpath": Path(abacus_jobpath).absolute(),
            "badercharge_run_workpath": Path(bader_results["work_path"]).absolute(),
            "bader_result_csv": Path(bader_result_csv).absolute()
        }
    except Exception as e:
        return {"message": f"Calculating Bader charge failed: {e}"}

class TestBaderChargeWorkflow(unittest.TestCase):

    def test_parse_abacus_cmd(self):
        cmd = "OMP_NUM_THREADS=4 /path/to/mpirun -n 8 /path/to/abacus"
        expected = {
            'openmp': True,
            'nthreads': 4,
            'mpi': '/path/to/mpirun',
            'nproc': 8,
            'abacus': '/path/to/abacus'
        }
        result = parse_abacus_cmd(cmd)
        self.assertDictEqual(result, expected)

    def test_ver_cmp(self):
        self.assertEqual(ver_cmp("1.0.0", "1.0.1"), -1)
        self.assertEqual(ver_cmp("1.0.1", "1.0.0"), 1)
        self.assertEqual(ver_cmp("1.0.0", "1.0.0"), 0)
        self.assertEqual(ver_cmp((1, 0, 0), (1, 0, 1)), -1)
        self.assertEqual(ver_cmp((1, 0, 1), (1, 0, 0)), 1)
        self.assertEqual(ver_cmp((1, 0, 0), (1, 0, 0)), 0)
        self.assertEqual(ver_cmp("1.0", "1.0.0"), 0)
        self.assertEqual(ver_cmp("v3.10.0", "v3.9.0.4"), 1)

if __name__ == "__main__":
    unittest.main(exit=True)

    # Example prompt to invoke the Bader charge calculation
    '''
    Hello, I want to calculate Bader charges for the system in the directory at `/home/xxx/abacus-develop/representation/examples/scf/lcao_Si2`, could you please help me do this job? I think you will need to run ABACUS first to calculate the charge density, and then if there are two spin channels, you will need to merge the charge density cube files and then run Bader analysis on the merged file. If there is only one spin channel, you can directly run Bader analysis on the charge density cube file. There are several executables you will need to complete the whole process. You can run the ABACUS executable directly with `abacus`, the cube manipulator is a Python script that you can find it at `/home/xxx/abacus-develop/representation/tools/plot-tools/cube_manipulator.py`. And the Bader analysis program is at `/home/xxx/soft/bader`.
    '''

