"""
Calculating elastic constants using ABACUS.
"""
import os
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import copy
from pymatgen.analysis.elasticity.elastic import Strain
from pymatgen.analysis.elasticity.elastic import ElasticTensor
from pymatgen.analysis.elasticity.strain import DeformedStructureSet
from pymatgen.analysis.elasticity.stress import Stress

from abacustest.lib_prepare.comm import kspacing2kpt
from abacustest.lib_prepare.abacus import AbacusStru, ReadInput, WriteKpt, WriteInput
from abacustest.lib_model.comm import check_abacus_inputs
from abacusagent.modules.util.comm import run_abacus, link_abacusjob, allocate_work_path, collect_metrics

def prepare_deformed_stru(
    input_stru_dir: Path,
    norm_strain: float,
    shear_strain: float
):
    """
    Generate deformed structures
    """
    input_params = ReadInput(os.path.join(input_stru_dir, "INPUT"))
    stru_file = input_params.get("stru_file", "STRU")
    original_stru = AbacusStru.ReadStru(os.path.join(input_stru_dir, stru_file))

    norm_strains = [-norm_strain, -0.5*norm_strain, +0.5*norm_strain, +norm_strain]
    shear_strains = [-shear_strain, -0.5*shear_strain, +0.5*shear_strain, +shear_strain]
    deformed_strus = DeformedStructureSet(original_stru.to_pymatgen(),
                                         symmetry=False,
                                         norm_strains=norm_strains,
                                         shear_strains=shear_strains)
    
    return deformed_strus

def prepare_deformed_stru_inputs(
    deformed_strus: DeformedStructureSet,
    work_path: Path,
    input_stru_dir: Path,
):
    """
    Prepare ABACUS inputs directories from deformed structures and prepared inputs templates
    """
    abacus_inputs_dirs = []
    input_params = ReadInput(os.path.join(input_stru_dir, "INPUT"))
    stru_file = input_params.get("stru_file", "STRU")
    original_stru = AbacusStru.ReadStru(os.path.join(input_stru_dir, stru_file))

    stru_counts = 1
    for deformed_stru in deformed_strus:
        abacus_inputs_dir = os.path.join(work_path, f'deformed-stru-{stru_counts:0>3d}')
        os.mkdir(abacus_inputs_dir)
        link_abacusjob(src=input_stru_dir,
                       dst=abacus_inputs_dir,
                       exclude=["STRU"],
                       exclude_directories=True,
                       copy_files=["INPUT", "KPT", "*log", "*json"])
        
        # Write deformed structure to ABACUS STRU format
        deformed_stru_abacus = copy.deepcopy(original_stru)
        deformed_stru_abacus.set_cell(deformed_stru.lattice.matrix.tolist(), bohr=False)
        deformed_stru_abacus.set_coord(deformed_stru.cart_coords.tolist(), bohr=False, direct=False)
        deformed_stru_abacus.write(os.path.join(abacus_inputs_dir, stru_file))

        abacus_inputs_dirs.append(Path(abacus_inputs_dir).absolute())
        stru_counts += 1

    return abacus_inputs_dirs

def collected_stress_to_pymatgen_stress(stress: List[float]):
    """
    Transform calculated stress (units in kBar) collected by abacustest
    to Pymatgen format (units in GPa)
    """
    return Stress(-0.1 * np.array([stress[0:3],
                                   stress[3:6],
                                   stress[6: ]])) # 1 kBar = 0.1 GPa

def abacus_cal_elastic(
    abacus_inputs_dir: Path,
    norm_strain: float = 0.01,
    shear_strain: float = 0.01,
    kspacing: float = 0.08,
    relax_force_thr_ev: float = 0.01,
    note=None,
    work_parent: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Calculate various elastic constants for a given structure using ABACUS. 
    Args:
        abacus_inputs_dir (str): Path to the ABACUS input files, which contains the INPUT, STRU, KPT, and pseudopotential or orbital files.
        norm_strain (float): Normal strain to calculate elastic constants, default is 0.01.
        shear_strain (float): Shear strain to calculate elastic constants, default is 0.01.
        kspacing (float): K-point spacing for ABACUS calculation, default is 0.08. Units in Bohr^{-1}.
        relax_force_thr_ev (float): Threshold for force convergence of the relax calculation for each deformed structure, default is 0.02. Units in eV/Angstrom.
        note: Optional task label used to name generated work directories. Agent-facing wrappers must pass a non-empty note; None is kept for backward-compatible internal calls.
        work_parent: Optional explicit parent for the generated work directory.
            When omitted, the historical cwd-relative allocation is kept.
    Returns:
        A dictionary containing the following keys:
        - elastic_cal_dir (Path): Work path of running abacus_cal_elastic. 
        - elastic_constants (np.array in (6,6) dimension): Calculated elastic constants in Voigt notation. Units in GPa.
        - bulk_modulus (float): Calculated bulk modulus in GPa.
        - shear_modulus (float): Calculated shear modulus in GPa.
        - young_modulus (float): Calculated Young's modulus in GPa.
        - poisson_ratio (float): Calculated Poisson's ratio.
    Raises:
        RuntimeError: If ABACUS calculation when calculating stress for input structure or deformed structures fails.
    """
    try:
        is_valid, msg = check_abacus_inputs(abacus_inputs_dir)
        if not is_valid:
            raise RuntimeError(f"Invalid ABACUS input files: {msg}")
        
        abacus_inputs_dir = Path(abacus_inputs_dir).absolute()
        work_path = allocate_work_path(note=note, work_parent=work_parent)
        input_stru_dir = Path(os.path.join(work_path, "input_stru")).absolute()

        link_abacusjob(src=abacus_inputs_dir,
                       dst=input_stru_dir,
                       copy_files=["INPUT", "STRU", "KPT"])

        input_params = ReadInput(os.path.join(input_stru_dir, "INPUT"))
        input_params['kspacing'] = kspacing
        stru = AbacusStru.ReadStru(os.path.join(input_stru_dir, input_params.get("stru_file", "STRU")))

        if 'kspacing' in input_params.keys():
            kpt = kspacing2kpt(input_params['kspacing'], stru.get_cell())
            WriteKpt(kpoint_list = kpt + [0, 0, 0],
                     file_name = os.path.join(input_stru_dir, input_params.get('kpoint_file', 'KPT')))
            del input_params['kspacing']
        input_params["calculation"] = 'relax'
        input_params["force_thr_ev"] = relax_force_thr_ev
        input_params["cal_stress"] = 1
        WriteInput(input_params, os.path.join(input_stru_dir, "INPUT"))

        deformed_strus = prepare_deformed_stru(input_stru_dir, norm_strain, shear_strain)
        strain = [Strain.from_deformation(d) for d in deformed_strus.deformations]

        deformed_stru_job_dirs = prepare_deformed_stru_inputs(deformed_strus, 
                                                              work_path,
                                                              input_stru_dir)

        all_dirs = [input_stru_dir] + deformed_stru_job_dirs
        run_abacus(all_dirs)

        collected_metrics = ["normal_end", "converge", "stress"]

        input_stru_result = collect_metrics(input_stru_dir, collected_metrics)
        if input_stru_result['converge'] is True:
            input_stru_stress = collected_stress_to_pymatgen_stress(input_stru_result['stress'])
        else:
            raise RuntimeError("SCF calculation for input structure doesn't converge")

        deformed_stru_stresses = []
        for idx, deformed_stru_job_dir in enumerate(deformed_stru_job_dirs):
            deformed_stru_result = collect_metrics(deformed_stru_job_dir, collected_metrics)
            if deformed_stru_result['converge'] is True:
                deformed_stru_stress = collected_stress_to_pymatgen_stress(deformed_stru_result['stress'])
                deformed_stru_stresses.append(deformed_stru_stress)
            else:
                raise RuntimeError(f"SCF calculation for deformed structure {idx} doesn't converge")

        result = ElasticTensor.from_independent_strains(strain,
                                                        deformed_stru_stresses,
                                                        eq_stress=input_stru_stress,
                                                        vasp=False)

        elastic_tensor = result.voigt.tolist()
        bv, gv = result.k_voigt, result.g_voigt
        ev = 9 * bv * gv / (3 * bv + gv)
        uV = (3 * bv - 2 * gv) / (6 * bv + 2 * gv)

        return {
            "elastic_cal_dir": Path(work_path).absolute(),
            "elastic_tensor": elastic_tensor,
            "bulk_modulus": float(bv),
            "shear_modulus": float(gv),
            "young_modulus": float(ev),
            "poisson_ratio": float(uV)
        }
    except Exception as e:
        return {"message": f"Calculating elastic properties failed: {e}"}
