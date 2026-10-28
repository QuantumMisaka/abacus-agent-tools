"""
Use Pyatb to do property calculation.
"""
import inspect
import os
from pathlib import Path
from typing import List, Dict, Any, Literal, Union, Optional
from abacusagent.modules.util.comm import (
    generate_stage_path,
    generate_work_path, 
    link_abacusjob, 
    run_abacus, 
    has_chgfile, 
    has_pyatb_matrix_files
)

from abacustest.lib_prepare.abacus import ReadInput, WriteInput, AbacusStru
from abacustest.lib_collectdata.collectdata import RESULT
from pyatb.easy_use.input_generator import *
from pyatb.easy_use.stru_analyzer import read_abacus_stru

from abacusagent.modules.util.comm import collect_metrics


def generate_input_chern_compatible(
    input_text,
    n_occu,
    occu_switch,
    dim,
    lattice_vectors,
    method,
    mp_density,
):
    """Call pyatb's Chern input helper across the 1.1.1/1.1.2 signature change."""
    params = inspect.signature(generate_input_chern).parameters
    if "mp_density" in params:
        return generate_input_chern(
            input_text,
            n_occu,
            occu_switch,
            dim,
            lattice_vectors,
            method,
            mp_density,
        )
    return generate_input_chern(
        input_text,
        n_occu,
        occu_switch,
        dim,
        lattice_vectors,
        method,
    )


def _property_work_path(note, work_parent) -> Path:
    """Allocate the property stage under an explicit parent when provided."""

    if work_parent is None:
        return Path(
            generate_work_path(note=note, fallback_name="property_calculation_scf")
        ).absolute()
    if not note or not str(note).strip():
        raise ValueError("property stage note must be non-empty with work_parent")
    return Path(generate_stage_path(str(note), parent=work_parent)).absolute()


def property_calculation_scf(
    abacus_inputs_path: Path,
    mode: Literal["nscf", "pyatb", "auto"] = "auto",
    always_run: bool = False,
    note=None,
    work_parent=None,
):
    """Perform the SCF calculation for property calculations like DOS or band structure.

    Args:
        abacus_inputs_path (Path): Path to the ABACUS input files.
        mode (Literal["nscf", "pyatb", "auto"]): Mode of operation, default is "auto".
            nscf: first run SCF with out_chg=1, then run nscf with init_chg=file.
            pyatb: run SCF with out_mat_r and out_mat_hs2 = 1, then calculate properties using Pyatb.
            auto: automatically determine the mode based on the input parameters. If basis is LCAO, use "pyatb", otherwise use "nscf".
        always_run (bool): Whether to always run the SCF calculation, even if the required output files already exist.
        note: Optional task label used to name generated work directories. Agent-facing wrappers must pass a non-empty note; None is kept for backward-compatible internal calls.
        work_parent: Optional explicit parent for the purpose-named property
            stage. When provided, ``note`` is used as the stage directory name.

    Returns:
        Dict[str, Any]: A dictionary containing the work path, normal end status, SCF steps, convergence status, and energies.
    """

    input_param = ReadInput(os.path.join(abacus_inputs_path, 'INPUT'))
    basis_type = input_param.get("basis_type", "pw").lower()

    if always_run:
        if mode == 'auto':
            if basis_type == 'lcao':
                mode = 'pyatb'
            else:
                mode = 'nscf'
        if mode == "nscf":
            input_param["calculation"] = "scf"
            input_param["out_chg"] = 1
        elif mode == "pyatb":
            input_param["calculation"] = "scf"
            input_param["out_mat_r"] = 1
            input_param["out_mat_hs2"] = 1
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'nscf', 'pyatb', or 'auto'.")
        
        work_path = _property_work_path(note, work_parent)
        link_abacusjob(src=abacus_inputs_path,
                       dst=work_path,
                       copy_files=["INPUT", "STRU", "KPT"])
        WriteInput(input_param, os.path.join(work_path, 'INPUT'))
        run_abacus(work_path, "abacus.log")
    else:
        if (mode in ["pyatb", "auto"] and has_pyatb_matrix_files(abacus_inputs_path)):
            print("Matrix files already exist, skipping SCF calculation.")
            work_path = abacus_inputs_path
        elif (mode in ["nscf", "auto"] and has_chgfile(abacus_inputs_path)):
            print("Charge files already exist, skipping SCF calculation.")
            work_path = abacus_inputs_path
        else:
            if mode == "auto":
                if basis_type.lower() == "lcao":
                    mode = "pyatb"
                else:
                    mode = "nscf"

            if basis_type == "pw" and mode == "pyatb":
                raise ValueError("Pyatb mode is not supported for PW basis. Please use 'nscf' mode instead.")

            work_path = _property_work_path(note, work_parent)
            link_abacusjob(src=abacus_inputs_path,
                           dst=work_path,
                           copy_files=["INPUT", "STRU", "KPT"])
            if mode == "nscf":
                input_param["calculation"] = "scf"
                input_param["out_chg"] = 1
            elif mode == "pyatb":
                input_param["calculation"] = "scf"
                input_param["out_mat_r"] = 1
                input_param["out_mat_hs2"] = 1
            else:
                raise ValueError(f"Invalid mode: {mode}. Use 'nscf', 'pyatb', or 'auto'.")

            WriteInput(input_param, os.path.join(work_path, 'INPUT'))
            run_abacus(work_path, "abacus.log")
        
    rs = RESULT(path=work_path, fmt="abacus")

    return {
        "work_path": Path(work_path).absolute(),
        "normal_end": rs["normal_end"],
        "scf_steps": rs["scf_steps"],
        "converge": rs["converge"],
        "energies": rs["energies"],
        "mode": mode
    }


class PyatbInputGenerator:

    def __init__(
        self,
        input: Path = Path("./"),
        output: Path = Path("./pyatb"),
        band: bool = False,
        kline: float = 0.01,
        knum: int = 0,
        kpath: Optional[str] = None,
        kmode: Optional[Literal['mp', 'line']] = None,
        dim: str = '3',
        tolerance: float = 1e-3,
        pdos: bool = False,
        findnodes: bool = False,
        erange: Union[List[float], float] = 4.0,
        frange: float = 1.0,
        optical: bool = False,
        jdos: bool = False,
        shift: bool = False,
        polar: bool = False,
        orange: Optional[Union[List[float], float]] = [0, 10],
        mp: float = 0.05,
        density: float = 0.05,
        bandunfolding: bool = False,
        m_matrix: Optional[List[List[int]]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ahc: bool = False,
        anc: bool = False,
        berry: bool = False,
        occu: int = 0,
        method: Literal['direct', 'Kobo'] = 'direct',
        berry_curvature_dipole: bool = False,
        cpge: bool = False,
        chern: bool = False,
        wilson_loop: bool = False,
        project_band: bool = False,
        spintexture: bool = False,
        bandrange: Optional[Union[List[int], int]] = None,
        max_kpoint_num: int = 4000,
        fermisurface: bool = False
    ):
        """
        Initialize the PyatbInputGenerator class. The values are same with main() in pyatb.easy_use.input_generator.main.
        Use `pyatb_input -h` to get detailed usage.
        """
        self.input = input
        self.output = output
        self.band = band
        self.kline = kline
        self.knum = knum
        self.kpath = kpath
        self.kmode = kmode
        self.dim = dim
        self.tolerance = tolerance
        self.pdos = pdos
        self.findnodes = findnodes
        self.erange = erange
        self.frange = frange
        self.optical = optical
        self.jdos = jdos
        self.shift = shift
        self.polar = polar
        self.mp = mp
        self.orange = orange
        self.density = density
        self.bandunfolding = bandunfolding
        self.m_matrix = m_matrix
        self.ahc = ahc
        self.anc = anc
        self.berry = berry
        self.occu = occu
        self.method = method
        self.berry_curvature_dipole = berry_curvature_dipole
        self.cpge = cpge
        self.chern = chern
        self.wilson_loop = wilson_loop
        self.project_band = project_band
        self.spintexture = spintexture
        self.bandrange = bandrange
        self.max_kpoint_num = max_kpoint_num
        self.fermisurface = fermisurface
    
    def run(self):
        """
        Do prepare the pyatb input files.
        """
        input_file = os.path.join(Path(self.input).absolute(), "INPUT")
        input_params = ReadInput(input_file)
        suffix = input_params.get("suffix", "ABACUS")
        abacus_out_suffix_dir = os.path.join(Path(self.input).absolute(), f"OUT.{suffix}")
        full_input_file = os.path.join(abacus_out_suffix_dir, "INPUT")
        full_input_params = ReadInput(full_input_file)

        latname = full_input_params.get("latname", 'none')
        nspin = full_input_params.get("nspin", 1)
        pp_dir = full_input_params.get("pseudo_dir", "./")
        orb_dir = full_input_params.get("orbital_dir", "./")

        collect_results = collect_metrics(self.input, ["energy", "efermi", "noccu_band", "nbands", "nelec"])
        e_tot, e_fermi, noccu_band, nbands, nelec = collect_results['energy'], collect_results['efermi'], collect_results['noccu_band'], collect_results['nbands'], collect_results['nelec']

        i_latname = None if latname == "none" else latname
        stru_file_path = os.path.join(Path(self.input).absolute(), full_input_params.get("stru_file", "STRU"))
        with open(stru_file_path, "r") as f_stru:
            ase_stru = read_abacus_stru(f_stru, i_latname, True)
        lattice_constant = 1.0
        lattice_vectors = ase_stru.get_cell()

        input_text = generate_input_init(nspin, e_fermi, abacus_out_suffix_dir, lattice_constant, lattice_vectors, self.max_kpoint_num)
        
        if self.bandrange:
            if ' ' in self.bandrange:
                band_range = [int(band) for band in self.bandrange.split()]
            elif len(self.bandrange.split()) == 1:
                band_range = int(self.bandrange.split()[0])
                band_range = [max(1, noccu_band - band_range), min(nbands, noccu_band + band_range)]
        else:   
            band_range = [max(1, noccu_band - 100), min(nbands, noccu_band + 100)]

        if self.orange:
            if len(self.orange) == 1:
                omega_range = [0.0, self.orange[0]]
            elif len(self.orange) == 2:
                omega_range = [self.orange[0], self.orange[1]]
        
        if self.erange:
            if type(self.erange) is float:
                energy_range = [-self.erange, self.erange]
            elif type(self.erange) is List and len(self.erange) == 2:
                energy_range = [-self.erange[0], self.erange[1]]
        
        if self.band:
            input_text = generate_input_band(input_text, ase_stru, self.kline, self.dim,  self.kmode, self.tolerance,  self.knum,  self.kpath)
        if self.project_band:
            input_text = generate_input_fatband(input_text, ase_stru, self.kline, self.dim, self.kmode, self.tolerance, self.knum, self.kpath,  band_range, nbands)
        if self.spintexture:
            input_text = generate_input_spintexture(input_text, ase_stru, self.kline, self.dim, self.kmode, self.tolerance, self.knum, self.kpath,  band_range, nbands)
        if self.bandunfolding:
            input_text = generate_input_bandunfold(input_text, ase_stru, self.kline, self.dim, self.kmode, self.tolerance, self.knum, self.m_matrix, self.kpath,  band_range, nbands)
        if self.ahc:
            input_text = generate_input_ahc(input_text,  self.dim, lattice_vectors, self.mp)
        if self.anc:
            input_text = generate_input_anc(input_text,  self.dim, lattice_vectors, self.method, self.mp, energy_range)
        if self.berry:
            input_text = generate_input_berry(input_text, ase_stru, noccu_band, self.occu, self.kline, self.dim, self.kmode, self.tolerance, self.knum, self.kpath, self.method, self.mp)
        if self.berry_curvature_dipole:
            input_text = generate_input_bcd(input_text,  self.dim, lattice_vectors,  e_fermi, energy_range)
        if self.cpge:
            input_text = generate_input_bcd(input_text,  self.dim,  lattice_vectors, e_fermi,energy_range)
        if self.pdos:
            input_text = generate_input_pdos(input_text, self.dim, lattice_vectors, e_fermi, energy_range)
        if self.findnodes:
            input_text = generate_input_findnodes(input_text, self.dim, lattice_vectors, e_fermi, energy_range)
        if self.optical:
            input_text = generate_input_optical(input_text, self.dim, lattice_vectors, noccu_band, omega_range, self.mp)
        if self.jdos:
            input_text = generate_input_jdos(input_text, self.dim, lattice_vectors, noccu_band, omega_range, self.mp)
        if self.shift:
            input_text = generate_input_shift(input_text, self.dim, lattice_vectors, noccu_band, omega_range, self.mp)
        if self.chern:
            input_text = generate_input_chern_compatible(input_text,  noccu_band, self.occu, self.dim, lattice_vectors, self.method, self.mp)
        if self.wilson_loop:
            input_text = generate_input_wilsonloop(input_text,  noccu_band, self.occu, self.dim, lattice_vectors, self.method)
        
        self.input_text = input_text
        self.write_pyatb_input()

    def write_pyatb_input(self):
        """
        Write the pyatb input file.
        """
        current_directory = os.path.abspath(os.getcwd())
        if self.output is not None:
            pyatb_directory = os.path.join(self.input, self.output)
            os.makedirs(pyatb_directory, exist_ok=True)
        elif os.path.commonpath([current_directory, self.input]) == self.input and current_directory != self.input:
            pyatb_directory = current_directory
        else:
            pyatb_directory = os.path.join(self.input, "pyatb")
            os.makedirs(pyatb_directory, exist_ok=True)

        input_file_path = os.path.join(pyatb_directory, "Input")
        with open(input_file_path, "w") as file:
            file.write(self.input_text)

    
