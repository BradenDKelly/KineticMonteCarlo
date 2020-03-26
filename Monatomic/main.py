import numpy as np
import json, sys

import MMMCGA_auxilary
from MMMCGA_blocks import blk_begin, blk_end, blk_add, run_begin, run_end
from MMMCGA_auxilary import introduction, conclusion, potential, potential_1, PotentialType
from MMMCGA_auxilary import metropolis, random_translate_vector, PrintPDB, InitCubicGrid, pick_r
from MMMCGA_auxilary import random_translate_vector_kmc, potential_2, FindOverLapCut, potential_lrc, pressure_lrc
from MMMCGA_auxilary import pressure_delta

from KolafaNezbeda import ULJ, PressureLJ, FreeEnergyLJ_res, FreeEnergyLJ


def calc_variables():
    """Calculates all variables of interest.
    They are collected and returned as a list, for use in the main program.
    """

    # In this example we simulate using the cut (but not shifted) potential
    # The values of < p_c >, < e_c > and density should be consistent (for this potential)
    # For comparison, long-range corrections are also applied to give
    # estimates of < e_f > and < p_f > for the full (uncut) potential
    # The value of the cut-and-shifted potential is not used, in this example

    import math
    from MMMCGA_auxilary import potential_lrc, pressure_lrc, pressure_delta, VariableType
    # from averages_module import msd, VariableType
    # from lrc_module import potential_lrc, pressure_lrc, pressure_delta
    # from mc_lj_module import force_sq

    # Preliminary calculations (n,r,total are taken from the calling program)
    vol = box ** 3  # Volume
    rho = nAtoms / vol  # Density
    # fsq = force_sq(box, r_cut, r)  # Total squared force

    # Variables of interest, of class VariableType, containing three attributes:
    #   .val: the instantaneous value
    #   .nam: used for headings
    #   .method: indicating averaging method
    # If not set below, .method adopts its default value of avg
    # The .nam and some other attributes need only be defined once, at the start of the program,
    # but for clarity and readability we assign all the values together below

    # Move acceptance ratio
    # m_r = VariableType(nam='Move ratio', val=m_ratio, instant=False)

    # Internal energy per atom for simulated, cut, potential
    # Ideal gas contribution plus cut (but not shifted) PE divided by N
    e_c = VariableType(nam='E/N cut', val=1.5 * temperature + total.pot / nAtoms / total.totalWeight)

    # Internal energy per atom for full potential with LRC
    # LRC plus ideal gas contribution plus cut (but not shifted) PE divided by N
    e_f = VariableType(nam='E/N full',
                       val=potential_lrc(rho, r_cut) + 1.5 * temperature + total.pot / nAtoms / total.totalWeight)

    # Residual energy per atom for full potential with LRC
    # LRC plus cut (but not shifted) PE divided by N
    e_r = VariableType(nam='E/N Residual', val=potential_lrc(rho, r_cut) + total.pot / nAtoms / total.totalWeight)

    # KolafaNezbeda EOS Residual energy per atom for full potential with LRC
    # LRC plus cut (but not shifted) PE divided by N
    eos_r = VariableType(nam='E/N EOS ', val=ULJ(temperature, rho))

    # KolafaNezbeda EOS LJ Chemical Potential with ideal gas contribution
    eos_f = VariableType(nam='mu EOS full ', val=FreeEnergyLJ(temperature, rho))

    # KolafaNezbeda EOS LJ Pressure
    eos_p = VariableType(nam='P EOS ', val=PressureLJ(temperature, rho))

    # KolafaNezbeda EOS LJ Chemical Potential
    eos_mu = VariableType(nam='Chem Pot EOS ', val=FreeEnergyLJ_res(temperature, rho))

    # Pressure for simulated, cut, potential
    # Delta correction plus ideal gas contribution plus total virial divided by V
    p_c = VariableType(nam='P cut',
                       val=pressure_delta(rho, r_cut) + rho * temperature + total.vir / vol / total.totalWeight)

    # Pressure for full potential with LRC
    # LRC plus ideal gas contribution plus total virial divided by V
    p_f = VariableType(nam='P full',
                       val=pressure_lrc(rho, r_cut) + rho * temperature + total.vir / vol / total.totalWeight)

    # chemical potential cut, but not shifted
    # ln( wt/N/W )
    c_c = VariableType(nam='Chem Pot', val=np.log(total.mu / total.totalWeight))

    # Configurational temperature
    # Total squared force divided by total Laplacian
    # t_c = VariableType(nam='T config', val=fsq / total.lap)

    # Heat capacity (full)
    # MSD potential energy divided by temperature and sqrt(N) to make result intensive; LRC does not contribute
    # We add ideal gas contribution, 1.5, afterwards
    c_f = VariableType(nam='Cv/N full', val=total.pot / (temperature * math.sqrt(nAtoms)) / total.totalWeight,
                       method=msd, add=1.5, instant=False)

    # Collect together into a list for averaging
    return [p_c, p_f, e_c, e_f, e_r, c_f, c_c, eos_r, eos_p, eos_mu, eos_f]  # , c_f,t_c,m_r,


# Read parameters in JSON format
try:
    nml = json.load(open("input.inp"))
except json.JSONDecodeError:
    print('Exiting on Invalid JSON format')
    sys.exit()

# Set default values, check keys and typecheck values
defaults = {"nblock": 10, "nstep": 1000, "temperature": 1.0, "r_cut": 2.5, "dr_max": 0.15, \
            "natoms": 256, "initConfig": "crystal", "overlap": 50.0, "rNumSeed":111}

for key, val in nml.items():
    if key in defaults:
        assert type(val) == type(defaults[key]), key + " has the wrong type"
    else:
        print('Warning', key, 'not in ', list(defaults.keys()))

# Set parameters to input values or defaults
nblock = nml["nblock"] if "nblock" in nml else defaults["nblock"]
nSteps = nml["nstep"] if "nstep" in nml else defaults["nstep"]
temperature = nml["temperature"] if "temperature" in nml else defaults["temperature"]
r_cut = nml["r_cut"] if "r_cut" in nml else defaults["r_cut"]
dr_max = nml["dr_max"] if "dr_max" in nml else defaults["dr_max"]
nAtoms = nml["natoms"] if "natoms" in nml else defaults["natoms"]
overlap = nml["overlap"] if "overlap" in nml else defaults["overlap"]
rNumSeed = nml["rNumSeed"] if "rNumSeed" in nml else defaults["rNumSeed"]
epsilon = nml["epsilon"] if "epsilon" in nml else print("no epsilon in input file")
sigma = nml["sigma"] if "sigma" in nml else print("no sigma in input file")
rho = nml["rho"] if "rho" in nml else print("no rho in input file")
outputInterval = nml["outputInterval"] if "outputInterval" in nml else print("no outputInterval in input file")
initialConfiguration = nml["initConf"] if "initConf" in nml else defaults["initConf"]

np.random.seed(rNumSeed)
box = (nAtoms / rho) ** (1 / 3)
if r_cut > box/2:
    r_cut = min(r_cut, box/2)
    print("r_cut too big for box, adjusting: ", r_cut)
overlap_cut = FindOverLapCut(overlap, sigma, epsilon)
print(nSteps, box, dr_max, r_cut)

if "crystal" in initialConfiguration.lower():
    r = InitCubicGrid(nAtoms, rho)
else:
    r = np.random.rand(nAtoms, 3) * box
    print("WARNING: overlap in energy routines do not handle overlap properly. Random init config not advised")
r = r - np.rint(r / box) * box  # Periodic boundaries

avg = 0
msd = 1
cke = 2
PrintPDB(r, 1, "test")
# Initial energy and overlap check
# total = potential_kmc(box, r_cut, r)
# print("PRINTING: ", total.pot, total.vir)
# assert not total.ovr, 'Overlap in initial configuration'

# Initialize arrays for averaging and write column headings
m_trial = 0

eno = np.zeros((nAtoms, nAtoms), dtype=float)
viro = np.zeros((nAtoms, nAtoms), dtype=float)
wt = np.zeros((nAtoms), dtype=float)
W = 0.0
histo = np.zeros((nAtoms), dtype=np.int)

ener = 0.0
vir = 0.0
mu = 0.0
wait = 0.0

for i in range(nAtoms):
    ener_vec, vir_vec = potential_2(r, box, r_cut, overlap_cut, i)
    eno[i, :] = ener_vec
    eno[:, i] = ener_vec
    viro[i, :] = vir_vec
    viro[:, i] = vir_vec
print("energy: ", np.sum(eno) / 2)
wt = np.exp(np.sum(eno, axis=1) / temperature)
W = np.sum(wt)
total = PotentialType(np.sum(eno) / 2, np.sum(viro) / 2, np.average(wt), W)
introduction()
n_avg = run_begin(calc_variables())

for blk in range(1, nblock + 1):  # Loop over blocks

    blk_begin(n_avg)

    for stp in range(nSteps):  # Loop over steps

        moves = 0

        for move in range(nAtoms):  # do nAtoms number of translations
            m_trial += 1

            wt = np.exp(np.sum(eno, axis=1) / temperature)
            W = np.sum(wt)

            part = pick_r(wt)
            histo[part] += 1

            rando = 0
            while rando == 0:
                rando = np.random.rand()
            twait = 1 / W * np.log(1 / rando)
            ener += (np.sum(eno) / 2 * twait)
            vir += (np.sum(viro) / 2 * twait)
            mu += (np.sum(wt) * twait)
            wait += twait

            total = total + PotentialType(np.sum(eno) / 2, np.sum(viro) / 2, np.mean(wt), W)

            ri = random_translate_vector_kmc(box)  # Trial move to new position (in box=1 units)
            r[part, :] = ri  # update position of particle

            ener_vec, vir_vec = potential_2(r, box, r_cut, overlap_cut, part)

            eno[part, :] = ener_vec
            eno[:, part] = ener_vec
            viro[part, :] = vir_vec
            viro[:, part] = vir_vec

        blk_add(calc_variables(), n_avg)
    print(m_trial, \
          "Full energy: ", ener / wait / nAtoms + potential_lrc(rho, r_cut) + 1.5*temperature, \
          "Res energy: ", ener / wait / nAtoms + potential_lrc(rho, r_cut),
          "Full press: ", rho * temperature + box**(-3) * vir / wait + pressure_lrc(rho, r_cut), \
          "Cut press: ", pressure_delta(rho, r_cut) + rho * temperature + box**(-3) * vir / wait, \
          "Res mu: ", np.log(mu / wait / nAtoms) * temperature + 2 * potential_lrc(rho, r_cut), \
          #"Full mu: ", np.log(rho) + np.log(mu / wait) + 2 * potential_lrc(rho, r_cut), \
          "Full mu T: ", (np.log(rho) + np.log(mu / wait / nAtoms))*temperature + 2 * potential_lrc(rho, r_cut), \
          #"Full mu full T: ", (np.log(rho) + np.log(mu / wait) + 2 * potential_lrc(rho, r_cut))*temperature
          )
    if blk == 5:
        ener = 0.0
        vir = 0.0
        mu = 0.0
        wait = 0.0
    PrintPDB(r, stp, "test")
    #blk_end(blk)  # Output block averages
    # sav_tag = str(blk).zfill(3) if blk<1000 else 'sav'    # Number configuration by block
    # write_cnf_atoms ( cnf_prefix+sav_tag, n, box, r*box ) # Save configuration

#run_end(calc_variables())

# total = potential(box, r_cut, r)  # Double check book-keeping
# assert not total.ovr, 'Overlap in final configuration'

# ( cnf_prefix+out_tag, n, box, r*box ) # Save configuration
conclusion()
