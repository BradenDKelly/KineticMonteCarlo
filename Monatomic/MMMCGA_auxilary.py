fast = True
slow = False
jit = True
avg = 0
msd = 1
cke = 2
import numba as nb
import numpy as np


class PotentialType:
    """A composite variable for interactions."""

    def __init__(self, pot, vir, mu, weight):
        r = 0
        while r == 0:
            r = np.random.rand()
        weight = 1 / weight * np.log(1.0 / r)
        self.pot = pot * weight   # potential energies
        self.vir = vir * weight   # virial energies
        self.mu = mu * weight     # chemical potential
        self.totalWeight = weight

    def __add__(self, other):
        pot = self.pot + other.pot
        vir = self.vir + other.vir
        mu = self.mu + other.mu
        totalWeight = self.totalWeight + other.totalWeight
        return PotentialType(pot, vir, mu, totalWeight)

    def __sub__(self, other):
        pot = self.pot - other.pot
        vir = self.vir - other.vir
        mu = self.mu - other.mu
        totalWeight = self.totalWeight - other.totalWeight
        return PotentialType(pot, vir, mu, totalWeight)


def introduction():
    """Prints out introductory statements at start of run."""

    print('Lennard-Jones potential')
    print('Cut (but not shifted)')
    print('Diameter, sigma = 1')
    print('Well depth, epsilon = 1')
    if fast:
        print('Fast NumPy potential routine')
    elif slow:
        print('Slow Python potential routine')
    elif jit:
        print('Using Numba for maximum speed')


def conclusion():
    """Prints out concluding statements at end of run."""

    print('Program ends')


def potential(box, r_cut, r):
    """Takes in box, cutoff range, and coordinate array, and calculates total potential etc.
    The results are returned as total, a PotentialType variable.
    """
    # Actual calculation performed by function potential_1

    n, d = r.shape
    assert d == 3, 'Dimension error for r in potential'

    total = PotentialType(pot=0.0, vir=0.0, ovr=False)

    for i in range(n - 1):
        partial = potential_1(r[i, :], box, r_cut, r[i + 1:, :])

        total = total + partial

    return total

def potential_kmc(box, r_cut, r):
    """Takes in box, cutoff range, and coordinate array, and calculates total potential etc.
    The results are returned as total, a PotentialType variable.
    """
    # Actual calculation performed by function potential_1

    n, d = r.shape
    assert d == 3, 'Dimension error for r in potential'

    total = PotentialType(pot=0.0, vir=0.0, ovr=False)

    for i in range(n):
        rj = np.delete(r,i,0)
        partial = potential_2(r[i, :], box, r_cut, rj)

        total = total + partial

    return total


def potential_1(ri, box, r_cut, r):
    """Takes in coordinates of an atom and calculates its interactions.
    Values of box, cutoff range, and partner coordinate array are supplied.
    The results are returned as partial, a PotentialType variable.
    """

    import numpy as np

    # partial.pot is the nonbonded cut (not shifted) potential energy of atom ri with a set of other atoms
    # partial.vir is the corresponding virial of atom ri
    # partial.lap is the corresponding Laplacian of atom ri
    # partial.ovr is a flag indicating overlap (potential too high) to avoid overflow
    # If this is True, the values of partial.pot etc should not be used
    # In general, r will be a subset of the complete set of simulation coordinates
    # and none of its rows should be identical to ri

    # It is assumed that positions are in units where box = 1
    # Forces are calculated in units where sigma = 1 and epsilon = 1

    nj, d = r.shape
    assert d == 3, 'Dimension error for r in potential_1'
    assert ri.size == 3, 'Dimension error for ri in potential_1'

    sr2_ovr = 1.77  # Overlap threshold (pot > 100)
    r_cut_box = r_cut  # / box
    r_cut_box_sq = r_cut_box ** 2
    # box_sq = box ** 2

    if fast:
        rij = ri - r  # Get all separation vectors from partners
        rij = rij - np.rint(rij / box) * box  # Periodic boundary conditions in box=1 units
        rij_sq = np.sum(rij ** 2, axis=1)  # Squared separations
        in_range = rij_sq < r_cut_box_sq  # Set flags for within cutoff
        # rij_sq = rij_sq  * box_sq                          # Now in sigma=1 units
        sr2 = np.where(in_range, 1.0 / rij_sq, 0.0)  # (sigma/rij)**2, only if in range
        ovr = sr2 > sr2_ovr  # Set flags for any overlaps  np.sqrt(rij_sq) < 0.75 #
        if np.any(ovr):
            partial = PotentialType(pot=0.0, vir=0.0, ovr=True)
            return partial
        sr6 = sr2 ** 3
        sr12 = sr6 ** 2
        pot = sr12 - sr6  # LJ pair potentials (cut but not shifted)
        vir = pot + sr12  # LJ pair virials
        partial = PotentialType(pot=np.sum(pot), vir=np.sum(vir), ovr=False)

    elif slow:
        partial = PotentialType(pot=0.0, vir=0.0, ovr=False)
        for rj in r:
            rij = ri - rj  # Separation vector
            rij = rij - np.rint(rij / box) * box  # Periodic boundary conditions in box=1 units
            rij_sq = np.sum(rij ** 2)  # Squared separation
            if rij_sq < r_cut_box_sq:  # Check within cutoff
                # rij_sq = rij_sq #* box_sq  # Now in sigma=1 units
                sr2 = 1.0 / rij_sq  # (sigma/rij)**2
                ovr = sr2 > sr2_ovr  # Overlap if too close
                if ovr:
                    partial.ovr = True
                    return partial
                sr6 = sr2 ** 3
                sr12 = sr6 ** 2
                pot = sr12 - sr6  # LJ pair potential (cut but not shifted)
                vir = pot + sr12  # LJ pair virial
                # lap = (22.0 * sr12 - 5.0 * sr6) * sr2  # LJ pair Laplacian
                partial = partial + PotentialType(pot=pot, vir=vir, ovr=ovr)
    elif jit:
        pot, vir, ovr = FastEnergy(sr2_ovr, r_cut_box_sq, r, ri, box)
        partial = PotentialType(pot=pot, vir=vir, ovr=ovr)
        # partial = PotentialType(pot=0.0, vir=0.0, ovr=False)
    else:
        print("no energy method has been selected for pair energies")

    # Multiply results by numerical factors
    partial.pot = partial.pot * 4.0  # 4*epsilon
    partial.vir = partial.vir * 24.0 / 3.0  # 24*epsilon and divide virial by 3
    # partial.lap = partial.lap * 24.0 * 2.0  # 24*epsilon and factor 2 for ij and ji

    return partial

def potential_2(r, box, r_cut, overlap, part):

    """Takes in coordinates of an atom and calculates its interactions.
    Values of box, cutoff range, and partner coordinate array are supplied.
    The results are returned as partial, a PotentialType variable.
    """

    import numpy as np

    # I have verified with MC code, which matches LJ EOS, that this energy
    # is correct for starting config of a crystal
    sr2_ovr = 1.77  # Overlap threshold (pot > 100)
    overlap_sq = overlap**2
    r_cut_box_sq = r_cut**2

    ri = r[part]
    rj=np.delete(r,part,0)

    nj, d = rj.shape
    assert d == 3, 'Dimension error for r in potential_1'
    assert ri.size == 3, 'Dimension error for ri in potential_1'

    rsq = 0.0
    """Calculate chosen particle's potential energy with rest of system """
    """Save the pairwise energies in a matrix, same with virial contribution"""
    rij = ri - rj  # Get all separation vectors from partners
    rij = rij - np.rint(rij / box) * box  # Periodic boundary conditions in box=1 units
    rij_sq = np.sum(rij ** 2, axis=1)  # Squared separations
    in_range = rij_sq < r_cut_box_sq  # Set flags for within cutoff
    in_overlap = rij_sq < overlap_sq
    sr2 = np.where(in_range, 1.0 / rij_sq, 0.0)  # (sigma/rij)**2, only if in range
    sr2 = np.where(in_overlap,1.0 / overlap_sq, sr2)

    sr6 = sr2 ** 3
    sr12 = sr6 ** 2
    pot = sr12 - sr6  # LJ pair potentials (cut but not shifted)
    vir = pot + sr12  # LJ pair virials

    pot = 4.0 * np.insert(pot, part, 0.0)
    vir = 24.0 / 3.0 * np.insert(vir, part, 0.0)

    return pot, vir

@nb.njit
def FastEnergy(sr2_ovr, r_cut_box_sq, p, pi, box):
    vir = 0.0
    pot = 0.0
    ovr = False
    for i in range(len(p)):
        rij = p[i] - pi  # Separation vector
        rij = rij - np.rint(rij / box) * box  # Periodic boundary conditions in box=1 units
        rij_sq = np.sum(rij ** 2)  # Squared separation

        # rij = np.abs(p[i] - pi )
        # rij = np.minimum(rij, box - rij)
        # rij_sq = np.sum(rij ** 2)

        if rij_sq < r_cut_box_sq:  # Check within cutoff
            # rij_sq = rij_sq  # * box_sq  # Now in sigma=1 units
            sr2 = 1.0 / rij_sq  # (sigma/rij)**2
            ovr = sr2 > sr2_ovr  # Overlap if too close
            if ovr:  # sr2 > sr2_ovr:
                return pot, vir, True

            sr6 = sr2 ** 3
            sr12 = sr6 ** 2
            pot += (sr12 - sr6)  # LJ pair potential (cut but not shifted)
            vir += (2 * sr12 - sr6)  # LJ pair virial
            # lap = (22.0 * sr12 - 5.0 * sr6) * sr2  # LJ pair Laplacian
            # partial = partial + PotentialType(pot=pot, vir=vir, ovr=ovr)
    return pot, vir, ovr


def metropolis(delta):
    """Conduct Metropolis test, with safeguards."""

    import numpy as np

    exponent_guard = 75.0

    if delta > exponent_guard:  # Too high, reject without evaluating
        return False
    elif delta < 0.0:  # Downhill, accept without evaluating
        return True
    else:
        zeta = np.random.rand()  # Uniform random number in range (0,1)
        return np.exp(-delta) > zeta  # Metropolis test


class VariableType:
    """Class encapsulating the essential information for simulation averages."""

    def __init__(self, nam, val, method=avg, add=0.0, e_format=False, instant=True):
        self.nam = nam
        self.val = val
        self.method = method
        self.add = add
        self.e_format = e_format
        self.instant = instant


def time_stamp():
    """Function to print date, time, and cpu time information."""

    import time

    print("{:45}{}".format("Date:", time.strftime("%Y/%m/%d")))
    print("{:47}{}".format("Time:", time.strftime("%H:%M:%S")))
    print("{:40}{:15.6f}".format("CPU time:", time.process_time()))


def potential_lrc(density, r_cut):
    """Calculates long-range correction for Lennard-Jones potential per atom."""

    import math

    # density, r_cut, and the results, are in LJ units where sigma = 1, epsilon = 1
    sr3 = 1.0 / r_cut ** 3
    return math.pi * ((8.0 / 9.0) * sr3 ** 3 - (8.0 / 3.0) * sr3) * density


def pressure_lrc(density, r_cut):
    """Calculates long-range correction for Lennard-Jones pressure."""

    import math

    # density, r_cut, and the results, are in LJ units where sigma = 1, epsilon = 1
    sr3 = 1.0 / r_cut ** 3
    return math.pi * ((32.0 / 9.0) * sr3 ** 3 - (16.0 / 3.0) * sr3) * density ** 2


def pressure_delta(density, r_cut):
    """Calculates correction for Lennard-Jones pressure due to discontinuity in the potential at r_cut."""

    import math

    # density, r_cut, and the results, are in LJ units where sigma = 1, epsilon = 1
    sr3 = 1.0 / r_cut ** 3
    return math.pi * (8.0 / 3.0) * (sr3 ** 3 - sr3) * density ** 2


def random_translate_vector(dr_max, old):
    """Returns a vector translated by a random amount."""

    import numpy as np

    # A randomly chosen vector is added to the old one

    zeta = np.random.rand(3)  # Three uniform random numbers in range (0,1)
    zeta = 2 * zeta - 1.0  # Now in range (-0.5,+0.5)
    return old + zeta * dr_max  # Move to new position

def random_translate_vector_kmc(box):
    """Returns a vector translated by a random amount."""

    import numpy as np

    # A randomly chosen vector is added to the old one

    zeta = np.random.rand(3)  # Three uniform random numbers in range (0,1)
    zeta = 2 * zeta - 1.0  # Now in range (-1,+1)
    return zeta * box/2  # Move to new position


def PrintPDB(r, step, name=""):
    f = open(str(name) + "system_step_" + str(step) + ".pdb", 'w')

    for i in range(len(r)):
        j = []
        j.append("ATOM".ljust(6))  # atom#6s
        j.append('{}'.format(str(i + 1)).rjust(5))  # aomnum#5d
        j.append("fill".center(4))  # atomname$#4s
        j.append("MOL".ljust(3))  # resname#1s
        j.append("A".rjust(1))  # Astring
        j.append(str(i + 1).rjust(4))  # resnum
        j.append(str('%8.3f' % (float(r[i][0]))).rjust(8))  # x
        j.append(str('%8.3f' % (float(r[i][1]))).rjust(8))  # y
        j.append(str('%8.3f' % (float(r[i][2]))).rjust(8))  # z\
        j.append(str('%6.2f' % (float(1))).rjust(6))  # occ
        j.append(str('%6.2f' % (float(0))).ljust(6))  # temp
        j.append("fill".rjust(12))  # elname
        # print(i,str(i).rjust(5),j[1], j[2])
        f.write('{}{} {} {} {}{}    {}{}{}{}{}{}\n'.format(j[0], j[1], j[2], j[3], j[4], j[5], j[6], j[7], j[8], j[9],
                                                           j[10], j[11]))

    f.close()


def InitCubicGrid(nMolInit, initDensity):
    """
    - Assign molecule COM's to a simple cubic lattice.
    - create entries in class "Mol" list, titled "mol"
    """
    rMol = np.zeros((nMolInit, 3), dtype=np.float_)
    box = (nMolInit / initDensity) ** (1.0 / 3.0)
    nCube = 2
    initConfig = 'centered'

    while nCube ** 3 < nMolInit:
        nCube += 1
        # initial position of particle 1
        posit = np.zeros((3))

        # begin assigning particle positions
        for i in range(nMolInit):
            coords = (posit + [0.5, 0.5, 0.5]) * (box / nCube)
            if 'centered' in initConfig:
                coords = coords - box / 2
            rMol[i, :] = coords

            # Advancing the index (posit)
            posit[0] += 1
            if posit[0] == nCube:
                posit[0] = 0
                posit[1] += 1

                if posit[1] == nCube:
                    posit[1] = 0
                    posit[2] += 1

    return np.asarray(rMol)

def pick_r(w):
    # pick a particle based on the "rosenbluth" weights
    zeta = np.random.rand() * np.sum(w) #self.totalMobility

    k = 0
    cumw = w[0]

    while True:
        if (zeta <= cumw): break  #
        k += 1
        if (k >= len(w)):
            print("Welp, we messed up. Probably forgot to start indexing at 0 for pick_r()")
            exit

        cumw += w[k]
    return k

def FindOverLapCut(overLap, sig, eps, guess_rij=0.6):
    import scipy
    import scipy.optimize
    ovr=float(scipy.optimize.fsolve(lambda x: overLap - LennardJones(x, sig, eps), guess_rij))
    print(ovr)
    return ovr

def LennardJones(rij, sigma, epsilon):
    sr = sigma / rij
    return 4.0 * epsilon * ((sr) ** 12 - (sr) ** 6)