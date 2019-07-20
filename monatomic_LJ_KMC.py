# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:09:22 2019

@author: Zarathustra
"""
import numpy as np
import numba as nb
#from numba import jit,njit,types,optional
import scipy

###############################################################################
#
#                    Simulation code for KMC-NVT monatomic LJ
#                            Written by Braden Kelly
#                                 July 19, 2019
#
###############################################################################

""" Brief notes about Kinetic Monte Carlo

1) it has absolutely zero kinetics inside it. None. Zip. Nada.
2) Every configuration has a weight (it turns out the weight is proportional
to the Boltzmann weight - exactly proportional in the case of two particles
overlapping).
3) Every move is accepted.
4) overlaps are allowed

    4b) Because we are sampling ALL phase space(overlaps are allowed), there
    are NO EXCLUDED VOLUMES - reverse widom method is 100% viable and applicable
    
    4c) Chemical Potentials of every particle are calculated in each configuration

    4d) this allows for a good "average" chemical potential of each particle type
    from each configuration, which in turn leads to a STELLAR ensemble average
    over the course of a simulation.
    
    i.e. for 500 particles of 1 type, each configuration is calculating 500
    widom deletions. You then multiply this sampling by the number of configurations
    generated... it is ALOT of widom deletion!
    
    As a perk, this method is only 5-10% slower than MC (taking exponentials 
    to get mobility so you can then choose which particle to move each configuration),
    but is ripe for non-trivial parallelization.
    
5) A translation move is radically different than MC. the particle is moved
    ANYWHERE in the box. Phase space is sampled very quickly. This is fortuitous for
    periodic boundaries... since you know the particle is being placed inside
    the box, you don't need to apply periodic boundaries after a trial move. I 
    have not tried calculated the savings in computational speed due to this.
    
6) The mobility is a relic of the protocol used in real KMC, really, it is 
chemical potential.

"""
###############################################################################
#
#            Simulation Variables (as few as possible)
#
###############################################################################

number_of_atoms = 500
number_of_atom_types = 1# not actually used, this code is for monatomic LJ
atomType = "Ar"
epsilon = 1.0
sigma   = 1.0
boxSize = 8.5
cutOff = boxSize / 2
temperature = 2.0
overLap = 100 / temperature # for numerical reasons, if the pair energy is over this, we cap it (we take exponentials)
nEquilSteps = 10000
outputInterval=100

###############################################################################
#
#       Some Class definitions
#
###############################################################################

class System():
    
    def __init__(self,number_of_atoms, epsilon, sigma, boxSize, temp, cutOff,atomType):
        self.natoms = number_of_atoms
        self.atomType = atomType
        self.eps    = epsilon
        self.sig    = sigma
        self.boxSize = boxSize
        self.volume  = boxSize ** 3
        self.rho     = self.natoms / self.volume
        self.pressure = 0.0
        self.setPressure = None
        self.virial = 0.0
        self.temp = temp
        self.rCut = cutOff
        self.rCut_sq = cutOff ** 2
        self.beta = 1 / temp
        self.positions = np.zeros((self.natoms,3),dtype=np.float_)
        self.velocities = None
        self.forces = None

    def GenerateRandomBox(self):
        self.positions = np.random.rand(self.natoms,3) * self.boxSize
     
    def GetPressure(self):
        self.pressure = 1/ self.beta / self.volume + 1 / 3 / self.volume * self.virial

#@staticmethod
@nb.njit #(nb.int64,nb.float64[:],nb.float64,nb.float64,nb.float64[:,:],nb.float64[:,:], nb.float64,nb.float64, nb.float64)       
def updateEnergies(part, r, box, r_cut_box_sq,energyMatrix, virialMatrix,
                   eps, sig, overlap_sq = 0.75):
    ri = r[part]
    rsq = 0.0
    for index, rj in enumerate(r):
        if index == part: continue
        rij = ri - rj            # Separation vector
        rij = rij - np.rint(rij / box ) * box # Mirror Image Seperation

        rsq = np.dot(rij,rij)  # Squared separation
        if rsq < r_cut_box_sq: # Check within cutoff
            if rsq < overlap_sq: 
                rsq = overlap_sq

            sr2    = sig / rsq    # (sigma/rij)**2
            sr6  = sr2 ** 3
            sr12 = sr6 ** 2
            pot  = 4 * eps * (sr12 - sr6)        # LJ pair potential (cut but not shifted)
            vir  = pot + sr12                    # LJ pair virial
            energyMatrix[part,index] = pot
            energyMatrix[index,part] = pot
            virialMatrix[part,index] = vir
            virialMatrix[index,part] = vir
    return energyMatrix, virialMatrix
              
class KMC_NVT():

    def __init__(self, steps, overLap, system, runType):

        self.nSteps = steps
        self.overLap = overLap
        self.system = system
        self.runType = runType
        self.mobilities = np.zeros((self.system.natoms),dtype=np.float_)
        self.rwEnergies = np.zeros((self.system.natoms,self.system.natoms),dtype=np.float_)
        self.rwVirials = np.zeros((self.system.natoms,self.system.natoms),dtype=np.float_)
        self.totalMobility = 0.0
        print('Beginning {0:s} for {1:d} steps at {2:f} temperature'.format(
                self.runType, self.nSteps, self.system.temp))
        self.OverCut()
        self.InitializeSimulation()

        #######################################################################
        #
        #                          NVT Simulation
        #
        #######################################################################        
        steps = 0
        while steps < self.nSteps:
            steps += 1

            self.UpdateMobilities()
            part = self.PickParticle()
            self.MoveParticle(part)
            self.UpdateEnergies(part)
            if steps % outputInterval == 0: print('Step {} particle {} '.format(steps, part))
            #print('max x coord: {} max y coord {} max z coord {}'.format((self.system.positions[part,0]),self.system.positions[part,1],self.system.positions[part,2]))
        #######################################################################   
    def UpdateMobilities(self):
        self.mobilities = self.CalculateParticleMobilities()
        self.totalMobility = np.sum(self.mobilities)
        
    def InitializeSimulation(self):
        """
        Calculate pairwise energy and virial matrix 
        both are size natoms x natoms (nmolecules x nmolecules in molecular systems)
        """
        for i in range(self.system.natoms):
            self.rwEnergies, self.rwVirials = self.UpdateEnergies(i)
                            
    def OverCut(self):

        self.overLapCut = FindOverLapCut(self.overLap,self.system.sig,
                                         self.system.eps)
        self.overLapCut_sq = self.overLapCut ** 2
        
    def Sample(self):
        raise NotImplementedError  
        
###############################################################################        
    def CalculateParticleMobilities(self):
        natoms = self.system.natoms
        rwEnergies = self.rwEnergies
        beta = self.system.beta
        return self.calculateParticleMobilities(natoms,rwEnergies, beta) # workaround to use @jit
        
    #@staticmethod
    #@nb.jit(nopython=True)
    def calculateParticleMobilities(self, natoms,rwEnergies, beta):
        
        mobilities = np.zeros((natoms), dtype=np.float_)
        
        for i in range(natoms):
            mobilities[i] = np.exp( beta * np.sum( rwEnergies[i,:]) )
        #print('maximum mobility: {} minimum mobility: {} '.format(max(mobilities), min(mobilities)))    
        return mobilities
############################################################################### 

    def UpdateEnergies(self,part):
        r = self.system.positions
        box = self.system.boxSize
        r_cut_box_sq = self.system.rCut_sq
        rwEnergies = self.rwEnergies
        rwVirials = self.rwVirials
        eps = self.system.eps
        sig = self.system.sig
        overlap_sq = self.overLapCut_sq
        return self.updateEnergies(part, r, box, r_cut_box_sq,rwEnergies, rwVirials,
                       eps, sig, overlap_sq )
        
    def updateEnergies(self,part, r, box, r_cut_box_sq,energyMatrix, virialMatrix,
                       eps, sig, overlap_sq = 0.75):
        ri = r[part]
        rsq = 0.0
        """Calculate chosen particle's potential energy with rest of system """
        """Save the pairwise energies in a matrix, same with virial contribution"""
        for index, rj in enumerate(r):
            if index == part: continue
            rij = ri - rj            # Separation vector
            rij = rij - np.rint(rij / box ) * box # Mirror Image Seperation
    
            rsq = np.dot(rij,rij)  # Squared separation
            if rsq < r_cut_box_sq: # Check within cutoff
                if rsq < overlap_sq: 
                    rsq = overlap_sq
    
                sr2    = sig / rsq    # (sigma/rij)**2
                sr6  = sr2 ** 3
                sr12 = sr6 ** 2
                pot  = 4 * eps * (sr12 - sr6)        # LJ pair potential (cut but not shifted)
                vir  = pot + sr12                    # LJ pair virial
                energyMatrix[part,index] = pot
                energyMatrix[index,part] = pot
                virialMatrix[part,index] = vir
                virialMatrix[index,part] = vir
        return energyMatrix, virialMatrix 
###############################################################################
   
    def MoveParticle(self,i):
        self.system.positions[i] = np.random.rand(3) * self.system.boxSize
                             
    def pick_r(self, w ): 
        # pick a particle based on the "rosenbluth" weights
        zeta = np.random.rand() * self.totalMobility 

        k    = 0
        cumw = w[0]
        
        while True: 
            if ( zeta <= cumw ): break # 
            k += 1
            if ( k > len( w ) ): 
                print("Welp, we messed up. Probably forgot to start indexing at 0 for pick_r()")
                exit

            cumw += w[k]
        return k

    def PickParticle(self):
        part = self.pick_r( self.mobilities )
        return part



def LennardJones(rij, sigma, epsilon):
    sr = sigma / rij
    return 4 * epsilon * ( ( sr )**12 - ( sr )**6  ) 
    
def FindOverLapCut(overLap,sig, eps, guess_rij = 0.8):
    return scipy.optimize.fsolve(lambda x: overLap - LennardJones(x,sig, eps), guess_rij) 

def PrintPDB(system,step, name=""):
    f = open(str(name) + "system_step_" + str(step) + ".pdb",'w') 

    for i in range(system.natoms):
        j = []
        j.append( "ATOM".ljust(6) )#atom#6s
        j.append( '{}'.format(str(i+1)).rjust(5) )#aomnum#5d
        j.append( system.atomType.center(4) )#atomname$#4s
        j.append( "MOL".ljust(3) )#resname#1s
        j.append( "A".rjust(1) )#Astring
        j.append( str(i+1).rjust(4) )#resnum
        j.append( str('%8.3f' % (float(system.positions[i][0]))).rjust(8) ) #x
        j.append( str('%8.3f' % (float(system.positions[i][1]))).rjust(8) )#y
        j.append( str('%8.3f' % (float(system.positions[i][2]))).rjust(8) ) #z\
        j.append( str('%6.2f'%(float(1))).rjust(6) )#occ
        j.append( str('%6.2f'%(float(0))).ljust(6) )#temp
        j.append( system.atomType.rjust(12) )#elname  
        #print(i,str(i).rjust(5),j[1], j[2])
        f.write('{}{} {} {} {}{}    {}{}{}{}{}{}\n'.format( j[0],j[1],j[2],j[3],j[4],j[5],j[6],j[7],j[8],j[9],j[10],j[11]))

    f.close()  

        
        
###############################################################################
#
#          Begin Simulation - Equilibrate then Production Run
#
###############################################################################       
        
        
# create a system
phase1 = System(number_of_atoms,epsilon, sigma, boxSize, temperature, cutOff, atomType)
phase1.GenerateRandomBox()

PrintPDB(phase1, 0,"pre_")

equilibrate = KMC_NVT(nEquilSteps,  overLap, phase1, "Equilibration")

PrintPDB(equilibrate.system, equilibrate.nSteps,"equil_")

production1  = KMC_NVT(100000,  overLap, equilibrate.system, "Production1")
#production2  = KMC_NVT(1000,  overLap, production1.system, "Production2")

PrintPDB(production1.system, production1.nSteps,"post_")
print(np.sum(equilibrate.mobilities) )
#print(equilibrate.rwEnergies)
#print(equilibrate.rwVirials)
print(phase1.rho)

"""
Notes - ToDo
add viral pressure
add tail corrections
add sampling of chemical potential
    sample weights
    
"""
