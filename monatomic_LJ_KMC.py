# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:09:22 2019

@author: Zarathustra
"""
import numpy as np
import numba as nb
#from numba import jit,njit,types,optional
import scipy

from KolafaNezbeda import ULJ, PressureLJ, FreeEnergyLJ_res
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
number_of_atom_types = 1 #not actually used, this code is for monatomic LJ
atomType = "Ar"
epsilon = 1.0
sigma   = 1.0
boxSize = 8.4
cutOff = boxSize / 2
temperature = 1.0
overLap = 50 / temperature # for numerical reasons, if the pair energy is over this, we cap it (we take exponentials)
nEquilSteps = 100000
outputInterval=2000

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
        self.beta = 1.0 / temp
        self.positions = np.zeros((self.natoms,3),dtype=np.float_)
        self.velocities = None
        self.forces = None

    def GenerateRandomBox(self):
        self.positions = np.random.rand(self.natoms,3) * self.boxSize
     
    def GetPressure(self,virial):
        return self.rho / self.beta  + virial / ( 3.0 * self.volume )
    
    def PressureTailCorrection(self):
        return  16.0 / 3.0 * np.pi * self.rho **2 * self.sig **3 * self.eps * ( (2.0/3.0)*(self.sig / self.rCut)**9 - (self.sig / self.rCut)**3 )
    def EnergyTailCorrection(self):
        return  8.0 / 3.0 * np.pi * self.rho * self.natoms * self.sig **3 * self.eps * ( (1.0/3.0)*(self.sig / self.rCut)**9 - (self.sig / self.rCut)**3 )
    def ChemPotTailCorrection(self):
        """ beta * mu_corr = 2 * u_corr """
        return 16.0 / 3.0 * np.pi * self.rho * self.sig **3 * self.eps * ( (1.0/3.0)*(self.sig / self.rCut)**9 - (self.sig / self.rCut)**3 )

""" THis is outside class since Numba has issues with compiling methods """
@nb.njit #(nb.int64,nb.float64[:],nb.float64,nb.float64,nb.float64[:,:],nb.float64[:,:], nb.float64,nb.float64, nb.float64)       
def updateEnergies(part, r, box, r_cut_box_sq,energyMatrix, virialMatrix,
                   eps, sig, overlap_sq = 0.75):
    ri = r[part]
    rsq = 0.0
    """Calculate chosen particle's potential energy with rest of system """
    """Save the pairwise energies in a matrix, same with virial contribution"""
    #for index, rj in enumerate(r):
    for index in range(0,len(r) ):
        if index == part: continue
        rij = ri - r[index] #rj            # Separation vector
        rij = rij - np.rint(rij / box ) * box # Mirror Image Seperation

        rsq = np.dot(rij,rij)  # Squared separation
        if rsq < r_cut_box_sq: # Check within cutoff
            if rsq < overlap_sq: 
                rsq = overlap_sq

            sr2    = sig / rsq    # (sigma/rij)**2
            sr6  = sr2 ** 3
            sr12 = sr6 ** 2
            pot  = 4.0 * eps * (sr12 - sr6)        # LJ pair potential (cut but not shifted)
            vir  = 24.0 * eps * (2.0 * sr12 - sr6)                    # LJ pair virial
            energyMatrix[part,index] = pot
            energyMatrix[index,part] = pot
            virialMatrix[part,index] = vir
            virialMatrix[index,part] = vir
                            
    return energyMatrix, virialMatrix

@nb.njit
def calculateParticleMobilities(natoms,rwEnergies, beta):
    
    mobilities = np.zeros((natoms), dtype=np.float_)
    
    for i in range(natoms):
        mobilities[i] = np.exp( beta * np.sum( rwEnergies[i,:]) )
        
    
    return mobilities

class KMCSample():
    def __init__(self):
        self.confWeight = 0.0
        self.totalWeight = 0.0
        self.configEnergy = 0.0
        self.configVirial = 0.0
        self.configPressure = 0.0
        self.configChemicalPotential = 0.0
        self.ensembleEnergy = 0.0
        self.ensemblePressure = 0.0
        self.ensembleChemicalPotential = 0.0
              
class KMC_NVT(KMCSample):

    def __init__(self, steps, overLap, system, runType):
        KMCSample.__init__(self)
        self.nSteps = steps
        self.overLap = overLap
        self.system = system
        self.runType = runType
        self.mobilities = np.zeros((self.system.natoms),dtype=np.float_)
        self.rwEnergies = np.zeros((self.system.natoms,self.system.natoms),dtype=np.float_)
        self.rwVirials = np.zeros((self.system.natoms,self.system.natoms),dtype=np.float_)
        self.totalMobility = 0.0
        self.configWeight = 0.0
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
            self.mobilities[np.isnan(self.mobilities)] = 1e30 # a catch incase of overflow
            part = self.PickParticle()
            self.MoveParticle(part)
            self.UpdateEnergies(part)
            
            if steps % outputInterval == 0:
                """
                mu/kT
                U/(NkT)
                """
                chemPot =  np.log( self.ensembleChemicalPotential / self.system.natoms / self.totalWeight )
                pressure = self.ensemblePressure / self.totalWeight
                energy = self.ensembleEnergy / self.totalWeight / self.system.natoms
                print('Step {} particle {} chem pot {:6.4} pressure {:6.4} energy/part {:6.4} LJ results are mu {:6.4} press {:6.4} energy {:6.4}'.
                      format(steps, part, 
                             chemPot+ self.system.ChemPotTailCorrection(),
                             pressure + self.system.PressureTailCorrection(), 
                              energy + (1.0/self.system.natoms) * self.system.EnergyTailCorrection() , 
                             FreeEnergyLJ_res(self.system.temp, self.system.rho) ,
                             PressureLJ(self.system.temp, self.system.rho),
                             ULJ(self.system.temp, self.system.rho)   ) )
                print('Energy tail correction: {} Pressure tail correction {} ChemPot tail correction {}'.format(
                      self.system.EnergyTailCorrection(), self.system.PressureTailCorrection(),self.system.ChemPotTailCorrection()   ) )
            self.Sample()
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
            
    """ given a maximum pair potential energy, find the corresponding distance"""                        
    def OverCut(self):

        self.overLapCut = FindOverLapCut(self.overLap,self.system.sig,
                                         self.system.eps)
        self.overLapCut_sq = self.overLapCut ** 2
        
    def Sample(self):
        r = 0
        while r == 0:
            r = np.random.rand()
        self.configWeight = 1.0 / self.totalMobility * np.log( 1.0 / r )
        self.totalWeight += self.configWeight
        self.configEnergy = np.sum(self.rwEnergies) / 2.0
        self.configVirial = np.sum(self.rwVirials) / 2.0
        self.configPressure = self.system.GetPressure(self.configVirial)
        self.configChemicalPotential = self.totalMobility 
        self.ensembleEnergy += self.configWeight * self.configEnergy
        self.ensemblePressure += self.configWeight * self.configPressure
        self.ensembleChemicalPotential += self.configWeight * self.configChemicalPotential
                                 
        
###############################################################################        
    def CalculateParticleMobilities(self):
        """Calls outside function because it is @njit"""
        natoms = self.system.natoms
        rwEnergies = self.rwEnergies
        beta = self.system.beta
        return calculateParticleMobilities(natoms,rwEnergies, beta) # workaround to use @jit
        
############################################################################### 

    def UpdateEnergies(self,part):
        """Calls outside function because it is @njit"""
        r = self.system.positions
        box = self.system.boxSize
        r_cut_box_sq = self.system.rCut_sq
        rwEnergies = self.rwEnergies
        rwVirials = self.rwVirials
        eps = self.system.eps
        sig = self.system.sig
        overlap_sq = self.overLapCut_sq
        return updateEnergies(part, r, box, r_cut_box_sq,rwEnergies, rwVirials,
                       eps, sig, overlap_sq )

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
            if ( k >= len( w ) ): 
                print("Welp, we messed up. Probably forgot to start indexing at 0 for pick_r()")
                print(len(self.mobilities), np.sum(self.mobilities), cumw,self.totalMobility,zeta )
                exit

            cumw += w[k]
        return k

    def PickParticle(self):
        part = self.pick_r( self.mobilities )
        return part



def LennardJones(rij, sigma, epsilon):
    sr = sigma / rij
    return 4.0 * epsilon * ( ( sr )**12 - ( sr )**6  ) 
    
def FindOverLapCut(overLap,sig, eps, guess_rij = 0.8):
    return float(scipy.optimize.fsolve(lambda x: overLap - LennardJones(x,sig, eps), guess_rij)) 

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

production1  = KMC_NVT(1000000,  overLap, equilibrate.system, "Production1")
#production2  = KMC_NVT(1000,  overLap, production1.system, "Production2")

PrintPDB(production1.system, production1.nSteps,"post_")
print(np.sum(equilibrate.mobilities) )
print(np.sum(production1.mobilities) )
#print(equilibrate.rwEnergies)
#print(equilibrate.rwVirials)
print(phase1.rho)
