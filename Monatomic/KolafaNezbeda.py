# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:07:20 2019

@author: Zarathustra
"""
# !/usr/bin/env python
# ===================================================================
#      Calculating the thermodynamic properties of the
#      LENNARD-JONES fluid
#
#      J. Kolafa, I. Nezbeda, Fluid Phase Equil. 100 (1994), 1
#
#      ALJ(T,rho)...Helmholtz free energy (including the ideal term)
#      PLJ(T,rho)...Pressure
#      ULJ(T,rho)...Internal energy
#      Run the code like this:
#      python KolafaNezbedaEOS.py [T] [rho_reduced]
# ===================================================================


import math
import sys
import numpy as np

PI = math.pi
pi = 3.141592654
gamma = 1.92907278

CAlj = np.ndarray((7, 7), dtype=float)
CAlj[:, :] = 0
CAlj[0, 2] = 2.01546797
CAlj[0, 3] = -28.17881636
CAlj[0, 4] = 28.28313847
CAlj[0, 5] = -10.42402873
CAlj[1, 2] = -19.58371655
CAlj[1, 3] = 75.62340289
CAlj[1, 4] = -120.70586598
CAlj[1, 5] = 93.92740328
CAlj[1, 6] = -27.37737354
CAlj[2, 2] = 29.34470520
CAlj[2, 3] = -112.35356937
CAlj[2, 4] = 170.64908980
CAlj[2, 5] = -123.06669187
CAlj[2, 6] = 34.42288969
CAlj[4, 2] = -13.37031968
CAlj[4, 3] = 65.38059570
CAlj[4, 4] = -115.09233113
CAlj[4, 5] = 88.91973082
CAlj[4, 6] = -25.62099890

CdhBH = np.ndarray(8, dtype=float)
CdhBH[7] = -0.58544978
CdhBH[6] = 0.43102052
CdhBH[5] = 0.87361369
CdhBH[4] = -4.13749995
CdhBH[3] = 2.90616279
CdhBH[2] = -7.02181962
CdhBH[0] = 0.02459877


def zHS(eta):
    this_zHS = (1.0 + eta * (1.0 + eta * (1.0 - eta / 1.5 * (1.0 + eta)))) / (1.0 - eta) ** 3
    return this_zHS


def betaAHS(eta):
    this_betaAHS = math.log(1.0 - eta) / 0.6 + eta * ((4.0 / 6. * eta - 33.0 / 6.) * eta + 34.0 / 6.) / (1. - eta) ** 2
    return this_betaAHS


# hBH diameter
def dLJ(T):
    isT = 1 / math.sqrt(T)
    this_dLJ = ((0.011117524191338 * isT - 0.076383859168060) * isT) * isT + 0.000693129033539
    this_dLJ = this_dLJ / isT + 1.080142247540047 + 0.127841935018828 * math.log(isT)
    return this_dLJ


def dC(T):
    sT = math.sqrt(T)
    this_dC = -0.063920968 * math.log(T) + 0.011117524 / T - 0.076383859 / sT + 1.080142248 + 0.000693129 * sT
    return this_dC


def dCdT(T):
    sT = math.sqrt(T)
    this_dCdT = 0.063920968 * T + 0.011117524 + (-0.5 * 0.076383859 - 0.5 * 0.000693129 * T) * sT
    return this_dCdT


def gammaBH(X):
    return 1.92907278


def DALJ(T, rho):
    sum1 = 2.01546797 + rho * (-28.17881636 + rho * (+28.28313847 + rho * (-10.42402873)))
    sum2 = -19.58371655 + rho * (75.62340289 + rho * ((-120.70586598) + rho * (93.92740328 + rho * (-27.37737354))))
    sum2 = sum2 / math.sqrt(T)
    sum3 = 29.34470520 + rho * ((-112.35356937) + rho * (+170.64908980 + rho * ((-123.06669187) + rho * 34.42288969)))
    sum4 = -13.37031968 + rho * (65.38059570 + rho * ((-115.09233113) + rho * (88.91973082 + rho * (-25.62099890))))
    sum4 = sum4 / T
    this_DALJ = (sum1 + sum2 + (sum3 + sum4) / T) * rho * rho
    return this_DALJ


# Test from fortran
def DALJ2(T, rho):
    DALJ = ((+2.01546797 + rho * (-28.17881636 + rho * (+28.28313847 + rho * (-10.42402873))))
            + (-19.58371655 + rho * (75.62340289 + rho * (
                    (-120.70586598) + rho * (93.92740328 + rho * (-27.37737354))))) / math.sqrt(T)
            + ((29.34470520 + rho * ((-112.35356937) + rho * (+170.64908980 + rho * ((-123.06669187)
                                                                                     + rho * 34.42288969)))) + (
                       -13.37031968 + rho * (65.38059570 + rho * ((-115.09233113) + rho * (88.91973082
                                                                                           + rho * (
                                                                                               -25.62099890))))) / T) / T) * rho * rho
    return DALJ


# test from Jiri
def sumCALJ(T, rho):
    sum = 0
    for i in range(-4, 1):
        if (i == -3): continue
        for j in range(2, 7):
            if (i == 0 and j == 6): continue
            sum += CAlj[abs(i), j] * T ** (i / 2.0) * rho ** j
    return sum


def AHS(eta, T):
    return T * (5.0 / 3.0 * math.log(1.0 - eta) + eta * (34.0 - 33 * eta + 4.0 * eta * eta) / (6.0 * (1.0 - eta) ** 2))


def dB2hBH(T):
    sum = 0
    for i in range(-7, 1):
        if (i == -1): continue
        sum += CdhBH[abs(i)] * T ** (i / 2.0)
    return sum


def ALJres(T, rho):
    eta = pi / 6.0 * rho * (dC(T)) ** 3
    return (AHS(eta, T) + rho * T * dB2hBH(T) * math.exp(-gamma * rho ** 2)) + sumCALJ(T, rho)


####################################################

def BC(T):
    isT = 1 / math.sqrt(T)
    sum1 = (((-0.58544978 * isT + 0.43102052) * isT + .87361369) * isT - 4.13749995) * isT + 2.90616279
    this_BC = (sum1 * isT - 7.02181962) / T + 0.02459877
    return this_BC


def BCdT(T):
    isT = 1 / math.sqrt(T)
    sum1 = ((-0.58544978 * 3.5 * isT + 0.43102052 * 3) * isT + 0.87361369 * 2.5) * isT - 4.13749995 * 2.
    this_BCdT = (sum1 * isT + 2.90616279 * 1.5) * isT - 7.02181962
    return this_BCdT


# internal energy
def ULJ(T, rho):
    dBHdT = dCdT(T)
    dB2BHdT = BCdT(T)
    d = dC(T)
    eta = PI / 6.0 * rho * d ** 3
    sum1 = 2.01546797 + rho * ((-28.17881636) + rho * (+28.28313847 + rho * (-10.42402873)))
    sum2 = -19.58371655 * 1.5 + rho * (75.62340289 * 1.5 + rho * (
            (-120.70586598) * 1.5 + rho * (93.92740328 * 1.5 + rho * (-27.37737354) * 1.5)))
    sum2 = sum2 / math.sqrt(T)
    sum3 = 29.34470520 * 2. + rho * (
            -112.35356937 * 2. + rho * (170.64908980 * 2. + rho * (-123.06669187 * 2. + rho * 34.42288969 * 2.)))
    sum4 = -13.37031968 * 3. + rho * (
            65.38059570 * 3. + rho * (-115.09233113 * 3. + rho * (88.91973082 * 3. + rho * (-25.62099890) * 3.)))
    sum4 = sum4 / T
    sum5 = (sum1 + sum2 + (sum3 + sum4) / T) * rho * rho
    thisULJ = 3. * (zHS(eta) - 1) * dBHdT / d + rho * dB2BHdT / math.exp(gammaBH(T) * rho ** 2) + sum5
    return thisULJ


#  Helmholtz free energy (including the ideal term)
def FreeEnergyLJ(T, rho):
    eta = PI / 6.0 * rho * dC(T) ** 3
    ALJ = math.log(rho) + betaAHS(eta) + rho * BC(T) / math.exp(gammaBH(T) * rho ** 2)
    ALJ = ALJ * T + DALJ(T, rho)
    return ALJ


# Helmholtz free energy (without ideal term)
def FreeEnergyLJ_res(T, rho):
    eta = PI / 6.0 * rho * dC(T) ** 3
    ALJres = (betaAHS(eta) + rho * BC(T) / math.exp(gammaBH(T) * rho ** 2)) * T + DALJ(T, rho)
    return ALJres


# pressure
def PressureLJ(T, rho):
    eta = PI / 6. * rho * (dC(T)) ** 3
    sum1 = 2.01546797 * 2 + rho * (-28.17881636 * 3 + rho * (28.28313847 * 4 + rho * (-10.42402873) * 5))
    sum2 = -19.58371655 * 2
    sum2 = sum2 + rho * (
            75.62340289 * 3 + rho * (-120.70586598 * 4 + rho * (93.92740328 * 5 + rho * (-27.37737354) * 6)))
    sum2 = sum2 / math.sqrt(T)
    sum3 = 29.34470520 * 2. + rho * ((-112.35356937) * 3. + rho * (
            +170.64908980 * 4. + rho * ((-123.06669187) * 5. + rho * 34.42288969 * 6.)))
    sum4 = -13.37031968 * 2. + rho * (
            65.38059570 * 3. + rho * (-115.09233113 * 4. + rho * (88.91973082 * 5. + rho * (-25.62099890) * 6.)))
    sum4 = sum4 / T
    sum5 = (sum1 + sum2 + (sum3 + sum4) / T) * rho ** 2
    PLJ = ((zHS(eta) + BC(T) / math.exp(gammaBH(T) * rho ** 2) * rho * (
            1.0 - 2 * gammaBH(T) * rho ** 2)) * T + sum5) * rho
    return PLJ


T = 1.0  # float(sys.argv[1])
rho = 0.8  # float(sys.argv[2])
num = 400
eta = PI / 6.0 * rho * dC(T) ** 3
Pressure = PressureLJ(T, rho)
Energy = ULJ(T, rho)
FreeEnergy = FreeEnergyLJ_res(T, rho)
FreeEnergy2 = FreeEnergyLJ(T, rho)
"""
print("-------------------------------------------------------------------")
print("Kolafa-Nezbeda Equation of states: ")
print("Reduced Pressure at (T = %.2f, rho = %.2f) is : %.6f "%(T, rho, Pressure))
print("Reduced Energy at (T = %.2f, rho = %.2f) is : %.6f "%(T, rho, Energy))
print("Reduced chem Pot at (T = %.2f, rho = %.2f) is : %.6f "%(T, rho, FreeEnergy))
print("Reduced chem Pot at (T = %.2f, rho = %.2f) is : %.6f "%(T, rho, FreeEnergy2))
print("-------------------------------------------------------------------")

print("Reduced DALJ mine at %.6f "%(DALJ(T,rho)))
print("Reduced DALJ fortran  %.6f "%(DALJ2(T,rho)))
print("Reduced DALJ Jiri %.6f "%(sumCALJ(T,rho)))

print("Reduced BC(T) mine at %.6f "%(BC(T)))
#print("Reduced DALJ fortran  %.6f "%(DALJ2(T,rho)))
print("Reduced dB2hBH Jiri %.6f "%(dB2hBH(T)))

print("Reduced AHS mine at %.6f "%(betaAHS(eta)))
#print("Reduced DALJ fortran  %.6f "%(DALJ2(T,rho)))
print("Reduced betaHS Jiri %.6f "%(AHS(eta,T)))

print("Reduced ALJres mine at %.6f "%(FreeEnergyLJ_res(T,rho)))
#print("Reduced DALJ fortran  %.6f "%(DALJ2(T,rho)))
print("Reduced ALJres Jiri %.6f "%(ALJres(T,rho)))
"""
