"""
This code is for IMM2019 HW3.
Written by Chun-Chuan Huang
"""
import numpy as np
from scipy import integrate
import math

# Useful parameters
pi = np.pi
exp = math.exp
zeta1 = 2.0925
zeta2 = 1.24
R = 1.4632
inf = np.inf

# Construct the AO basis
phi_GF = lambda a, r, ra: (2 * a / pi) ** 0.75 * exp(-a * (abs(r - ra) ** 2))
phi_H = lambda r: 0.44635 * phi_GF(0.109818, r, 0) + 0.535328 * phi_GF(0.405771, r, 0) + 0.154329 * phi_GF(2.22766, r, 0)
phi_He = lambda r: 0.44635 * phi_GF(0.109818, r, 0) + 0.535328 * phi_GF(0.405771, r, 0) + 0.154329 * phi_GF(2.22766, r, 0)

'''
# Construct the AO basis
phi = lambda zeta, r, _R:  pow(pow(zeta, 3)/pi, 0.5)*exp(-zeta*(abs(r - _R)**2))
phi_H = lambda r: phi(1.24, r, 0)
phi_He = lambda r: phi(2.0925, r, 0)

# For problem a
# First, Normalization
N_H = pow(integrate.quad(lambda r: phi_H(r)*phi_H(r), -inf, inf)[0], 0.5)
N_He = pow(integrate.quad(lambda r: phi_He(r)*phi_He(r), -inf, inf)[0], 0.5)

phi_H = lambda r: phi(1.24, r, R)/N_H
phi_He = lambda r: phi(2.0925, r, R)/N_He

print(integrate.quad(lambda r: phi_H(r)*phi_H(r), -inf, inf))
print(integrate.quad(lambda r: phi_He(r)*phi_He(r), -inf, inf))

Overlap = pow(pi/(zeta1 + zeta2), 1.5)*exp(-zeta1*zeta2 * (R**2) / (zeta1 + zeta2)) / (N_He * N_H)
'''
print(Overlap)
print('Program works successfully!! Go buy some beer!!')
