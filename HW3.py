"""
This code is for IMM2019 HW3.
Written by Chun-Chuan Huang
"""
import numpy as np
import math
from scipy import integrate

# First, build up the AO basis set
GF_1s = lambda a, r, R: pow(2*a/np.pi, 0.75)*math.exp(-a*pow(abs(r - R), 2))
phi_He = lambda r: 0.444635*GF_1s(0.480844, r, 0) + 0.535328*GF_1s(1.77669, r, 0) + 0.154329*GF_1s(9.75393, r, 0)
phi_H = lambda r: 0.444635*GF_1s(0.168856, r, 1.4632) + 0.535328*GF_1s(0.623913, r, 1.4632) + \
                  0.154329*GF_1s(3.42525, r, 1.4632)

# Normalized the basis function
N_He, tmp = integrate.quad(lambda r: phi_He(r)*phi_He(r), -np.inf, np.inf)
N_H, tmp = integrate.quad(lambda r: phi_H(r)*phi_H(r), -np.inf, np.inf)

N_He = pow(N_He, 0.5)
N_H = pow(N_H, 0.5)

phi_He = lambda r: (0.444635*GF_1s(0.480844, r, 0) + 0.535328*GF_1s(1.77669, r, 0) + 0.154329*GF_1s(9.75393, r, 0))/N_He
phi_H = lambda r: (0.444635*GF_1s(0.168856, r, 1.4632) + 0.535328*GF_1s(0.623913, r, 1.4632) + \
                  0.154329*GF_1s(3.42525, r, 1.4632))/N_H

# For problem (a)
AO_basis = lambda r: np.array([phi_He(r), phi_H(r)]).reshape([2, 1])
S = lambda r: np.dot(AO_basis(r), AO_basis(r).transpose())
overlap = np.zeros((2, 2))

for i in range(2):
    for j in range(2):
        overlap[i, j], tmp = integrate.quad(lambda r: S(r)[i, j], -np.inf, np.inf)

print(overlap)
print('Program works successfully!! Go buy some beers!!!')
