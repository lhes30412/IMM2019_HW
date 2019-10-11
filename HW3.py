"""
This code is for IMM2019 HW3.
Written by Chun-Chuan Huang
"""
import numpy as np
import scipy as sp

def S_int(a, b, Rab):
    """
    This function is used to calculate the overlap integral
    :param a: exponent of Gaussian function which is centered at atom A
    :param b: exponent of Gaussian function which is centered at atom B
    :param Rab: Distance between atom A, B
    :return: Unnormalized overlap integral
    """

    S = pow(np.pi/(a + b), 1.5) * np.exp(-a * b * (Rab ** 2) / (a + b))

    return S

def T_int(a, b, Rab):
    """
    This function is used to calculate the kinetic energy integral
    :param a: exponent of Gaussian function which is centered at atom A
    :param b: exponent of Gaussian function which is centered at atom B
    :param Rab: Distance between atom A, B
    :return: kinetic energy integral
    """
    ab = a * b
    a_b = a + b
    T = ab / a_b * (3 - 2 * ab * (Rab ** 2) / a_b) * (pow(np.pi/(a + b), 1.5)) * \
        (np.exp(-a * b * (Rab ** 2) / (a + b)))

    return T

def V_int(a, b, Rab, Rcp, Zc):
    """
    This function is used to calculated the nuclear attraction integrals
    F0 is corresponded to the error function
    :param a: exponent of Gaussian function which is centered at atom A
    :param b: exponent of Gaussian function which is centered at atom B
    :param Rab: Distance between atom A, B
    :param Rcp:
    :param Zc:
    :return:
    """
    ab = a * b
    a_b = a + b
    V = - (2 * pi / (a_b) * F0(a_b  * (Rcp ** 2)) * exp(- ab * Rab ** 2 / a_b)) * Zc

    return V

def F0(t):
    """

    :param t:
    :return:
    """
    if (t < 1e-6):
        return 1 - t / 3
    else:
        return 0.5 * (pi / t) ** 0.5 * sp.erf(t ** 0.5)


# Useful parameters
pi = np.pi
exp = np.exp
Zeta1 = 2.0925
Zeta2 = 1.24
R = 1.4632
inf = np.inf

# Construct the AO basis set
# Standard STO-3G information for zeta = 1.0
Coeff = np.array([0.444635, 0.535328, 0.154329])
Expon = np.array([0.109818, 0.405771, 2.227660])

Expon_He = np.zeros([3])
Expon_H = np.zeros([3])
Coeff_He = np.zeros([3])
Coeff_H = np.zeros([3])


for i in range(3):
    Expon_He[i] = Expon[i] * (Zeta1 ** 2)
    Expon_H[i] = Expon[i] * (Zeta2 ** 2)
    Coeff_He[i] = Coeff[i] * ((2 * Expon_He[i] / pi) ** 0.75)
    Coeff_H[i] = Coeff[i] * ((2 * Expon_H[i] / pi) ** 0.75)

# For problem 1. (a)
# The integrals can be generated as followed
S = np.zeros([2, 2])
T = np.zeros([2, 2])
V = np.zeros([2, 2])

for i in range(3):
    for j in range(3):
        # Overlap matrix
        S[0, 0] += S_int(Expon_He[i], Expon_He[j], 0) * Coeff_He[i] * Coeff_He[j]
        S[0, 1] += S_int(Expon_He[i], Expon_H[j], R) * Coeff_He[i] * Coeff_H[j]
        S[1, 1] += S_int(Expon_H[i], Expon_H[j], 0) * Coeff_H[i] * Coeff_H[j]
        S[1, 0] = S[0, 1]

        # Kinetic Matrix
        T[0, 0] += T_int(Expon_He[i], Expon_He[j], 0) * Coeff_He[i] * Coeff_He[j]
        T[0, 1] += T_int(Expon_He[i], Expon_H[j], R) * Coeff_He[i] * Coeff_H[j]
        T[1, 1] += T_int(Expon_H[i], Expon_H[j], 0) * Coeff_H[i] * Coeff_H[j]
        T[1, 0] = T[0, 1]

        # Nuclear attraction integrals

print(S)
print(T)
print('Program works successfully!! Go buy some beer!!')
