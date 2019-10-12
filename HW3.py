"""
This code is for IMM2019 HW3.
"""
import numpy as np
import scipy
from scipy import special

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
    :param Rcp: Distance between centre C and P
    :param Zc: Effective core nuclear charge
    :return:
    """
    ab = a * b
    a_b = a + b
    V = - (2 * pi / (a_b) * F0(a_b * (Rcp ** 2)) * exp(- ab * (Rab ** 2) / a_b)) * Zc

    return V

def F0(t):
    """

    :param t:
    :return:
    """
    if (t < 1e-6):
        return 1 - t / 3
    else:
        return 0.5 * (pi / t) ** 0.5 * special.erf(t ** 0.5)

def ERI_int(a, b, c, d, Rab, Rcd, Rpq):
    """
    This function is used to calculate two electron integrals
    :param a: exponent of Gaussian function which is centered at atom A
    :param b: exponent of Gaussian function which is centered at atom B
    :param c: exponent of Gaussian function which is centered at atom C
    :param d: exponent of Gaussian function which is centered at atom D
    :param Rab: Distance between atom A and B
    :param Rcd: Distance between atom C and D
    :param Rpq: Distance between centre P and Q
    :return:
    """
    eri = 2 * (pi ** 2.5) / ((a + b) * (c + d) * pow(a + b + c + d, 0.5)) * exp(- a * b * (Rab ** 2) / (a + b) -
                                                                                c * d * (Rcd ** 2) / (c + d)) * F0(
        (a + b) * (c + d) * (Rpq ** 2) / (a + b + c + d)
    )

    return eri


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
Va = np.zeros([2, 2])
Vb = np.zeros([2, 2])
Za = 2
Zb = 1

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
        Rap = Expon_H[j] * R / (Expon_He[i] + Expon_H[j])
        Rbp = R - Rap
        Va[0, 0] += V_int(Expon_He[i], Expon_He[j], 0, 0, Za) * Coeff_He[i] * Coeff_He[j]
        Va[0, 1] += V_int(Expon_He[i], Expon_H[j], R, Rap, Za) * Coeff_He[i] * Coeff_H[j]
        Va[1, 1] += V_int(Expon_H[i], Expon_H[j], 0, R, Za) * Coeff_H[i] * Coeff_H[j]
        Va[1, 0] = Va[0, 1]

        Vb[0, 0] += V_int(Expon_He[i], Expon_He[j], 0, R, Zb) * Coeff_He[i] * Coeff_He[j]
        Vb[0, 1] += V_int(Expon_He[i], Expon_H[j], R, Rbp, Zb) * Coeff_He[i] * Coeff_H[j]
        Vb[1, 1] += V_int(Expon_H[i], Expon_H[j], 0, 0, Zb) * Coeff_H[i] * Coeff_H[j]
        Vb[1, 0] = Vb[0, 1]

H_core = T + Va + Vb

'''
print("For problem (a)")
print(S)
print(T)
print(Va)
print(Vb)
'''

# For problem (b)
# Calculate the two electron integrals
ERI = np.zeros([2, 2, 2, 2])

for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                Rap = Expon_H[i] * R / (Expon_H[i] + Expon_He[j])
                Rbp = R - Rap
                Raq = Expon_H[k] * R / (Expon_H[k] + Expon_He[l])
                Rbq = R - Raq
                Rpq = Rap - Raq

                ERI[0, 0, 0, 0] += ERI_int(Expon_He[i], Expon_He[j], Expon_He[k], Expon_He[l], 0, 0, 0) * Coeff_He[i] *\
                                   Coeff_He[j] * Coeff_He[k] * Coeff_He[l]

                ERI[1, 0, 0, 0] += ERI_int(Expon_H[i], Expon_He[j], Expon_He[k], Expon_He[l], R, 0, Rap) * Coeff_H[i] *\
                Coeff_He[j] * Coeff_He[k] * Coeff_He[l]

                ERI[1, 0, 1, 0] += ERI_int(Expon_H[i], Expon_He[j], Expon_H[k], Expon_He[l], R, R, Rpq) * Coeff_H[i] * \
                Coeff_He[j] * Coeff_H[k] * Coeff_He[l]

                ERI[1, 1, 0, 0] += ERI_int(Expon_H[i], Expon_H[j], Expon_He[k], Expon_He[l], 0, 0, R) * Coeff_H[i] * \
                Coeff_H[j] * Coeff_He[k] * Coeff_He[l]

                ERI[1, 1, 1, 0] += ERI_int(Expon_H[i], Expon_H[j], Expon_H[k], Expon_He[l], 0, R, Rbq) * Coeff_H[i] * \
                Coeff_H[j] * Coeff_H[k] * Coeff_He[l]

                ERI[1, 1, 1, 1] += ERI_int(Expon_H[i], Expon_H[j], Expon_H[k], Expon_H[l], 0, 0, 0) * Coeff_H[i] * \
                Coeff_H[j] * Coeff_H[k] * Coeff_H[l]

                ERI[0, 1, 0, 0] = ERI[1, 0, 0, 0]
                ERI[0, 0, 1, 0] = ERI[1, 0, 0, 0]
                ERI[0, 0, 0, 1] = ERI[1, 0, 0, 0]

                ERI[1, 0, 0, 1] = ERI[1, 0, 1, 0]
                ERI[0, 1, 0, 1] = ERI[1, 0, 1, 0]
                ERI[0, 1, 1, 0] = ERI[1, 0, 1, 0]

                ERI[0, 0, 1, 1] = ERI[1, 1, 0, 0]

                ERI[0, 1, 1, 1] = ERI[1, 1, 1, 0]
                ERI[1, 0, 1, 1] = ERI[1, 1, 1, 0]
                ERI[1, 1, 0, 1] = ERI[1, 1, 1, 0]
"""
print("For problem (b):")
print(ERI)
"""

# For problem (c)
toler = 1e-11
MaxIter = 250
Iter = 0

# Initial guess density matrix
P = np.zeros([2, 2])

# Diagonalize the overlap matrix and obtain the transform matrix by X = U(s ** -0.5) U(degar)
Eigen_S, U = np.linalg.eigh(S)
Eigen_S = np.diag(Eigen_S)
X = np.linalg.multi_dot([U,
                         scipy.linalg.fractional_matrix_power(Eigen_S, -0.5),
                         U.transpose().conjugate()])

while Iter < MaxIter:
    Iter += 1
    print("Iter No.:", Iter)
    # Calculate the two electron of the Fock Matrix G = sum(k, l) P[k, l] * (ij|kl) - 0.5(il|kj)
    G = np.zeros([2, 2])

    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    G[i, j] += P[k, l] * (ERI[i, j, k, l] - 0.5 * ERI[i, l, k, j])

    # Construct the Fock matrix by adding H_core and G
    F = H_core + G

    energy_ele = np.sum(0.5 * P * (H_core + F))
    print('Electronic energy = ', energy_ele)

    # Transform the Fock matrix F' = X'FX
    F_prime = np.linalg.multi_dot([X.transpose().conjugate(), F, X])

    # Diagonalize F' to obtain C' and epsilon
    epsilon, C_prime = np.linalg.eigh(F_prime)

    # Transform the C' back to C by C = XC'
    C = np.dot(X, C_prime)

    # Construct a new density matrix from the C by P = 2 C'C
    P_old = P.copy()
    P = np.zeros([2, 2])

    for i in range(2):
        for j in range(2):
            for k in range(1):
                P[i, j] += 2 * C[i, k] * C[j, k]

    # Check whether the procedure is converged by check the density matrix
    Delta = P - P_old
    Delta = np.sqrt(np.sum(Delta ** 2) / 4)
    print("Delta", Delta)

    # Once converged
    if Delta < toler:
        # Add the nuclear repulsion energy
        energy = energy_ele + Za * Zb / R
        print('\nConverged!!!')
        print("Calculation converged with electronic energy:", energy_ele)
        print("Calculation converged with total energy:", energy)
        print('MO Coefficients:\n', C)
        print('Orbital Energies:\n', np.diag(epsilon))

        break


print('Program works successfully!! Go buy some beer!!')
