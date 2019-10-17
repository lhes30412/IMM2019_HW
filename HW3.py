"""
This code is for IMM2019 HW3.
With references:
Basic Structure:
http://nznano.blogspot.com/2018/03/simple-quantum-chemistry-hartree-fock.html
[Szabo:1996] appendix B
Revised into numpy:
https://github.com/psi4/psi4numpy/blob/master/Self-Consistent-Field/RHF.py
"""
import numpy as np
import scipy
from scipy import special
import matplotlib.pyplot as plt

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
    :param Rab: Distance between atoms
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
    The error function need in calculating electron interaction
    Note that when t close to zero, we replace the result with 1 - t / 3 followed the method in Szabo to prevent inf / 0
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
    (phi_A phi_B|phi_C phi_D)
    :param a: exponent of Gaussian function which is centered at A
    :param b: exponent of Gaussian function which is centered at B
    :param c: exponent of Gaussian function which is centered at C
    :param d: exponent of Gaussian function which is centered at D
    :param Rab: Distance between A and B
    :param Rcd: Distance between C and D
    :param Rpq: Distance between centre P and Q
    :return:
    """
    eri = 2 * (pi ** 2.5) / ((a + b) * (c + d) * pow(a + b + c + d, 0.5)) * exp(- a * b * (Rab ** 2) / (a + b) -
                                                                                c * d * (Rcd ** 2) / (c + d)) * F0(
        (a + b) * (c + d) * (Rpq ** 2) / (a + b + c + d)
    )

    return eri


def HFSCF(_r, molecular):
    """
    This function is used to perform HF-SCF
    :param _r: Distance between the atoms
    :return:
    """

    # Construct the AO basis set
    # Standard STO-3G information for zeta = 1.0
    Coeff = np.array([0.444635, 0.535328, 0.154329])
    Expon_He = np.array([0.480844, 1.77669, 9.75393])
    Expon_H = np.array([0.168856, 0.623913, 3.42525])

    Coeff_He = np.zeros([3])
    Coeff_H = np.zeros([3])


    for i in range(3):
        Coeff_He[i] = Coeff[i] * ((2 * Expon_He[i] / pi) ** 0.75)
        Coeff_H[i] = Coeff[i] * ((2 * Expon_H[i] / pi) ** 0.75)

    # For problem a-f, we need HeH+
    if molecular == 'HeH+':
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
                S[0, 1] += S_int(Expon_He[i], Expon_H[j], _r) * Coeff_He[i] * Coeff_H[j]
                S[1, 1] += S_int(Expon_H[i], Expon_H[j], 0) * Coeff_H[i] * Coeff_H[j]
                S[1, 0] = S[0, 1]

                # Kinetic Matrix
                T[0, 0] += T_int(Expon_He[i], Expon_He[j], 0) * Coeff_He[i] * Coeff_He[j]
                T[0, 1] += T_int(Expon_He[i], Expon_H[j], _r) * Coeff_He[i] * Coeff_H[j]
                T[1, 1] += T_int(Expon_H[i], Expon_H[j], 0) * Coeff_H[i] * Coeff_H[j]
                T[1, 0] = T[0, 1]

                # Nuclear attraction integrals
                Rap = Expon_H[j] * _r / (Expon_He[i] + Expon_H[j])
                Rbp = _r - Rap
                Va[0, 0] += V_int(Expon_He[i], Expon_He[j], 0, 0, Za) * Coeff_He[i] * Coeff_He[j]
                Va[0, 1] += V_int(Expon_He[i], Expon_H[j], _r, Rap, Za) * Coeff_He[i] * Coeff_H[j]
                Va[1, 1] += V_int(Expon_H[i], Expon_H[j], 0, _r, Za) * Coeff_H[i] * Coeff_H[j]
                Va[1, 0] = Va[0, 1]

                Vb[0, 0] += V_int(Expon_He[i], Expon_He[j], 0, _r, Zb) * Coeff_He[i] * Coeff_He[j]
                Vb[0, 1] += V_int(Expon_He[i], Expon_H[j], _r, Rbp, Zb) * Coeff_He[i] * Coeff_H[j]
                Vb[1, 1] += V_int(Expon_H[i], Expon_H[j], 0, 0, Zb) * Coeff_H[i] * Coeff_H[j]
                Vb[1, 0] = Vb[0, 1]

        H_core = T + Va + Vb

        # For problem (b)
        # Calculate the two electron integrals
        ERI = np.zeros([2, 2, 2, 2])

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        Rap = Expon_H[i] * _r / (Expon_H[i] + Expon_He[j])
                        Raq = Expon_H[k] * _r / (Expon_H[k] + Expon_He[l])
                        Rbq = _r - Raq
                        Rpq = Rap - Raq

                        ERI[0, 0, 0, 0] += ERI_int(Expon_He[i], Expon_He[j], Expon_He[k], Expon_He[l], 0, 0, 0) * Coeff_He[i] *\
                                           Coeff_He[j] * Coeff_He[k] * Coeff_He[l]

                        ERI[1, 0, 0, 0] += ERI_int(Expon_H[i], Expon_He[j], Expon_He[k], Expon_He[l], _r, 0, Rap) * Coeff_H[i] *\
                        Coeff_He[j] * Coeff_He[k] * Coeff_He[l]

                        ERI[1, 0, 1, 0] += ERI_int(Expon_H[i], Expon_He[j], Expon_H[k], Expon_He[l], _r, _r, Rpq) * Coeff_H[i] * \
                        Coeff_He[j] * Coeff_H[k] * Coeff_He[l]

                        ERI[1, 1, 0, 0] += ERI_int(Expon_H[i], Expon_H[j], Expon_He[k], Expon_He[l], 0, 0, _r) * Coeff_H[i] * \
                        Coeff_H[j] * Coeff_He[k] * Coeff_He[l]

                        ERI[1, 1, 1, 0] += ERI_int(Expon_H[i], Expon_H[j], Expon_H[k], Expon_He[l], 0, _r, Rbq) * Coeff_H[i] * \
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

        '''
        print("For problem (b):")
        print('(11|11): ', ERI[0, 0, 0, 0])
        print('(21|11): ', ERI[1, 0, 0, 0])
        print('(21|21): ', ERI[1, 0, 1, 0])
        print('(22|11): ', ERI[1, 1, 0, 0])
        print('(22|21): ', ERI[1, 1, 1, 0])
        print('(22|22): ', ERI[1, 1, 1, 1])
        '''

    # For problem (e), we need to evaluate H2
    elif molecular == 'H2':
        # For problem 1. (a)
        # The integrals can be generated as followed
        S = np.zeros([2, 2])
        T = np.zeros([2, 2])
        Va = np.zeros([2, 2])
        Vb = np.zeros([2, 2])
        Zb = 1

        for i in range(3):
            for j in range(3):
                # Overlap matrix
                S[0, 0] += S_int(Expon_H[i], Expon_H[j], 0) * Coeff_H[i] * Coeff_H[j]
                S[0, 1] += S_int(Expon_H[i], Expon_H[j], _r) * Coeff_H[i] * Coeff_H[j]
                S[1, 1] = S[0, 0]
                S[1, 0] = S[0, 1]

                # Kinetic Matrix
                T[0, 0] += T_int(Expon_H[i], Expon_H[j], 0) * Coeff_H[i] * Coeff_H[j]
                T[0, 1] += T_int(Expon_H[i], Expon_H[j], _r) * Coeff_H[i] * Coeff_H[j]
                T[1, 1] = T[0, 0]
                T[1, 0] = T[0, 1]

                # Nuclear attraction integrals
                Rap = Expon_H[j] * _r / (Expon_H[i] + Expon_H[j])
                Rbp = _r - Rap
                Va[0, 0] += V_int(Expon_H[i], Expon_H[j], 0, 0, Zb) * Coeff_H[i] * Coeff_H[j]
                Va[0, 1] += V_int(Expon_H[i], Expon_H[j], _r, Rap, Zb) * Coeff_H[i] * Coeff_H[j]
                Va[1, 1] += V_int(Expon_H[i], Expon_H[j], 0, _r, Zb) * Coeff_H[i] * Coeff_H[j]
                Va[1, 0] = Va[0, 1]

                Vb = Va
                Vb[0, 0] = Va[1, 1]
                Vb[1, 1] = Va[0, 0]

        H_core = T + Va + Vb

        # For problem (b)
        # Calculate the two electron integrals
        ERI = np.zeros([2, 2, 2, 2])

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        Rap = Expon_H[i] * _r / (Expon_H[i] + Expon_H[j])
                        Raq = Expon_H[k] * _r / (Expon_H[k] + Expon_H[l])
                        Rbq = _r - Raq
                        Rpq = Rap - Raq

                        ERI[0, 0, 0, 0] += ERI_int(Expon_H[i], Expon_H[j], Expon_H[k], Expon_H[l], 0, 0, 0) * Coeff_H[i] *\
                                           Coeff_H[j] * Coeff_H[k] * Coeff_H[l]

                        ERI[1, 0, 0, 0] += ERI_int(Expon_H[i], Expon_H[j], Expon_H[k], Expon_H[l], _r, 0, Rap) * Coeff_H[i] *\
                        Coeff_H[j] * Coeff_H[k] * Coeff_H[l]

                        ERI[1, 0, 1, 0] += ERI_int(Expon_H[i], Expon_H[j], Expon_H[k], Expon_H[l], _r, _r, Rpq) * Coeff_H[i] * \
                        Coeff_H[j] * Coeff_H[k] * Coeff_H[l]

                        ERI[1, 1, 0, 0] += ERI_int(Expon_H[i], Expon_H[j], Expon_H[k], Expon_H[l], 0, 0, _r) * Coeff_H[i] * \
                        Coeff_H[j] * Coeff_H[k] * Coeff_H[l]

                        ERI[1, 1, 1, 0] += ERI_int(Expon_H[i], Expon_H[j], Expon_H[k], Expon_H[l], 0, _r, Rbq) * Coeff_H[i] * \
                        Coeff_H[j] * Coeff_H[k] * Coeff_H[l]

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

    # For problem (c)
    toler = 1e-14
    MaxIter = 2000
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
        # print("Iter No.:", Iter)
        # Calculate the two electron of the Fock Matrix G = sum(k, l) P[k, l] * (ij|kl) - 0.5(il|kj)
        G = np.zeros([2, 2])

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        G[i, j] += P[k, l] * (ERI[i, j, k, l] - 0.5 * ERI[i, l, k, j])

        # Construct the Fock matrix F = H_core + G
        F = H_core + G

        # Calculate the electronic energy, E_ele = 0.5 * P (H_core + F)
        energy_ele = np.sum(0.5 * P * (H_core + F))
        # print('Electronic energy = ', energy_ele)

        # Transform the Fock matrix F' = X'FX
        F_prime = np.linalg.multi_dot([X.transpose().conjugate(), F, X])

        # Diagonalize F' to obtain C' and epsilon
        epsilon, C_prime = np.linalg.eigh(F_prime)

        # Transform the C' back to C by C = XC'
        C = np.dot(X, C_prime)

        # Construct a new density matrix from the C by P = 2 C'C
        P_old = P.copy()
        C_occ = C[:, :1]
        P = 2 * np.dot(C_occ, C_occ.transpose())

        # Check whether the procedure is converged by check the density matrix
        # Check the RMS of the density matrix
        Delta = P - P_old
        Delta = np.sqrt(np.sum(Delta ** 2) / 4)
        # print("Delta", Delta)

        # Once converged
        if Delta < toler:
            # Add the nuclear repulsion energy
            if molecular == 'HeH+':
                _energy = energy_ele + Za * Zb / _r
            elif molecular == 'H2':
                _energy = energy_ele + Zb / _r
            '''
            print('\nConverged!!!')
            print("Calculation converged with electronic energy:", energy_ele)
            print("Calculation converged with total energy:", energy)
            print('MO Coefficients:\n', C)
            print('Orbital Energies:\n', np.diag(epsilon))
            '''
            break

    return _energy, ERI, epsilon, H_core

def CI_calculation(_eri, _epsilon, _h):

    # First construct the 3X3 CI matrix, in order ground state, single excitation, double excitation
    CI_matrix = np.zeros([3, 3])

    #
    CI_matrix[0, 2] = _eri[0, 1, 1, 0]
    CI_matrix[1, 1] = _epsilon[1] - _epsilon[0] - _eri[1, 1, 0, 0] + 2 * _eri[1, 0, 0, 1]
    CI_matrix[1, 2] = pow(2, -0.5) * (2 * _h[0, 1] + 2 * _eri[0, 1, 1, 1] + 2 * _eri[0, 0, 1, 0] - 2 * _eri[0, 1, 1, 1])
    CI_matrix[2, 2] = 2 * (_epsilon[1] - _epsilon[0]) + _eri[0, 0, 0, 0] + _eri[1, 1, 1, 1] - 4 * _eri[0, 0, 1, 1]\
                      + 2 * _eri[0, 1, 1, 0]

    CI_matrix += CI_matrix.transpose() - np.diag(CI_matrix.diagonal())

    energy_correct, CI_coefficient = np.linalg.eigh(CI_matrix)

    return energy_correct, CI_coefficient


# Useful parameters
np.set_printoptions(precision=7, linewidth=200, threshold=2000, suppress=True)
pi = np.pi
exp = np.exp
r = 1.4632
energy_HeH, ERI, e, h = HFSCF(r, 'HeH+')
energy_H2, ERI, e, h = HFSCF(1.4, 'H2')
# For problem (d)
# Perform HF-SCF from 0.5 to 5 a.u.
'''
distance = np.linspace(0.5, 5, 500)
energy_surface_HeH = np.zeros([500])
energy_surface_H2 = np.zeros([500])

for index, R in enumerate(distance):
    energy_surface_HeH[index], tmp1, tmp2, tmp3 = HFSCF(R, 'HeH+')
    energy_surface_H2[index], tmp1, tmp2, tmp3 = HFSCF(R, 'H2')

# Plot the energy surface
fig = plt.plot(distance, energy_surface_HeH, '-', label='HeH+')
fig = plt.plot(distance, energy_surface_H2, '-', label='H2')
plt.ylabel('E (a.u.)')
plt.xlabel('R (a.u.)')
plt.legend(loc='best')
plt.show()

print('Equlibrium Bond Length:', distance[energy_surface_HeH.argmin()], '\nMinimum Energy:', energy_surface_HeH.min())

# For problem (f), perform the full CI calculation
E_correct, CI_coeffient = CI_calculation(ERI, e, h)
print('Correction Energy:\n', E_correct)
print('CI Coefficients:\n', CI_coeffient)
'''
print('Program works successfully!! Go buy some beer!!')
