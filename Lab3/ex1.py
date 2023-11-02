
import numpy as np 
import matplotlib.pyplot as plt

def fourier_matrix(n):
    omega = np.exp(-2j * np.pi / n) # am folosit np.exp in loc de math.e pentru exponentiere
    j, k = np.meshgrid(np.arange(n), np.arange(n))
    F = np.power(omega, j * k)
    return F

def check_ortogonal_matrix(F):
    n = F.shape[0]
    eps = 1e-10
    transpose_product = np.abs(np.matmul(F, F.conj().T))
    identity_matrix = np.identity(n)
    norm = np.linalg.norm(transpose_product - n * identity_matrix)
    return norm < eps

def check_complex_matrix(F):
    return True in np.iscomplex(F)

if __name__ == "__main__":
    N = 8 
    F = fourier_matrix(N)

    print("Complex" if check_complex_matrix(F) else "Not complex")
    print("Ortogonal" if check_ortogonal_matrix(F) else "Not ortogonal")

    fig, axs = plt.subplots(nrows=N, ncols=2, figsize=(10, 10))

    for i in range(N):
        real_part = np.real(F[i])
        imag_part = np.imag(F[i])
        axs[i][0].plot(range(N), real_part)
        axs[i][1].plot(range(N), imag_part)

    fig.savefig('ex1.pdf')
    fig.show()
