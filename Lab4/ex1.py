import sys
sys.path.append("..")

from Lab3.ex1 import fourier_matrix
from Lab3.ex2 import signal
import matplotlib.pyplot as plt 

import numpy as np
import time

def get_time_for_brute_fft(x):
    timp_start = time.time()
    fr_matrix = fourier_matrix(len(x))
    fr = np.matmul(fr_matrix, x)
    timp_end = time.time()
    return timp_end - timp_start

def get_time_for_np_fft(x):
    timp_start = time.time()
    fr = np.fft.fft(x)
    timp_end = time.time()
    return timp_end - timp_start

freq = 15
semnal_compus = signal(freq)

N = [128, 256, 512, 1024, 2048, 4096, 8192]
xs = [np.linspace(0, 1, n) for n in N]

timpi_brut = [np.log(get_time_for_brute_fft(x)) for x in xs]
timpi_np = [np.log(get_time_for_np_fft(x)) for x in xs]

plt.figure()
plt.plot(N, timpi_brut, label="Brute")
plt.plot(N, timpi_np, label="Numpy")
plt.xlabel("N")
plt.ylabel("log(timp)")
plt.legend()
plt.savefig("ex1.pdf")
plt.savefig("ex1.png")
plt.show()