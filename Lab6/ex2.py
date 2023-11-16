
import numpy as np 

N = 6
polinom_p = np.random.randint(0, 20, size=N)
polinom_q = np.random.randint(0, 20, size=N)

def convolve_fft(p, q):
    marime = len(p) + len(q) - 1
    return np.round(np.fft.ifft(np.fft.fft(p, n=marime) * np.fft.fft(q, n=marime)).real)

def convolve(p, q):
    r = np.zeros(len(p) + len(q) - 1)
    for i in range(len(p)):
        for j in range(len(q)):
            r[i + j] += p[i] * q[j]
    return r

print(convolve_fft(polinom_p, polinom_q))
print(convolve(polinom_p, polinom_q))