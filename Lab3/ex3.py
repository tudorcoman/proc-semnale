
from ex2 import signal 
import numpy as np
import matplotlib.pyplot as plt 

def pondere(p, signal): # p * signal
    return lambda x: p * signal(x)

def add_signals(*signals):
    return lambda x: sum([s(x) for s in signals])

def fourier_brute(esantioane, omega):
    n = len(esantioane)
    ind = np.arange(n)
    arr = esantioane * np.exp(-2 * np.pi * 1j * ind * omega / n)
    return np.sum(arr)

semnal_compus = add_signals(signal(5), pondere(1/2, signal(10)), pondere(1/6, signal(15)))
f_esant = 100
x = np.linspace(0, 1, f_esant)

m_max = int(f_esant / 5)


fig, ax = plt.subplots(nrows=1, ncols=2)
fig.tight_layout(pad=2.0)
ax[0].plot(x, semnal_compus(x))
ax[0].set_title("Semnal compus")
ax[0].set_xlabel("Timp (s)")
ax[0].set_ylabel("x(t)")

x_gr = [i * 5 for i in range(m_max)]
y_gr = [fourier_brute(semnal_compus(x), i) for i in x_gr]
ax[1].set_title("Fourier")
ax[1].set_xlabel("Frecventa (Hz)")
ax[1].set_ylabel("|$X(\\omega)$|")
ax[1].stem(x_gr, y_gr)
plt.savefig("ex3.pdf")
plt.show()
