
import numpy as np 
import matplotlib.pyplot as plt

def rectangle(n):
    return np.ones(n)

def hanning(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)

def apply_window(x, window):
    return x * window

Nw = 200 

sine = lambda x: np.sin(2 * np.pi * 100 * x)
sp = np.linspace(0, 0.01 * 15, Nw)

x = sine(sp)

fig, axs = plt.subplots(3, 1)

axs[0].plot(x)
axs[0].set_title("Original")

axs[1].plot(apply_window(x, rectangle(Nw)))
axs[1].set_title("Dreptunghi")

axs[2].plot(apply_window(x, hanning(Nw)))
axs[2].set_title("Hanning")

fig.tight_layout()
plt.savefig("grafice/ex3.pdf")
plt.savefig("grafice/ex3.png")
plt.show()

