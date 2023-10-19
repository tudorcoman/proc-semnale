
import numpy as np 
import matplotlib.pyplot as plt 

def eroare(x, y):
    return np.abs(x - y)

x = np.linspace(-np.pi/2, np.pi/2, 1000)

sin = lambda x: np.sin(x)
taylor = lambda x: x 
pade = lambda x: (x - ((7*(x**3)) / 60)) / (1 + ((x**2)/20))

fig, axs = plt.subplots(2)
fig.suptitle("Sin vs Taylor vs Pade")

axs[0].plot(x, sin(x), color="blue", label="sin")
axs[0].plot(x, taylor(x), color="red", label="taylor")
axs[0].plot(x, pade(x), color="green", label="pade")
axs[0].legend()


axs[1].semilogy(x, eroare(sin(x), taylor(x)), color="blue", label="taylor error")
axs[1].semilogy(x, eroare(sin(x), pade(x)), color="red", label="pade error")
axs[1].legend()

plt.show()