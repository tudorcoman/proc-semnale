import numpy as np
import matplotlib.pyplot as plt

def x(t):
    return np.cos(520 * np.pi * t + np.pi / 3)

def y(t):
    return np.cos(280 * np.pi * t - np.pi / 3)

def z(t):
    return np.cos(120 * np.pi * t + np.pi / 3)

fig, axs = plt.subplots(3)
fig.suptitle("Semnalele x, y, z continue (simulate)")

# write code

START = 0
STOP = 0.03  
STEP = 0.0005
arr = np.arange(START, STOP + STEP / 2, STEP)

axs[0].plot(arr, x(arr))
axs[1].plot(arr, y(arr))
axs[2].plot(arr, z(arr))
plt.show()

FREQ = 200 #Hz
TIME = 0.3 # seconds 

# x[n] = x(nT)
esantion = []
esantion_x = []
esantion_y = []
esantion_z = []

for i in range(0, int(TIME * FREQ) + 1):
    esantion.append(i)
    esantion_x.append(x(i / FREQ))
    esantion_y.append(y(i / FREQ))
    esantion_z.append(z(i / FREQ))

    

fig, axs = plt.subplots(3)
fig.suptitle("Esantioane semnale x, y, z")

axs[0].stem(esantion, esantion_x)
axs[1].stem(esantion, esantion_y)
axs[2].stem(esantion, esantion_z)
plt.show()