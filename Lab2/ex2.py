
import numpy as np 
import matplotlib.pyplot as plt

def sinus_cu_faza(faza, freq):
    return lambda x: np.sin(2 * np.pi * x * freq + faza)

def get_gamma(snr, x, z):
    nrx = np.sum(np.square(x))
    nrz = np.sum(np.square(z))

    gamma_patrat = snr * nrx / nrz 
    return np.sqrt(gamma_patrat)

def semnal_cu_zgomot(x, semnal, gamma, zgomot):
    return semnal(x) + gamma * zgomot 

f = 5
x = np.linspace(0, 1, 1000)

semnal_1 = sinus_cu_faza(0, f)
semnal_2 = sinus_cu_faza(np.pi / 2, f)
semnal_3 = sinus_cu_faza(np.pi, f)
semnal_4 = sinus_cu_faza(3 * np.pi / 2, f)

z_1 = np.random.normal(0, 1, 1000)
z_2 = np.random.normal(0, 1, 1000)
z_3 = np.random.normal(0, 1, 1000)
z_4 = np.random.normal(0, 1, 1000)

g_1 = get_gamma(0.1, semnal_1(x), z_1)
g_2 = get_gamma(1, semnal_2(x), z_2)
g_3 = get_gamma(10, semnal_2(x), z_2)
g_4 = get_gamma(100, semnal_2(x), z_2)

sz1 = lambda x: semnal_cu_zgomot(x, semnal_1, g_1, z_1)
sz2 = lambda x: semnal_cu_zgomot(x, semnal_2, g_2, z_2)
sz3 = lambda x: semnal_cu_zgomot(x, semnal_3, g_3, z_3)
sz4 = lambda x: semnal_cu_zgomot(x, semnal_4, g_4, z_4)

fig, axs = plt.subplots(2)
axs[0].plot(x, semnal_1(x), color="blue", label="0")
axs[0].plot(x, semnal_2(x), color="red", label="pi/2")
axs[0].plot(x, semnal_3(x), color="green", label="pi")
axs[0].plot(x, semnal_4(x), color="orange", label="3pi/2")
axs[0].legend()

axs[1].plot(x, sz4(x), color="orange", label="3pi/2")
axs[1].plot(x, sz3(x), color="green", label="pi")
axs[1].plot(x, sz2(x), color="red", label="pi/2")
axs[1].plot(x, sz1(x), color="blue", label="0")
axs[1].legend()
plt.show()