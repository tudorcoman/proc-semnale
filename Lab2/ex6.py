import sys
sys.path.append("..")

from Lab1.ex2 import get_sinusoidal
from ex1 import sample_space
import matplotlib.pyplot as plt 

frecventa_esantionare = 10

x = sample_space(1, frecventa_esantionare)

first_signal = get_sinusoidal(frecventa_esantionare / 2)
second_signal = get_sinusoidal(frecventa_esantionare / 4)
third_signal = get_sinusoidal(0)

fig, axs = plt.subplots(3)
fig.suptitle("Semnale cu frecventa fs/2, fs/4 si 0")

axs[0].plot(x, first_signal(x))
axs[1].plot(x, second_signal(x))
axs[2].plot(x, third_signal(x))
plt.show()
