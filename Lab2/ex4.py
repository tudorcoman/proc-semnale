import sys
sys.path.append("..")

from Lab1.ex2 import get_sinusoidal, get_sawtooth
from ex1 import sample_space
import matplotlib.pyplot as plt 

sin_signal = get_sinusoidal(3)
saw_signal = get_sawtooth(20)

x = sample_space(1, 500)

fig, axs = plt.subplots(3)
fig.suptitle("Semnale adunate")

x = sample_space(1, 1000)
axs[0].plot(x, sin_signal(x))
axs[1].plot(x, saw_signal(x))
axs[2].plot(x, sin_signal(x) + saw_signal(x))
plt.show()
