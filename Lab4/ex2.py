
import numpy as np 
import matplotlib.pyplot as plt

def generate_identical_signal(k, freq, under_nyquist):
    return lambda x: np.sin(2 * np.pi * x * (freq + k * under_nyquist))

def generate_f_desen(signal):
    freq = 1000
    xs = np.linspace(0, 1, freq + 1)
    return xs, [signal(x) for x in xs]

freq = 5
signal = lambda x: np.sin(2 * np.pi * x * freq)

under_nyquist = 7

first_k = 2
first_identical_signal = generate_identical_signal(first_k, freq, under_nyquist)

second_k = 4
second_identical_signal = generate_identical_signal(second_k, freq, under_nyquist)


esantioane = np.linspace(0, 1, under_nyquist + 1)

signal_xs, signal_desen = generate_f_desen(signal)
first_xs, first_identical_signal_desen = generate_f_desen(first_identical_signal)
second_xs, second_identical_signal_desen = generate_f_desen(second_identical_signal)

fig, axs = plt.subplots(3)
fig.tight_layout(pad=2.0)
axs[0].plot(signal_xs, signal_desen)
axs[0].stem(esantioane, [signal(x) for x in esantioane])

axs[1].plot(first_xs, first_identical_signal_desen)
axs[1].stem(esantioane, first_identical_signal(esantioane))

axs[2].plot(second_xs, second_identical_signal_desen)
axs[2].stem(esantioane, second_identical_signal(esantioane))

plt.savefig("ex2.pdf")
plt.savefig("ex2.png")
plt.show()