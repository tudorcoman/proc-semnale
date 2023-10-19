
import numpy as np 
import matplotlib.pyplot as plt 

def sin_signal(f, phi):
    return lambda x: np.sin(2 * np.pi * f * x + phi)

def cos_signal(f, phi):
    return lambda x: np.cos(2 * np.pi * f * -x + phi)

def sample_space(t, number_of_samples):
    return np.linspace(0, t, number_of_samples)

if __name__ == "__main__":
    first_signal = sin_signal(5, 0)
    second_signal = cos_signal(5, np.pi / 2)

    fig, axs = plt.subplots(2)
    fig.suptitle("Semnal sinusoidal si semnal cosinusoidal")

    x = sample_space(1, 1000)
    axs[0].plot(x, first_signal(x))
    axs[1].plot(x, second_signal(x))
    plt.show()