
import numpy as np 
import matplotlib.pyplot as plt

def signal(freq):
    return lambda x: np.sin(2 * np.pi * x * freq + np.pi / 2)

def plot_signal_real(signal, es_freq, ax):
    x = np.linspace(0, 1, es_freq)
    y = signal(x)
    ax.axhline(0, color='black')
    ax.plot(x, y)
    return y 

def calculate_hull(x, omega):
    n = len(x)
    ind = np.arange(n)
    y = x * np.exp(-2 * np.pi * 1j * omega * ind / n) 
    return y

def plot_signal_complex(x, ax):
    y = calculate_hull(x, 1)
    ax.axhline(0, color='black')
    ax.plot(y.real, y.imag)

def plot_hull(x, omega, ax):
    y = calculate_hull(x, omega)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.set_title(f"$\\omega = {omega}$")
    ax.plot(y.real, y.imag)

if __name__ == "__main__":
    sig_freq = 3 #Hz
    es_freq = 1000 #Hz

    first_fig, first_axs = plt.subplots(nrows=1, ncols=2)
    y = plot_signal_real(signal(sig_freq), es_freq, first_axs[0])
    plot_signal_complex(y, first_axs[1])
    first_fig.tight_layout(pad=2.0)
    first_fig.savefig('ex2_1.pdf')
    first_fig.show()

    second_fig, second_axs = plt.subplots(nrows=2, ncols=2)
    plot_hull(y, 7, second_axs[0][0])
    plot_hull(y, 3, second_axs[0][1])
    plot_hull(y, 10, second_axs[1][0])
    plot_hull(y, 17, second_axs[1][1])
    second_fig.tight_layout(pad=2.0)
    second_fig.savefig('ex2_2.pdf')
    second_fig.show()

