
# Obtinem clopotul lui Gauss dupa mai multe convolutii 

import numpy as np 
import matplotlib.pyplot as plt 

def convolution(x):
    X = np.fft.fft(x, n=len(x) * 2 - 1)
    N = len(X)
    y = np.fft.ifft(X * X)
    return y

if __name__ == "__main__":
    x = np.random.rand(100)
    x_iter = [x]
    iterations = 3
    fig, axs = plt.subplots((iterations + 1) // 2, 2)
    fig.suptitle("Convolutii")
    for step in range(iterations + 1):
        if step:
            x_nou = np.copy(x_iter[-1])
            x_nou = convolution(x_nou)
            x_iter.append(x_nou)
            axs[step // 2, step % 2].plot(x_nou)
            axs[step // 2, step % 2].set_title(f"Step {step}")
        else:
            axs[step // 2, step % 2].plot(x)
            axs[step // 2, step % 2].set_title(f"Original")
    fig.tight_layout()
    plt.savefig("grafice/ex1.pdf")
    plt.savefig("grafice/ex1.png")
    plt.show()