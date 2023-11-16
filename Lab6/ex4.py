
from scipy import signal
import numpy as np 
import matplotlib.pyplot as plt
import sys 

def medie_alunecatoare(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_filters(order, x, norm_Wn, rp, filename):
    # d) 

    # filtru Butterworth
    bb, ab = signal.butter(order, norm_Wn, btype='low')

    # filtru Chebyshev
    bc, ac = signal.cheby1(order, rp, norm_Wn, btype='low')

    # e)

    plt.figure()
    b = signal.filtfilt(bb, ab, x)
    c = signal.filtfilt(bc, ac, x)

    plt.plot(x, 'k-', label='input')
    plt.plot(b, 'b-', linewidth=2, label='butter')
    plt.plot(c, 'r-', linewidth=2, label='cheby')
    plt.legend(loc='best')
    if filename is not None:
        plt.savefig(f"grafice/{filename}.pdf")
        plt.savefig(f"grafice/{filename}.png")
    plt.show()

if __name__ == "__main__":
    subpunct = sys.argv[1]

    x = np.genfromtxt("../Lab5/data/Train.csv", delimiter=",")
    x = x[1:][:, 2] # eliminam header-ul si pastram doar ultima coloana
    
    x = x[:72] # a) selectam primele 3 zile

    if subpunct == "b":
        fig, axd = plt.subplot_mosaic([["e", "e"], ["a", "b"], ["c", "d"]])

        fig.suptitle("Medie alunecatoare")
        
        arr = [5, 9, 13, 17]
        for k, ax in axd.items(): 
            if k == "e":
                ax.plot(x)
                ax.set_title(f"Original")
            else:
                w = arr.pop(0)
                print(w)
                ax.plot(medie_alunecatoare(x, w))
                ax.set_title(f"Window = {w}")

        fig.tight_layout()
        plt.savefig("grafice/ex4b.pdf")
        plt.savefig("grafice/ex4b.png")
        plt.show()
    elif subpunct == "e" or subpunct == "f":
        # c) 
        freq_orig = 1 / 3600 
        nyquist = freq_orig / 2
        Wn = freq_orig / 4 # am ales aceasta frecventa pentru a minimiza pierderea informatiei si pentru a pastra cat mai neted graficul 
        norm_Wn = Wn / nyquist
        print(Wn)
        print(norm_Wn)

        rp = 5
        
        if subpunct == "e":
            plot_filters(5, x, norm_Wn, rp, "ex4e")
            # Aleg filtrul Butterworth pentru a pastra forma undei cat mai intacta
            # deoarece doresc pastrarea "pattern-urilor" de trafic
        else:
            orders = [3, 6, 9, 12, 20]
            for order in orders:
                plot_filters(order, x, norm_Wn, rp, f"ex4f{order}")

            # cele mai bune rezultate le-am obtinut cu ordinele 6 si 9 pe Butterworht 

            # alt rp pentru a observa diferentele
            rps = [1, 3, 5, 7, 9]
            for rp in rps:
                plot_filters(5, x, norm_Wn, rp, None)

            # marirea rp-ului va face filtrul Chebyshev sa fie mai puternic
            # si automat scade "puterea" semnalului filtrat