
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import sys 

# credits: https://stackoverflow.com/questions/67378357/how-to-calculate-the-first-monday-of-the-month-python-3-3 
from datetime import datetime, timedelta
def find_first_monday(year, month):
    d = datetime(year, int(month), 7)
    offset = -d.weekday() # weekday == 0 means Monday
    return d + timedelta(offset)

if __name__ == "__main__":
    subpunct = sys.argv[1]
    x = np.genfromtxt("data/Train.csv", delimiter=",")

    x = x[1:][:, 2] # eliminam header-ul si pastram doar ultima coloana
    
    sample_rate = 1.0/3600 # Hz 
    T = 1.0 / sample_rate
    
    X = np.fft.fft(x)
    N = len(X)
    X = abs(X / N)[:N // 2]

    frecvente = sample_rate * np.linspace(0,N/2,N//2)/N

    magnitudini = np.abs(X)

    half_spectrum = len(frecvente) // 2
    frecv_pos = frecvente[:half_spectrum]
    magn_pos = magnitudini[:half_spectrum]

    if subpunct in ["a", "b", "c"]:
        print("Read ex1.md")

    elif subpunct == "d":

        plt.figure(figsize=(10, 5))
        plt.plot(frecv_pos, magn_pos)
        plt.title('FFT')
        plt.xlabel('Frecventa (Hz)')
        plt.ylabel('Magnitudine')
        plt.grid(True)
        plt.savefig(f"grafice/1d.pdf")
        plt.savefig(f"grafice/1d.png")
        plt.show()

    elif subpunct == "e":
        # componenta continua este magnitudinea frecventei 0 
        # (frecventa 0 este componenta continua)
        # totodata, media sin sau cos e 0, deci componenta continua e media semnalului

        cont = np.abs(X[0])
        medie = np.mean(x)
        
        print(f"Componenta continua (din FFT): {cont}")
        print(f"Componenta continua (din medie): {medie}")

    elif subpunct == "f":
        biggest_four = np.argsort(magn_pos)[-4:]
        np.set_printoptions(suppress = True)
        print(f"Primele 4 frecvente: {frecv_pos[biggest_four]}")

        frecventa_zilnica = sample_rate / 24
        frecventa_saptamanala = sample_rate / 24 / 7
        frecventa_anuala = sample_rate / 24 / 365
        frecventa_lunara = frecventa_anuala * 12
        

        print(f"Frecventa zilnica: %0.10f" % (frecventa_zilnica))
        print(f"Frecventa saptamanala: %0.10f" % (frecventa_saptamanala))
        print(f"Frecventa lunara: %0.10f" % (frecventa_lunara))
        print(f"Frecventa anuala: %0.10f" % (frecventa_anuala))

    elif subpunct == "g":
        # 1000 esantioane ~= o luna jumatate 
        # putem incepe peste un an intr-o zi de luni 
        zi_de_luni = find_first_monday(2013, 1)
        
        # o luna = 30 zile = 720 ore = 720 esantioane
        df = pd.read_csv('data/Train.csv').dropna()
        df["Datetime"] = pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M')
        x = df[df["Datetime"] >= zi_de_luni]       
        timp = [i for i in range(744)]
        x = x[:744]
        print(x)
        plt.plot(timp, x['Count'])
        plt.title('Trafic in ianuarie 2013')
        plt.xlabel('Esantioane')
        plt.ylabel('Numar masini')
        plt.grid(True)
        plt.savefig(f"grafice/1g.pdf")
        plt.savefig(f"grafice/1g.png")
        plt.show()