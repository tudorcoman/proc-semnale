
from scipy.io.wavfile import read 
import matplotlib.pyplot as plt

import numpy as np 
import sys 

def group_array(arr):
    n = len(arr)
    one_percent = n // 100
    return [arr[i:i+one_percent] for i in range(0, n - one_percent, one_percent // 2)]

def fft_on_groups(groups):
    return [np.fft.fft(group) for group in groups]

sound_name = sys.argv[1]
sound_path = f"./sounds/{sound_name}.wav"
fs, sound = read(sound_path)


groups = group_array(sound)
np_groups = np.array(groups)
fft_groups = fft_on_groups(np_groups)

fft_matrix = np.array(fft_groups).T # transpusa pentru a avea fiecare fft pe coloana 

# spectograma pe coloanele fft-urilor
plt.xlabel("Timp (s)")
plt.ylabel("Frecventa (Hz)")
plt.specgram(fft_matrix, Fs=fs, cmap="jet")
plt.colorbar()
plt.ylim(-2e4, 2e4)
plt.savefig(f"spectogram_{sound_name}.pdf")

