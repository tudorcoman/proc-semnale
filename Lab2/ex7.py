import sys
sys.path.append("..")

from Lab1.ex2 import get_sinusoidal
from ex1 import sample_space
import matplotlib.pyplot as plt 

frecventa_esantionare = 20
frecventa_semnal = 2
t = 5

x = sample_space(t, t * frecventa_esantionare)

signal = get_sinusoidal(frecventa_semnal)

x_1 = x[::4]
x_2 = x[1::4]

fig, axs = plt.subplots(3)
fig.suptitle("Semnal original vs semnale fragmentate")

axs[0].plot(x, signal(x))
axs[1].plot(x_1, signal(x_1))
axs[2].plot(x_2, signal(x_2))

plt.show()

'''
sinusoida din mijloc este mai putin neteda decat prima 
pentru ca practic frecventa de esantionare este impartita la 4, 
ca la exercitiul anterior; practic, are loc o interpolare liniara
in cazul graficelor 2 si 3 

inceperea cu al doilea element "shifteaza in stanga" sinusoida din mijloc si
cele doua semnale nu mai seamana deloc
'''
