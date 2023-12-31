## (a) Care este frecventa de esantionare a semnalului din `Train.csv` (revedeti sectiunea pentru detalii despre cum a fost achizitionat acesta)?

In fisier avem cate un row pentru fiecare ora ($1 \ ora = 60 \ minute = 3600 \ secunde$), deci frecventa de esantionare este $\frac{1}{3600}$ Hz. 

## (b) Ce interval de timp acopera esantioanele din fisier? 

Primul esantion are timestamp-ul `25.08.2012 00:00`, in timp ce ultimul are timestamp-ul `25.09.2014 23:00`. Prin urmare, intervalul de timp este de 2 ani, 1 luna si 23 de ore. In ore, ar insemna `18288` de ore (18288 de inregistrari, o inregistrare / ora). 

## (c) Considerand ca semnalul a fost esantionat corect (fara aliere) si optim, care este frecventa maxima prezenta in semnal?

Conform teoremei Nyquist-Shannon, frecventa maxima care poate aparea in semnal este jumatate din frecventa de esantionare, adica $\frac{1}{7200}$ Hz. 

## (f) Care sunt frecventele principale continue in semnal, asa cum apar in transformata Fourier? Mai exact, determinati primele 4 cele mai mari valori ale modulului transformatei si specificatii caror frecvente (in Hz) le corespund. Caror fenomene periodice din semnal se asociaza fiecare?

Top 4 frecvente:

1. $0$ Hz (componenta continua)
2. $0.00000002$ Hz 
3. $0.00000003$ Hz 
4. $0.00001158$ Hz 

Am calculat niste frecvente "tipice" (in Hz): 

Frecventa zilnica: $0.0000115741$
Frecventa saptamanala: $0.0000016534$
Frecventa lunara: $0.0000003805$
Frecventa anuala: $0.0000000317$

A treia frecventa este aproximativ egala cu frecventa anuala, deci putem asocia acest fenomen cu un "eveniment" ce are loc o data pe an in acea intersectie, ce genereaza aglomeratie. 

A doua frecventa este aproximativ egala cu frecventa pentru $\frac{2}{3}$ dintr-un an, adica $8$ luni. Explicatia este posibil sa fie similara cu cea de mai sus. 

A patra frecventa este aproximativ egala cu frecventa zilnica, care probabil corespunde cu un eveniment de tip "rush hour".

## (h) Nu se cunoaste data la care a inceput masurarea acestui semnal. Concepeti o metoda (descrieti in cuvinte) prin care sa determinati, doar analizand semnalul in timp, aceasta data. Comentati ce neajunsuri ar putea avea solutia propusa si care sunt factorii de care depinde acuratetea ei.

Folosind rezultatele de la punctul f, am putea face un FFT pentru a determina magnitudinea frecventelor precum cea anuala. Pe baza componentei continue, am putea construi un semnal care sa aiba doar aceasta frecventa, si sa facem un fel de "overlap" intre semnalul compus si semnalul simplu. 

Daca avem mai multe date despre natura traficului din intersectia respectiva, putem sa estimam cand au loc acele "spike-uri" anuale, si, cunoscand frecventa de esantionare, sa mergem in spate pentru a determina data la care a inceput semnalul. 

Neajunsul acestei metode este ca se bazeaza pe diverse "supozitii", precum cea ca spike-ul anual coincide cu un eveniment anume. 

## (i) Filtrati semnalul, eliminati componentele de frecventa inalta (la alegerea voastra care/cate, dar alegerea sa se poata justifica).

Am eliminat cele mai inalte 10% frecvente, considerand ca este o marja suficient de mica pentru a fi considerata "zgomot". Am incercat si sa elimin doar 5%, dar am ramas la 10% pentru a fi mai vizibil pe grafic.