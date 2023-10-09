import numpy as np
import matplotlib.pyplot as plt
import sys 

def get_sinusoidal(FREQUENCY):
    return lambda x: np.sin(2 * np.pi * x * FREQUENCY)

def get_square(FREQUENCY):
    sinus = get_sinusoidal(FREQUENCY)
    return lambda x: np.sign(sinus(x))

def get_sawtooth(FREQUENCY):
    return lambda x: (np.floor(FREQUENCY * x) - FREQUENCY * x + 1/2)

def sample_and_plot(signal, samples, time):
    x = []
    y = [] 
    t = 1 # second 
    freq = time / samples 

    for i in range(samples):
        t = i * freq 
        x.append(t)
        y.append(signal(t))

    fig = plt.figure()
    plt.stem(x[:50], y[:50])
    plt.show()

if __name__ == "__main__":
    subpunct = sys.argv[1]

    if subpunct == "a":
        # punctul a 
        SIGNAL_FREQUENCY = 400 
        #PERIOD = 1 / SIGNAL_FREQUENCY
        SAMPLES = 1600

        #semnal_a = lambda x: np.sin(2 * np.pi * x / PERIOD)
        sample_and_plot(get_sinusoidal(SIGNAL_FREQUENCY), SAMPLES, 1)
    elif subpunct == "b":
        # punctul b 

        SIGNAL_FREQUENCY = 800
        TIME = 3
        #PERIOD = 1 / SIGNAL_FREQUENCY
        SAMPLES = 3200 * TIME  
        
        #semnal_b = lambda x: np.sin(2 * np.pi * x / PERIOD)
        sample_and_plot(get_sinusoidal(SIGNAL_FREQUENCY), SAMPLES, TIME)
    elif subpunct == "c":
        SIGNAL_FREQUENCY = 240 
        TIME = 2
        SAMPLES = 500
        sample_and_plot(get_sawtooth(SIGNAL_FREQUENCY), SAMPLES, TIME)
    elif subpunct == "d":
        SIGNAL_FREQUENCY = 300 
        TIME = 5 
        SAMPLES = 1200 * TIME 
        sample_and_plot(get_square(SIGNAL_FREQUENCY), SAMPLES, TIME)

    elif subpunct == "e":
        # punctul e 

        img_rd = np.random.rand(128, 128)
        plt.imshow(img_rd)
        plt.show()
    elif subpunct == "f":
        # punctul f (o sa fie o imagine in forma de T)

        mat = np.zeros((128, 128))
        for i in range(10):
            mat[i] = np.ones(128)
        for i in range(60, 69):
            mat[:, i] = np.ones(128)

        plt.imshow(mat)
        plt.show()