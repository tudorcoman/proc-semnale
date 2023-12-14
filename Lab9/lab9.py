
import sys
sys.path.append("..")

from Lab8.lab8 import generate_time_series
import numpy as np 
import matplotlib.pyplot as plt 

def exponential_average(alpha, timeseries):
    smoothed = [timeseries[0]]
    for i in range(1, len(timeseries)):
        smoothed.append(alpha * timeseries[i] + (1 - alpha) * smoothed[i - 1])
    return smoothed

def calculate_squared_error(alpha, timeseries):
    smoothed = exponential_average(alpha, timeseries)
    sum = 0 
    for i in range(len(timeseries) - 1):
        sum += (smoothed[i] - timeseries[i + 1]) ** 2
    return sum 

def find_alpha(timeseries):
    errors = []
    min_error = float("inf")
    best_alpha = None
    space = np.linspace(0, 1, 1000)
    for alpha in space:
        error = calculate_squared_error(alpha, timeseries)
        errors.append(error)
        if error < min_error:
            min_error = error
            best_alpha = alpha
    return best_alpha

def find_thetas(epsilon, timeseries, p, n):
    y_vector = timeseries[p:]
    eps_matrix = np.array([epsilon[i:i+p] for i in range(n - p)])
    return eps_matrix, np.linalg.lstsq(eps_matrix, y_vector, rcond=None)[0]

def moving_average_model(timeseries, p):
    avg = np.mean(timeseries)
    epsilon = np.random.normal(0, 1, len(timeseries))
    eps_matrix, thetas = find_thetas(epsilon, timeseries, p, len(timeseries))
    x = eps_matrix @ thetas
    new_timeseries = avg + epsilon[p:] + x
    return new_timeseries

x = np.linspace(0, 1, 1000)
trend, season, noise, timeseries = generate_time_series(x)
smoothed_timeseries = exponential_average(0.05, timeseries)

fig, axs = plt.subplots(2, 1)
axs[0].plot(x, timeseries)
axs[0].set_title("Timeseries")
axs[1].plot(x, smoothed_timeseries)
axs[1].set_title("Smoothed timeseries")
fig.tight_layout()
plt.savefig("exponential_average.pdf")
plt.savefig("exponential_average.png")
plt.show()

print("Best alpha: ", find_alpha(timeseries))

fig, axs = plt.subplots(2, 1)
axs[0].plot(x, timeseries)
axs[0].set_title("Timeseries")
q = 300
axs[1].plot(x[q:], moving_average_model(timeseries, q))
axs[1].set_title(f"MA model (q={q})")
fig.tight_layout()
plt.savefig("moving_average.pdf")
plt.savefig("moving_average.png")
plt.show()
