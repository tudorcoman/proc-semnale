
import sys
sys.path.append("..")

from Lab8.lab8 import generate_time_series
from statsmodels.tsa.arima.model import ARIMA
import numpy as np 
import matplotlib.pyplot as plt 

def exponential_average(alpha, timeseries):
    smoothed = [timeseries[0]]
    for i in range(1, len(timeseries)):
        smoothed.append(alpha * timeseries[i] + (1 - alpha) * smoothed[i - 1])
    return smoothed

def calculate_error(model_timeseries, expected_timeseries):
    if len(model_timeseries) < len(expected_timeseries):
        diff = len(expected_timeseries) - len(model_timeseries)
        expected_timeseries = expected_timeseries[diff:]

    sum = 0
    for i in range(len(model_timeseries)):
        sum += (model_timeseries[i] - expected_timeseries[i]) ** 2
    return sum / len(model_timeseries)

def calculate_squared_error(alpha, timeseries):
    smoothed = exponential_average(alpha, timeseries)
    return calculate_error(smoothed[:-1], timeseries[1:])

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
    avg = np.mean(timeseries[p:])
    epsilon = np.random.normal(0, 1, len(timeseries))
    eps_matrix, thetas = find_thetas(epsilon, timeseries, p, len(timeseries))
    x = eps_matrix @ thetas
    new_timeseries = avg + epsilon[p:] + x
    return new_timeseries

def ar_y_matrix(timeseries, p, n):
    y_matrix = np.array([timeseries[p-i:n-i] for i in range(1, p + 1)])
    return y_matrix.T

def arma_model_train(timeseries, p, q):
    epsilon = np.random.normal(0, 1, len(timeseries))
    eps_matrix, _ = find_thetas(epsilon, timeseries, q, len(timeseries))
    y_matrix = ar_y_matrix(timeseries, p, len(timeseries))

    if p < q:
        y_matrix = y_matrix[q - p:]
    elif p > q:
        eps_matrix = eps_matrix[p - q:]

    mat = np.concatenate((y_matrix, eps_matrix), axis=1)
    params = np.linalg.lstsq(mat, timeseries[max(p, q): ], rcond=None)[0]

    return params, epsilon

def arma_model_test(train, p, q, params, eps, predicts):
    ans = train.copy()
    predictions = []
    while predicts > 0:
        e = np.random.normal(0, 1)
        y = params.T @ np.concatenate((ans[-p : ], eps[-q : ])) + e 
        ans = np.append(ans, y)
        eps = np.append(eps, e)
        predictions.append(y)
        predicts -= 1
    
    return predictions

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

q = 300
fig, axs = plt.subplots(2, 1)
axs[0].plot(x[q:], timeseries[q:])
axs[0].set_title("Timeseries")
axs[1].plot(x[q:], moving_average_model(timeseries, q))
axs[1].set_title(f"MA model (q={q})")
fig.tight_layout()
plt.savefig("moving_average.pdf")
plt.savefig("moving_average.png")
plt.show()


train_size = 900
train_timeseries = timeseries[:train_size]
test_timeseries = timeseries[train_size:]

best_p_and_q = None 
min_error = float("inf")

for p in range(2, 21):
    for q in range(2, 21):
        params, eps = arma_model_train(train_timeseries, p, q)
        predictions = arma_model_test(train_timeseries, p, q, params, eps, len(test_timeseries))
        error = calculate_error(predictions, test_timeseries)
        if error < min_error:
            min_error = error
            best_p_and_q = (p, q)

print("Best p and q: ", best_p_and_q)
fig = plt.figure()
plt.plot(x, timeseries)
best_p, best_q = best_p_and_q
plt.title(f"Timeseries with ARMA predictions (p={best_p}, q={best_q})")
params, eps = arma_model_train(train_timeseries, best_p, best_q)
predictions = arma_model_test(train_timeseries, best_p, best_q, params, eps, len(test_timeseries))
plt.plot(x[train_size:], predictions)

fig.tight_layout()
plt.savefig("arma_model.pdf")
plt.savefig("arma_model.png")
plt.show()

## custom model 

model = ARIMA(train_timeseries, order=(2, 0, 2))
results = model.fit()
predictions = results.predict(len(test_timeseries))

fig = plt.figure()
plt.plot(np.concatenate((train_timeseries, predictions)), color='orange')
plt.plot(timeseries)
plt.title("Timeseries with custom ARMA predictions")
plt.savefig("custom_arma_model.pdf")
plt.savefig("custom_arma_model.png")
plt.show()