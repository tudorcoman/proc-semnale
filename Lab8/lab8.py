import numpy as np
import matplotlib.pyplot as plt

def generate_trend(sp):
    fct = lambda x: 5 * x * x + 3 * x + 2
    return fct(sp)

def generate_szn(sp):
    first_period_fct = lambda x: np.sin(2 * np.pi * x) + 9
    second_period_fct = lambda x: np.sin(4 * np.pi * x) + 5
    return first_period_fct(sp) + second_period_fct(sp)

def generate_noise(sp):
    # white gaussian noise 
    return np.random.normal(0, 0.0005, len(sp))

def generate_time_series(sp):
    first_component = generate_trend(sp)
    second_component = generate_szn(sp)
    third_component = generate_noise(sp)
    return (first_component, second_component, third_component, 
            first_component + second_component + third_component)

### punctul a

x = np.linspace(0, 1, 1000)
trend, season, noise, timeseries = generate_time_series(x)

fig, axs = plt.subplots(2, 2)

axs[0][0].plot(x, timeseries)
axs[0][0].set_title("Timeseries")

axs[0][1].plot(x, trend)
axs[0][1].set_title("Trend")

axs[1][0].plot(x, season)
axs[1][0].set_title("Season")

axs[1][1].plot(x, noise)
axs[1][1].set_title("Noise")

fig.tight_layout()
plt.savefig("timeseries.pdf")
plt.savefig("timeseries.png")
plt.show()

## punctul b
## generate the autocorrelation array for timeseries
## and plot it
def autocorrelation(x):
    mean = np.mean(x)
    std = np.std(x)
    y = (x - mean) / std
    autocorr = np.correlate(y, y, mode='full')
    autocorr_normalized = autocorr[len(autocorr) // 2:] / len(x)
    return autocorr_normalized

autocorr = autocorrelation(np.copy(timeseries))

fig, ax = plt.subplots()
ax.plot(np.arange(1000), autocorr)
ax.set_title("Autocorrelation")
plt.savefig("autocorrelation.pdf")
plt.savefig("autocorrelation.png")
plt.show()

## punctul c 

def x_star(timeseries, p, training_threshold):
    y_vector = timeseries[training_threshold:p:-1]
    y_matrix = np.array([timeseries[i:i-p:-1] for i in range(training_threshold-1, p-1, -1)])
    return np.linalg.lstsq(y_matrix, y_vector, rcond=None)[0]

def predict(x_star, timeseries, i, p):
    beta = np.reshape(x_star, (-1, 1))
    
    y_lags = np.flip(timeseries[i-1:i-p-1:-1]).reshape(-1, 1)
    y_hat_next = beta.T @ y_lags
    
    return y_hat_next[0, 0]


training_threshold = 700 
predictions = [] 
p = 50
xs = x_star(timeseries, p, training_threshold)

for i in range(training_threshold + 1, len(timeseries)):
    predictions.append((i, predict(xs, timeseries, i, p)))

fig = plt.figure()
plt.plot(x, timeseries)
plt.vlines(x[training_threshold], ymin=min(timeseries), ymax=max(timeseries), colors='r', linestyles='dashed')
for i, pred in predictions:
   plt.scatter(x[i], pred, color='r')
plt.title("Timeseries with predictions")
plt.savefig("predictions.pdf")
plt.savefig("predictions.png")
plt.show()

## punctul d 

def hyperparameter_tuning(timeseries):
    minimum_error = 100000
    best_p = None 
    best_training_threshold = None 

    for training_threshold in range(2, len(timeseries) - 1):
        print(training_threshold)
        print("Best p: ", best_p)
        print("Best training threshold: ", best_training_threshold)
        for p in range(2, training_threshold):
            xs = x_star(timeseries, p, training_threshold)
            error = 0
            prediction = predict(xs, timeseries, training_threshold + 1, p)
            error = (prediction - timeseries[training_threshold + 1])

            if error < minimum_error:
                minimum_error = error
                best_p = p
                best_training_threshold = training_threshold
    return best_p, best_training_threshold


bp, btt = hyperparameter_tuning(timeseries)
print("Best p: ", bp)
print("Best m: ", btt)

''' 
Best p: 456
Best m: 912
'''
