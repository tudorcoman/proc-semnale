
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import pandas as pd

date = pd.read_csv('co2.csv')

# remove unnecessary columns
date = date.drop(['decimal'], axis=1)

#print(date)

monthly_avg = pd.DataFrame(date)
monthly_avg['year_month'] = date['year'].astype(str) + '-' + date['month'].astype(str)
monthly_avg.drop(['year', 'month', 'day'], axis=1, inplace=True)
monthly_avg = monthly_avg.groupby('year_month').mean()
print(monthly_avg)

fig, ax = plt.subplots()
ax.plot(monthly_avg)
ax.xaxis.set_major_locator(ticker.MultipleLocator(72))
plt.draw()
plt.savefig("ex3_a.pdf")
plt.savefig("ex3_a.png")
plt.show()


X = np.arange(1, len(monthly_avg['ppm']) + 1)
X = np.vstack([np.ones(len(X)), X]).T
coef, _, _, _ = np.linalg.lstsq(X, monthly_avg['ppm'], rcond=None)
trend = X.dot(coef)

print(trend)
detrend_monthly_avg = monthly_avg['ppm'] - trend

fig, ax = plt.subplots()
ax.plot(detrend_monthly_avg)
ax.xaxis.set_major_locator(ticker.MultipleLocator(72))
plt.savefig("ex3_b.pdf")
plt.savefig("ex3_b.png")
plt.show()