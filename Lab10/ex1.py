
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal

### 1D
mean = 5
variance = 2

distribution = np.random.normal(mean, np.sqrt(variance), 1000)

plt.hist(distribution, bins=30, density=True)

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = 1/(np.sqrt(2 * np.pi * variance)) * np.exp(- (x - mean)**2 / (2 * variance))
plt.plot(x, p, 'k', linewidth=2)
plt.savefig("ex1_1d.pdf")
plt.savefig("ex1_1d.png")
plt.show()

### 2D
mean = [0, 0] 
covariance_matrix = np.array([[1, 0.6], [0.6, 2]])

distribution = np.random.multivariate_normal(mean, covariance_matrix, 1000)

x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
x, y = np.meshgrid(x, y)

pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.view_init(30, 100)

ax.plot_surface(x, y, multivariate_normal.pdf(pos, mean, covariance_matrix), cmap='viridis', linewidth=0)
ax.plot_wireframe(x, y, multivariate_normal.pdf(pos, mean, covariance_matrix), rstride=5, cstride=5)

ax.scatter(x, y, -0.01, color='k', marker='.', alpha=0.1)
ax.contourf(x, y, multivariate_normal.pdf(pos, mean, covariance_matrix), zdir='z', offset=-0.01, cmap='viridis')

x_marginal = np.linspace(-4, 4, 100)
y_marginal = np.linspace(-4, 4, 100)

x_marginal_pdf = gaussian_kde(distribution[:, 0])
x_marginal_values = np.linspace(-4, 4, 100)

y_marginal_pdf = gaussian_kde(distribution[:, 1])
y_marginal_values = np.linspace(-4, 4, 100)

ax.bar(x_marginal_values, x_marginal_pdf(x_marginal_values), zs=-4, zdir='y', alpha=0.6)
ax.bar(y_marginal, y_marginal_pdf(y_marginal_values), zs=4, zdir='x', alpha=0.6)

plt.savefig("ex1_2d.pdf")
plt.savefig("ex1_2d.png")
plt.show()