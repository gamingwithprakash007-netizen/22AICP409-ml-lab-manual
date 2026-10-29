import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def generate_surface(mean, cov):
    x, y = np.linspace(-5, 5, 100), np.linspace(-5, 5, 100)
    x1, x2 = np.meshgrid(x, y)
    pos = np.dstack((x1, x2))
    pdf = multivariate_normal(mean=mean, cov=cov).pdf(pos)
    return x1, x2, pdf

distributions = [
    (np.array([0, 0]), np.eye(2)),
    (np.array([0, 0]), np.array([[1, -0.1], [-0.8, 1]]))
]

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
plt.suptitle("Bivariate Normal Distributions")
ax[0].set_title("Independent Variables")
ax[1].set_title("Correlated Variables")

for i, dist in enumerate(distributions):
    x1, x2, pdf = generate_surface(dist[0], dist[1])
    con = ax[i].contourf(x1, x2, pdf, 100, cmap='rainbow')
    ax[i].set_xlabel("$x_1$", fontsize=13)
    ax[i].set_ylabel("$x_2$", fontsize=13)
    ax[i].axis([-3.5, 3.5, -3.5, 3.5])
    c_bar = fig.colorbar(con, ax=ax[i])
    c_bar.ax.set_ylabel("$P(x_1, x_2)$", fontsize=13)

plt.show()
