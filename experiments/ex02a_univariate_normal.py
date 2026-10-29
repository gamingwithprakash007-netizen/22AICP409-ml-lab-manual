import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.linspace(-3, 5, 150)
params = [(0, 1, "$N(0,1)$"), (2, 3, "$N(2,3)$"), (0, 0.2, "$N(0,0.2)$")]

fig, ax = plt.subplots(figsize=(6, 4))

for mean, variance, label in params:
    pdf = norm.pdf(x, loc=mean, scale=np.sqrt(variance))
    ax.plot(x, pdf, label=label)

ax.set_title("Univariate Normal Distributions")
ax.set_xlabel("$x$")
ax.set_ylabel("density: $P(x)$")
ax.set_xlim((-3, 5))
ax.set_ylim((0, 1))
ax.legend()
plt.show()
