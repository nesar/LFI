import elfi
import scipy.stats as ss
import numpy as np

def simulator(mu, sigma, alpha, batch_size=1, random_state=None):
    mu, sigma, alpha = np.atleast_1d(mu, sigma, alpha)
    return ss.norm.rvs(mu[:, None], sigma[:, None], alpha[:, None], size=(batch_size, 100), random_state=random_state)

    # return np.sin(mu[:, None]) + np.exp(sigma[:, None])*alpha[:, None]

def mean(y):
    return np.mean(y, axis=1)

def var(y):
    return np.var(y, axis=1)


mu = elfi.Prior('uniform', -2, 4)
sigma = elfi.Prior('uniform', 1, 4)
alpha = elfi.Prior('uniform', 2, 5)

# Set the generating parameters that we will try to infer
mean0 = 1
std0 = 3
alpha0 = 3.5

# Generate some data (using a fixed seed here)
np.random.seed(20170525)
y0 = simulator(mean0, std0, alpha0)
print(y0)


# Add the simulator node and observed data to the model
sim = elfi.Simulator(simulator, mu, sigma, alpha, observed=y0)

# Add summary statistics to the model
S1 = elfi.Summary(mean, sim)
S2 = elfi.Summary(var, sim)


# Specify distance as euclidean between summary vectors (S1, S2) from simulated and
# observed data
d = elfi.Distance('euclidean', S1, S2)

#### Plot the complete model (requires graphviz)
# elfi.draw(d)



rej = elfi.Rejection(d, batch_size=10000, seed=30052017)
res = rej.sample(10000, threshold=.5)
print(res)



import matplotlib.pyplot as plt
res.plot_marginals()
plt.show()