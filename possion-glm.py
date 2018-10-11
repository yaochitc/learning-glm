import numpy as np
import pandas as pd
import pymc3 as pm

import matplotlib.pyplot as plt

# decide poisson theta values
theta_noalcohol_meds = 1    # no alcohol, took an antihist
theta_alcohol_meds = 3      # alcohol, took an antihist
theta_noalcohol_nomeds = 6  # no alcohol, no antihist
theta_alcohol_nomeds = 36   # alcohol, no antihist

# create samples
q = 1000
df = pd.DataFrame({
        'nsneeze': np.concatenate((np.random.poisson(theta_noalcohol_meds, q),
                                   np.random.poisson(theta_alcohol_meds, q),
                                   np.random.poisson(theta_noalcohol_nomeds, q),
                                   np.random.poisson(theta_alcohol_nomeds, q))),
        'alcohol': np.concatenate((np.repeat(False, q),
                                   np.repeat(True, q),
                                   np.repeat(False, q),
                                   np.repeat(True, q))),
        'nomeds': np.concatenate((np.repeat(False, q),
                                      np.repeat(False, q),
                                      np.repeat(True, q),
                                      np.repeat(True, q)))})

with pm.Model() as model:
    fml = 'nsneeze ~ alcohol * nomeds'
    pm.glm.GLM.from_formula(fml, df, family=pm.glm.families.Poisson())
    trace = pm.sample(2000, tune=2000, cores=2)

pm.traceplot(trace)
pm.plot_posterior(trace)

plt.show()

