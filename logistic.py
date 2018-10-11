import numpy as np
import pandas as pd
import pymc3 as pm

import matplotlib.pyplot as plt

data = pd.read_csv("data/adult.data", header=None, names=['age', 'workclass', 'fnlwgt',
                                                          'education-categorical', 'educ',
                                                          'marital-status', 'occupation',
                                                          'relationship', 'race', 'sex',
                                                          'captial-gain', 'capital-loss',
                                                          'hours', 'native-country',
                                                          'income'])

data = data[~pd.isnull(data['income'])]

income = 1 * (data['income'] == " >50K")
age2 = np.square(data['age'])

data = data[['age', 'educ', 'hours']]
data['age2'] = age2
data['income'] = income

with pm.Model() as logistic_model:
    # Define priors
    intercept = pm.Normal('intercept', 0, sd=20)

    beta = pm.Normal('beta', 0, sd=1, shape=4)

    # Define likelihood
    likelihood = pm.Bernoulli('y',
                              pm.math.sigmoid(intercept + beta[0] * data['age'] + beta[1] * data['educ']
                                              + beta[2] * data['hours'] + beta[3] * data['age2']),
                              observed=data['income'])
    trace = pm.sample(2000, chains=1, tune=1000)

pm.traceplot(trace)

plt.show()
