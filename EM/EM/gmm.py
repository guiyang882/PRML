import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import math
import matplotlib.mlab as mlab

def gaussian(x, mean, dev=0.1):
    # standard deviation, square root of variance
    return 1/math.sqrt(2*math.pi)/dev*math.exp(-(x-mean)**2/2/dev**2)

# Generating data
N=1000
a=0.3
sample1=random.normal(0, 0.1, size=N*a)
sample2=random.normal(3, 0.1, size=N*(1-a))
sample=np.concatenate([sample1,sample2])

hist, bin_edges = np.histogram(sample, bins=100)

# Learning parameters
max_iter = 50
# Initial guess of parameters and initializations
params = np.array([-1,1,0.5])

# EM loop
counter = 0
converged = False

plabel1=np.zeros(sample.shape)
plabel2=np.zeros(sample.shape)

counter=0
criterion=0.1
converged=False

while not converged and counter<100:
    counter+=1
    mu1, mu2, pi_1 = params

    # Expectation
    # Find the probabilty of labeling data points
    for i in range(len(sample)):
        cdf1=gaussian(sample[i], mu1)
        cdf2=gaussian(sample[i], mu2)

        pi_2=1-pi_1

        plabel1[i]=cdf1*pi_1/(cdf1*pi_1+cdf2*pi_2)
        plabel2[i]=cdf2*pi_2/(cdf1*pi_1+cdf2*pi_2)

    # Maximization
    # From the labeled data points, 
    # find mean through averaging (aka ML)
    mu1=sum(sample*plabel1)/sum(plabel1)
    mu2=sum(sample*plabel2)/sum(plabel2)
    pi_1=sum(plabel1)/len(sample)
    newparams=np.array([mu1, mu2, pi_1])
    print params

    # Convergence check
    if np.max(abs(np.asarray(params)-np.asarray(newparams)))<criterion:
        converged=True

    params=newparams

plt.title('Histogram of fake data')
plt.hist(sample,bins=100, normed=True)
x=np.linspace(sample.min(), sample.max(), 100)
plt.plot(x, mlab.normpdf(x,mu1, 0.1))
plt.plot(x, mlab.normpdf(x,mu2, 0.1))
plt.show()
