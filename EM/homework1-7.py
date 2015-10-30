#!/usr/bin/env python
# coding=utf-8

import numpy as np 
import math

def dist_euclidean(x,y):
    if x.shape != y.shape:
        print "Input illegale"
        return None
    d = x.shape[0]
    y = y.reshape(d,1)
    return np.dot(x,y)[0]

def dist_mahalanobis(x,mu,sigma):
    sigma_inv = np.linalg.inv(sigma)
    d = mu.shape[0]
    return np.dot(np.dot((x-mu),sigma_inv),(x.T-mu.T).reshape(d,1))[0]

def mean_array(data):
    n_col,n_row = 0,0
    if len(data.shape) == 1:
        n_col = data.shape[0]
        n_row = 1
    else:
        n_col = data.shape[0]
        n_row = data.shape[1]
    mean_vec = []
    for i in range(0,n_row):
        mean_vec.append(1.0*sum(data[:,i])/n_col)
    return np.array(mean_vec)

def p_Gaussian(x,mu,sigam):
    if x.shape != mu.shape:
        return -1
    if x.shape[0] != sigam.shape[0] or sigma.shape[0] != sigam.shape[1]:
        return -1
    d = mu.shape[0]
    sigam_det = np.linalg.det(sigam)
    A = 1/(math.pow(2*math.pi,d/2.0)*math.sqrt(sigam_det))
    ma_dist = dist_mahalanobis(x,mu,sigam)
    B = math.pow(math.e,-0.5*ma_dist)
    return A * B

def discriminant_Gaussian(x,mu,sigma,p_omega):
    if x.shape != mu.shape:
        return -1
    if x.shape[0] != sigma.shape[0] or sigma.shape[0] != sigma.shape[1]:
        return -1
    d = mu.shape[0]
    A1 = -0.5*dist_mahalanobis(x,mu,sigma)
    A2 = -0.5*d*math.log(2*math.pi) + -0.5*math.log(np.linalg.det(sigma)) + math.log(p_omega)
    return A1 + A2

def createGaussianSamples(mu,sigam):
    # first to judge the illegale
    d = mu.shape[0]
    if len(mu.shape) != 1:
        print "Mu input illegale"
    if sigma.shape != (d,d):
        print "Sigma input illegale"
    x = np.array([1,4,5]) 
    print p_Gaussian(x,mu,sigma)

if __name__ == "__main__":
    mu = np.array([1,2,2])
    sigma = np.array([[1,0,0],[0,5,2],[0,2,5]])
    createGaussianSamples(mu,sigma)
    x = np.array([1,2,3])
    y = np.array([2,3,4])
    print dist_euclidean(x,y)
    print mean_array(sigma)
    print discriminant_Gaussian(x,mu,sigma,0.3)
