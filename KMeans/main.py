# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 20:28:43 2015

@author: fighter
"""
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_scatter(data,lables):
    fig = plt.figure()
    ax3D = fig.add_subplot(111, projection='3d')
    ax3D.scatter(data[:,0], data[:,1], data[:,2], s=10, marker='o') 
    plt.show()

def createDataSet():
    data = [[4,2,5],[10,5,2],[5,8,7],[1,1,1],[2,3,2],[3,6,9],[11,9,2],[1,4,6],[9,1,7],[5,6,7]]
    lables = ['A','A','A','B','B','B','C','C','C','C']
    return np.array(data),np.array(lables)

# calculate Euclidean distance
def euclDistance(vector1, vector2):
	return sqrt(sum(power(vector2 - vector1, 2)))

# init centroids with random samples
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    centroids[0,:] = dataSet[0]    
    centroids[1,:] = dataSet[3]
    centroids[2,:] = dataSet[6]    
    return centroids

# k-means cluster
def kmeans(dataSet, k):
	numSamples = dataSet.shape[0]
	# first column stores which cluster this sample belongs to,
	# second column stores the error between this sample and its centroid
	clusterAssment = mat(zeros((numSamples, 2)))
	clusterChanged = True

	## step 1: init centroids
	centroids = initCentroids(dataSet, k)

	while clusterChanged:
		clusterChanged = False
		## for each sample
		for i in xrange(numSamples):
			minDist  = 100000.0
			minIndex = 0
			## for each centroid
			## step 2: find the centroid who is closest
			for j in range(k):
				distance = euclDistance(centroids[j, :], dataSet[i, :])
				if distance < minDist:
					minDist  = distance
					minIndex = j
			
			## step 3: update its cluster
			if clusterAssment[i, 0] != minIndex:
				clusterChanged = True
				clusterAssment[i, :] = minIndex, minDist**2

		## step 4: update centroids
		for j in range(k):
			pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
			centroids[j, :] = mean(pointsInCluster, axis = 0)

	print 'Congratulations, cluster complete!'
	return centroids, clusterAssment

if __name__ == "__main__":
    data,lables = createDataSet()
    plot_scatter(data,lables)
    centroids, clusterAssment = kmeans(data, 3)
    print centroids
    print clusterAssment