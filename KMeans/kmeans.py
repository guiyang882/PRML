# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 20:28:43 2015

@author: fighter
"""
import numpy as np

def createGaussSample(mu,Sigma,Num):
    x, y = np.random.multivariate_normal(mu, Sigma, Num).T
    return np.array([x,y]).T

def getTestData():
    mu1 = [1,-1]
    mu2 = [5.5,-4.5]
    mu3 = [1,4]
    mu4 = [6,4.5]
    mu5 = [9,0.0]
    Sigma = [[1,0],[0,1]]
    Num = 200
    data1 = createGaussSample(mu1,Sigma,Num)
    data2 = createGaussSample(mu2,Sigma,Num)
    data3 = createGaussSample(mu3,Sigma,Num)
    data4 = createGaussSample(mu4,Sigma,Num) 
    data5 = createGaussSample(mu5,Sigma,Num)
    dataSet = np.vstack((data1,data2,data3,data4,data5))
    label = []
    for item in range(5):
        for index in range(Num):
            label.append(item)
    return dataSet,np.array(label)
    
# calculate Euclidean distance
def euclDistance(vector1, vector2):
	return np.sqrt(sum(np.power(vector2 - vector1, 2)))

# init centroids with random samples
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = np.zeros((k, dim))
    index = np.random.randint(0,numSamples,k)
    centroids = dataSet[index]
    return centroids

# k-means cluster
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    clusterAssment = np.mat(np.zeros((numSamples, 2)))
    clusterChanged = True

	## step 1: init centroids
    centroids = initCentroids(dataSet, k)
    print "Init Centroids"
    print centroids
    print
    
    while clusterChanged:
        clusterChanged = False
        for i in xrange(numSamples):
            minDist  = 100000.0
            minIndex = 0
            
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist  = distance
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist**2
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = np.mean(pointsInCluster, axis = 0)
    return centroids, clusterAssment

def show_Data(dataSet,labels,fig_name):
    plt.figure()
    for index in range(len(dataSet)):
        if labels[index] == 0:
            plt.scatter(dataSet[index][0],dataSet[index][1],color = 'r')
        elif labels[index] == 1:
            plt.scatter(dataSet[index][0],dataSet[index][1],color = 'g')
        elif labels[index] == 2:
            plt.scatter(dataSet[index][0],dataSet[index][1],color = 'b')
        elif labels[index] == 3:
            plt.scatter(dataSet[index][0],dataSet[index][1],color = 'y')
        elif labels[index] == 4:
            plt.scatter(dataSet[index][0],dataSet[index][1],color = 'c')
        else:
            pass
    plt.title(fig_name)
    plt.show()

def calc_error(dataSet,new_label,new_cluster):
    for index in range(len(new_cluster)):
        print "Class ", index
        print sum(new_label==index)
        
    delta_len = len(dataSet) / len(new_cluster)
    start,end = 0,delta_len
    old_cluster = []
    for index in range(len(new_cluster)):
        old_cluster.append(np.mean(dataSet[start:end],axis = 0))
        start = end
        end = end + delta_len
    old_cluster = np.array(old_cluster)
    old_cluster = np.sort(old_cluster,axis=0)
    new_cluster = np.sort(new_cluster,axis=0)
    print old_cluster
    print new_cluster
    diff = old_cluster - new_cluster
    print "The var of diff"
    print np.var(diff,axis=0)
    print 

if __name__ == "__main__":
    data,labels = getTestData()
    show_Data(data,labels,"Original Figure")
    centroids, clusterAssment = kmeans(data, 5)
    print "New Centroids"
    print centroids
    print 
    print "DataSet Label"
    print clusterAssment
    print 
    show_Data(data,clusterAssment[:,0],"KMeans Figure")
    calc_error(data,clusterAssment[:,0],centroids)
    