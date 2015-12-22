# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
from kmeans import kmeans

def readData(filename):
    handle = open(filename,'r')
    dataSet = []
    for line in handle.readlines():
        seq_data = line.split(" ")
        seq_data[0] = float(seq_data[0])
        seq_data[1] = float(seq_data[1])
        dataSet.append(seq_data)
    return np.array(dataSet)
    
def createUnDirectedGraph(dataSet,k_neighbor,Sigma = 0.5):
    nSamples = len(dataSet)
    distArr = np.zeros([nSamples,nSamples])
    unDG = np.zeros([nSamples,nSamples])
    W_ij = np.zeros([nSamples,nSamples])
    for x in range(nSamples):
        for y in range(x+1,nSamples):
            distArr[x][y] = sum((dataSet[x] - dataSet[y]) ** 2)
            distArr[y][x] = distArr[x][y]
        # get the k_neighbors
        tmp = distArr[x,:]
        info_list = zip(range(0,nSamples),tmp)
        info_list = sorted(info_list,key = lambda x:x[1])       
        for index in range(1,k_neighbor+1):
            unDG[x][info_list[index][0]] = 1
            unDG[info_list[index][0]][x] = 1
            W_ij[x][info_list[index][0]] = np.exp(-1.0 * info_list[index][1] / (2.0 * Sigma * Sigma))
            W_ij[info_list[index][0]][x] = W_ij[x][info_list[index][0]]
    return distArr,unDG,W_ij
    
def showData(dataSet,clusterA,clusterB):
    plt.figure()
    for index in clusterA:
        plt.scatter(dataSet[index,0],dataSet[index,1],color = 'r')
    for index in clusterB:
        plt.scatter(dataSet[index,0],dataSet[index,1],color = 'b')
    plt.title("Show The Data")
    plt.show()

def getNormLaplacian(W):
    D = sum(W,axis=1) * np.eye(len(W))
    L = D-W
    Dn = np.power(np.linalg.matrix_power(D,-1),0.5)
    Lsym = np.dot(np.dot(Dn,L),Dn)
    return Lsym

def getKSmallestEigVec(Lsym,k):
    eigval,eigvec=linalg.eig(Lsym)
    dim=len(eigval)
    dictEigval=dict(zip(eigval,range(0,dim)))
    kEig=sorted(eigval,reverse=False)[0:k]
    ix=[dictEigval[i] for i in kEig]
    return eigval[ix],eigvec[:,ix]

def calc_Accu(clusterA,clusterB):
    N1,N2 = 0,0
    for index in clusterA:
        if index < 100:
            N1 = N1 + 1
    for index in clusterB:
        if index >= 100:
            N2 = N2 + 1
    print "The Accu is:"
    print (N1 + N2) * 1.0 / 200 * 100
    return (N1 + N2) * 1.0 / 200 * 100

if __name__ == "__main__":
    dataSet = readData("data.txt")
    nodeNum = len(dataSet)
    #showData(dataSet,range(0,nodeNum/2),range(nodeNum/2,nodeNum))
    result_info = []
    for Sigma in np.linspace(0.05,1,50):
        _,_,W_ij = createUnDirectedGraph(dataSet,4,Sigma)
        Lsym=getNormLaplacian(W_ij)
        _,kEigVec=getKSmallestEigVec(Lsym,2)
        info = sum(kEigVec**2,axis=1)**0.5   
        for index in range(nodeNum):
            if kEigVec[index][0] == 0 and kEigVec[index][1] == 0:
                pass
            else:
                kEigVec[index] = kEigVec[index] / info[index]
        _,clusterLabel = kmeans(kEigVec,2)
        clusterA,clusterB = [],[]
        for index in range(len(clusterLabel)):
            if clusterLabel[index][0] == 1:
                clusterB.append(index)
            elif clusterLabel[index][0] == 0:
                clusterA.append(index)
        accu = calc_Accu(clusterA,clusterB)
        result_info.append([Sigma,accu])
        #showData(dataSet,clusterA,clusterB)
    result_info = np.array(result_info)
    plt.plot(result_info[:,0],result_info[:,1])
    plt.title("The line Accu with Sigma, When K = 4")
    plt.xlabel("Sigma value")
    plt.ylabel("Accu info")
    plt.show()
    