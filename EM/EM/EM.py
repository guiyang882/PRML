import os,sys,math
import numpy as np
import csv
import json

import matplotlib.pyplot as plt

def dist_mahalanobis(x,mu,sigma):
    sigma_inv = np.linalg.inv(sigma)
    d = mu.shape[0]
    return np.dot(np.dot((x-mu),sigma_inv),(x.T-mu.T).reshape(d,1))[0]

def p_Gaussian(x,mu,sigma):
    if x.shape != mu.shape:
        return -1
    if x.shape[0] != sigma.shape[0] or sigma.shape[0] != sigma.shape[1]:
        return -1
    d = mu.shape[0]
    sigam_det = np.linalg.det(sigma)
    A = 1/(math.pow(2*math.pi,d/2.0)*math.sqrt(sigam_det))
    ma_dist = dist_mahalanobis(x,mu,sigma)
    B = math.pow(math.e,-0.5*ma_dist)
    return A * B

class EMTools:
    def __init__(self):
        self.m_original_data = None
        self.m_config = {}

    def __del__(self):
        pass
    
    def init_Config(self,filePath):
        if os.path.exists(filePath) == False:
            print "The EM config parameters is Error !"
            sys.exit(1)
        handle = open(filePath,"r")
        j_data = json.load(handle)
        handle.close()
        self.m_config = j_data

    def show_data(self,data):
        if type(data) == np.ndarray:
            pass
        else:
            data = np.array(data)
        plt.figure(1)
        plt.scatter(data[:,0:1],data[:,1:2])
        plt.show()
    
    def normalize(self):
        max1,min1 = max(self.m_original_data[:,0:1]),min(self.m_original_data[:,0:1])
        max2,min2 = max(self.m_original_data[:,1:2]),min(self.m_original_data[:,1:2])
        self.m_original_data[:,0:1] = (self.m_original_data[:,0:1]-min1)/max1
        self.m_original_data[:,1:2] = (self.m_original_data[:,1:2]-min2)/max2

    ## read data from csv file,each line means a vector
    def readCloudPoint(self,filePath):
        if os.path.exists(filePath) == False:
            print "The Data File Not exists !"
            sys.exit(1)
        handle = open(filePath,"r")
        reader = csv.reader(handle)
        original_data = []
        index = 0
        for line in reader:
            index = index + 1
            if index == 1:
                continue
            cell = []
            for item in line:
                cell.append(float(item))
            original_data.append(cell)
        handle.close()
        self.m_original_data = np.array(original_data)
        self.normalize()
        #self.show_data(self.m_original_data)

    def membership_relationship(self,test_point):
        gaus_ik = []
        for i in range(0,self.m_config["num_class"]):
            pi_k = self.m_config["parameters"]["delta"][i]
            mu = np.array(self.m_config["parameters"]["data"][i]["mu"])
            sigma = np.array(self.m_config["parameters"]["data"][i]["sigma"])
            tmp = p_Gaussian(np.array(test_point),mu,sigma)
            gaus_ik.append(pi_k*tmp)
        w_ij = []
        for i in range(0,len(gaus_ik)):
            w_ij.append(gaus_ik[i]/sum(gaus_ik))
        return w_ij
    
    def Expection_Maximum(self):
        T = []
        for item in self.m_original_data:
            T.append(self.membership_relationship(item))
        T_array = np.array(T)
        sum_T = sum(T_array)
        mu1 = sum(T_array[:,0:1]*self.m_original_data)/sum_T[0]
        mu2 = sum(T_array[:,1:2]*self.m_original_data)/sum_T[1]
        # print (mu1,mu2)
        start_sigma1 = np.array([[0,0],[0,0]])
        start_sigma2 = np.array([[0,0],[0,0]])
        for index,item in enumerate(self.m_original_data):
            a,b = item-mu1,item-mu1
            a.shape,b.shape =(2,1),(1,2)
            start_sigma1 = start_sigma1 + T_array[index][0]*np.dot(a,b)
            start_sigma2 = start_sigma2 + T_array[index][1]*np.dot(a,b)
        expc_sigmas = (start_sigma1/sum_T[0],start_sigma2/sum_T[1])
        # print expc_sigmas
        new_delta = sum_T/len(T)
        self.m_config["parameters"]["delta"] = new_delta.tolist()
        self.m_config["parameters"]["data"][0]["mu"] = mu1.tolist()
        self.m_config["parameters"]["data"][1]["mu"] = mu2.tolist()
        self.m_config["parameters"]["data"][0]["sigma"] = expc_sigmas[0].tolist()
        self.m_config["parameters"]["data"][1]["sigma"] = expc_sigmas[1].tolist()
        print self.m_config

    def main(self):
        index = 0
        while index < self.m_config["parameters"]["termination"]:
            self.Expection_Maximum()
            index = index + 1

if __name__ == "__main__":
    obj = EMTools()
    obj.readCloudPoint("test.csv")
    obj.init_Config("em_config.json")
    obj.main()
