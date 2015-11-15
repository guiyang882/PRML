# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import csv
import math
import copy

def createData(filename):
    handle = open(filename,'r')
    reader = csv.reader(handle)
    data = []
    index = 0
    for line in reader:
        index = index + 1
        if index == 1:
            continue
        data.append([float(line[0]),float(line[1]),line[2]])
    handle.close()
    return np.array(data)
    
class Perception:
    def __init__(self,eta = 1.0,init_weight = None,iter_times = None):
        self.eta = eta
        self.iter_times = iter_times
        self.init_weight = init_weight
        self.iter = 0
        
    def train_batch(self,train_samples):
        tr_data = self.getTrainMatrix(train_samples)
        if self.init_weight == None:
            self.init_weight = np.ones(tr_data.shape[1])*0
        while True:
            self.iter = self.iter + 1            
            output = np.dot(tr_data,self.init_weight)
            if sum(output <=0) == 0:
                print "Final Weight ",self.init_weight
                print "Iter Times ",self.iter
                self.plot_info(tr_data,self.init_weight)                
                break
            modified = tr_data[output <= 0]
            delta_path = []
            for index in range(modified.shape[1]):
                delta_path.append(sum(modified[:,index]))
            self.init_weight = np.array(delta_path) * self.eta + self.init_weight

    def train_single(self,train_samples):
        tr_data = self.getTrainMatrix(train_samples)
        if self.init_weight == None:
            self.init_weight = np.ones(tr_data.shape[1])*0
        while True:
            self.iter = self.iter + 1
            start_weight = copy.copy(self.init_weight)
            for item in tr_data:
                result = np.dot(item,self.init_weight)
                if result <= 0.0:
                    self.init_weight = self.init_weight + item
            if sum(start_weight-self.init_weight) == 0.0:
                print "Final Weight ",self.init_weight
                print "Iter Times ",self.iter
                self.plot_info(tr_data,self.init_weight)
                break
            if self.iter >= 500:
                self.plot_info(tr_data,self.init_weight)
                print "Not Classification !"
                break
    
    def train_HoKashyap(self,train_samples):
        tr_data = self.getTrainMatrix(train_samples)
        if self.init_weight == None:
            self.init_weight = random.uniform(0.01,0.05,tr_data.shape[1])
            self.init_weight[0] = 0
        y_plus = np.dot(tr_data.T,tr_data)
        y_plus = np.dot(np.linalg.inv(y_plus),tr_data.T)
        b = random.uniform(0.1,0.5,len(tr_data))
        b_min = random.uniform(1,1,len(tr_data)) * 0.5
        
        while True:
            self.iter = self.iter + 1
            output = np.dot(tr_data,self.init_weight) - b
            for i in range(len(output)):
                output[i] = 0.5 * (output[i] + math.fabs(output[i]))
            b = b + 2 * self.eta * output
            self.init_weight = np.dot(y_plus,b)
            if sum(np.fabs(output) > b_min) == 0:
                print "Output ",np.fabs(output)
                print "Final Weight ",self.init_weight
                print "Iter Times ",self.iter
                self.plot_info(tr_data,self.init_weight)
                break
    
    def plot_info(self,train_samples,weight = None):
        for i in range(len(train_samples)):
            if train_samples[i][2] == 1.0:
                plt.scatter(train_samples[i,0],train_samples[i,1],marker="x",color='r')
            elif train_samples[i][2] == -1.0:
                plt.scatter(-1.0*train_samples[i,0],-1.0*train_samples[i,1],marker="o",color='b')
        if weight == None:
            pass
        else:
            x = np.linspace(-10,10,201)
            #print -weight[0]/weight[1],-weight[2]/weight[1]
            y = -weight[0]/weight[1] * x - weight[2]/weight[1]
            plt.plot(x,y,'g')
        plt.grid(True)
        plt.show()
    
    def getTrainMatrix(self,train_samples):
        data_matrix = []
        for item in train_samples:
            if item[-1] == 'c1':
                tmp = [float(x) for x in item[:-1]]
                tmp.append(1.0)
                data_matrix.append(tmp)
            elif item[-1] == 'c2':
                tmp = [-1.0*float(x) for x in item[:-1]]
                tmp.append(-1.0)
                data_matrix.append(tmp)
        return np.array(data_matrix)

if __name__ == "__main__":
    filename = "data.csv"
    train_samples = createData(filename)
    obj = Perception()
    obj.train_single(train_samples)
    #obj.train_batch(train_samples)
    #obj.train_HoKashyap(train_samples)