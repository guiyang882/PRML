#!/usr/bin/env python
# coding=utf-8

import random
import numpy as np
import matplotlib.pyplot as plt
import math

def random_data_1(bound_low,bound_high,n,data_type):
    vec = []
    for i in range(0,n):
        a = 0
        if data_type == "double":
            a = random.uniform(bound_low,bound_high)
        if data_type == "int":
            a = random.randint(bound_low,bound_high)
        vec.append(a)
    return vec

def random_data_2(bound_low,bound_high,n,data_type):
    bl = random.randint(bound_low,bound_high)
    bh = random.randint(bound_low,bound_high)
    return random_data_1(min(bl,bh),max(bl,bh),n,data_type)

def calc_mean_sigma(data):
    mean_val = 1.0*sum(data)/len(data)
    sum_val = 0
    for i in range(0,len(data)):
        sum_val = sum_val + data[i]*data[i]
    mean_val2 = 1.0*sum_val/len(data)
    return  mean_val2 - mean_val

def create_Random_Sample(n):
    data = []
    for i in range(0,int(math.pow(10,n))):
        data.extend(random_data_2(-100,100,100,"int"))
    return data

def histogram_plot(data,bound_low,bound_high):
    pos_y = []
    for i in range(0,bound_high-bound_low+1):
        pos_y.append(0)
    for i in range(0,len(data)):
        pos_y[data[i]+100] = pos_y[data[i]+100] + 1
    plt.plot(np.linspace(bound_low,bound_high,bound_high-bound_low+1),pos_y)
    plt.xlabel("from -100 to 100")
    plt.ylabel("statistical times")
    plt.title("Statistical Frequency Distribution")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    bound_low,bound_high = -100,100
    print random_data_2(-100,100,10,"int")
    data = create_Random_Sample(3)
    histogram_plot(data,bound_low,bound_high)
