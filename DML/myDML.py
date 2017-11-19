# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 24:00:00 2017

@author: Zhiyu
"""
# import your module here
import numpy as np
import time
from threading import Thread
import functools
import random
import math

# (global) variable definition here
TRAINING_TIME_LIMIT = 60 * 10

# class definition here

# function definition here
def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco

@timeout(TRAINING_TIME_LIMIT)
def train(traindata):
    sample = 36            # 采样集 
    step = 0.05           # 步长
    iteration = 1000       # 迭代次数

    train_X = traindata[0] # instances of training data
    train_Y = traindata[1] # labels of training data
    n, d = train_X.shape   
    global A
    A = np.eye(d, d)
    while iteration > 0 :
        #对原始数据进行采样
        sub_data_index = np.zeros((sample), dtype = np.int32)
        for i in range(sample):
            sub_data_index[i] = random.randint(0, n - 1)

        #获得P矩阵
        P = np.zeros((n, n))
        A_train_X = np.dot(A, train_X.T)
        for i in sub_data_index:
            Pi = np.exp(- np.linalg.norm(np.tile(A_train_X[:,i], (n, 1)).T - A_train_X, axis = 0))
            Pi = Pi / (np.sum(Pi) - 1.0 + 0.000000000001) #加一个很小的数，以防分母为0
            P[i] = Pi
            P[i][i] = 0

        #梯度下降
        sum_i = np.zeros((d, d))
        for i in sub_data_index:
            sum_before = np.zeros((d, d))
            sum_after = np.zeros((d, d))
            Pi = 0.0
            for j in range(n):
                temp = P[i][j] * np.outer(train_X[i] - train_X[j], train_X[i] - train_X[j])
                sum_before += temp
                if train_Y[i] == train_Y[j]:
                    sum_after += temp
                    Pi += P[i][j]
            sum_i += Pi * sum_before - sum_after
        A += step * 2 * np.dot(A, sum_i)
        iteration -= 1
    return 0

def Euclidean_distance(inst_a, inst_b):
    return np.linalg.norm(inst_a - inst_b)

def distance(inst_a, inst_b):
    dist = np.linalg.norm(np.dot(A, inst_a) - np.dot(A, inst_b)) 
    return dist

# main program here
if  __name__ == '__main__':
    pass
