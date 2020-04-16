#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @File: logiticRegression.py
# @Author: dengzj
# @E-mail: 1104397829@qq.com
import math
import random
import numpy as np
def sigmoid_lr(x):
    return 1/(1+np.exp(-x))
#输入训练数据集，标签，迭代方式，迭代轮数，学习率
def logitic_regression(dataset,labels,ITER_WAY,epochs,rate):
    # 定义回归系数,初始化为1
    q = np.ones(dataset.shape[1])
    #批量梯度下降
    if ITER_WAY==0:

        for i in range(epochs):
            for j in range(dataset.shape[1]):
                #记录总和
                temp=0
                for m in range(dataset.shape[0]):
                    temp+=(np.matmul(dataset[m],q)-labels[m])*dataset[m][j]
                q[j]-=rate/m*temp
        return q.transpose()
    #随机梯度下降
    if ITER_WAY==1:
        for i in range(epochs):
            for m in range(dataset.shape[0]):
                for j in range(dataset.shape[1]):
                    q[j]=q[j]-rate*(np.matmul(dataset[m],q)-labels[m])*dataset[m][j]
        return q.transpose()

    #mini-batch
    if ITER_WAY==2:
        batch_size=2
        for i in range(epochs):
            #可通过这种方式不断变化rate
            #rate = 0.01 +0.1/(1+j)
            for j in range(math.floor(dataset.shape[0]/batch_size)):
                for k in range(dataset.shape[1]):
                    temp=0
                    for l in range(batch_size):
                        temp += (np.matmul(dataset[j*batch_size+l], q) - labels[j*batch_size+l]) * dataset[j*batch_size+l][k]
                    q[k] -= rate / batch_size * temp
        return q.transpose()

if __name__ == '__main__':
    dataset=np.array([[1,2,3,4],[4,5,6,3],[3,4,5,2]])
    labels=np.array([1,0,1])
    print(logitic_regression(dataset,labels,0,100,0.01))