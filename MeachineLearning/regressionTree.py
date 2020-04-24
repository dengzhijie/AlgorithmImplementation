#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @File: regressionTree.py
# @Author: dengzj
# @E-mail: 1104397829@qq.com

#cart回归树的实现
import numpy as np
#定义根据数值来分割数据集的方法，根据指定feature的数值将数据分成两个数据集
def split_data(dataset,feature,value):
    left_dataset=[data for data in dataset if data[feature]<=value]
    right_dataset=[data for data in dataset if data[feature]>value]
    return right_dataset,left_dataset
#计算平均值
def dataset_mean(dataset):
    list_data=[data[-1] for data in dataset]
    return np.mean(list_data)
#计算均方误差
def dataset_mean_error(dataset):
    mean_d=dataset_mean(dataset)
    list_data_error = [(data[-1]-mean_d)*(data[-1]-mean_d) for data in dataset]
    return np.mean(list_data_error)

#选择最优的划分特征和对应的值
#注意回归树的特征每次不用删减，同个特征在一次划分后还可以继续用其划分
def choose_best_feature(dataset,pos=(1,4)):
    m=len(dataset)
    n=len(dataset[0])
    list_d=[data[-1] for data in dataset]
    if len(list(set(list_d)))==1:
        return None,dataset_mean(dataset)
    #划分前的均方误差
    error_data=dataset_mean_error(dataset)
    best_error=float('inf')
    best_value=0
    best_feature=0
    for feature in range(n-1):
        set_value=[data[feature] for data in dataset]
        for value in set_value:
            right_d,left_d=split_data(dataset,feature,value)
            #
            if len(right_d)<pos[1] or len(left_d)<pos[1]:
                continue
            error_value_r=dataset_mean_error(right_d)
            error_value_l = dataset_mean_error(left_d)
            new_error=error_value_r+error_value_l
            if new_error<best_error:
                best_error=new_error
                best_feature=feature
                best_value=value
    #满足下列条件时，直接返回叶节点，即数据均值
    if (error_data-best_value)<pos[0]:
        return None,dataset_mean(dataset)
    right_d, left_d = split_data(dataset, best_feature, best_value)
    if len(right_d)<pos[2] or len(left_d)<pos[2]:
        return None,dataset_mean(dataset)
    return best_feature,best_value

def createRegressionTree(dataset):
    res_tree={}
    feat,val=choose_best_feature(dataset)

    if feat==None:
        return val
    res_tree['feature'] = feat
    res_tree['value'] = val
    right_d,left_d=split_data(dataset,feat,val)
    res_tree['left']=createRegressionTree(left_d)
    res_tree['right']=createRegressionTree(right_d)
    return res_tree



if __name__=='__main__':

    data_test = [
        [1, 0, 0,0],
        [0, 1, 0,0],
        [0, 0, 1,0],
        [0, 0, 0,1]
    ]
    a,b=split_data(data_test,1,0.5)
    print(a)
    print(b)
    print(choose_best_feature(data_test))
    print(createRegressionTree(data_test))