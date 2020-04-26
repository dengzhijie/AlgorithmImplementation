#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @File: k-means.py
# @Author: dengzj
# @E-mail: 1104397829@qq.com

#用np来表示数据类型，便于计算
import numpy as np
import math
a=np.array([[1,2],[3,4]])
b=np.array([[1,2],[3,4]])
#t=a[np.where(a[:,0]==1)]
m=np.where(a[:,0]==1)
print(b[m])
# print(t)
#print(np.mean(a,axis=0))
#print(a[:,0].A==1)
b=np.array([5,2])

# print(a-b)
# print(np.mat(np.zeros((3,4))))
#计算欧式距离
def dis_eclud(A,B):
    sub_AB=A-B
    #print(sub_AB.shape)
    #必须用np中的sqrt和sum函数才能对数组中每个元素进行计算
    return np.sqrt(np.sum(np.power(sub_AB,2)))
#随机分配k个质心
def rand_cent_k(dataset,k):
    n=np.shape(dataset)[1]
    cent_k=np.mat(np.zeros((k,n)))
    for j in range(n):
        minJ=min(dataset[:,j])
        rangeJ=float(max(dataset[:,j])-minJ)
        #生成k行1列的随机数
        cent_k[:,j]=minJ+rangeJ*np.random.rand(k,1)
    return cent_k
def k_means(dataset,k):
    m=np.shape(dataset)[0]
    print(m)
    n=np.shape(dataset)[1]
    #记录结果，所属簇号及簇距离
    result=np.mat(np.zeros((m,2)))
    cent_k=rand_cent_k(dataset,k)
    flag=True

    while flag:
        print(cent_k)
        for i in range(m):
            flag=False
            #所属簇号
            min_index_i=-1
            #到该簇距离
            min_dis_i=float('inf')
            for j in range(k):
                #print(j,cent_k[j],dataset[m])
                #print(m,dataset)
                dis_i_m=dis_eclud(dataset[i],cent_k[j])
                #print('aaa:',dis_i_m)
                if dis_i_m<min_dis_i:
                    min_dis_i=dis_i_m
                    min_index_i=j
            if min_index_i!=result[i,0]:

                flag=True
            result[i,:] = min_index_i,min_dis_i**2

        #更新cent_k
        for i in range(k):
            print(dataset)
            print(np.where(result[:,0]==i)[0])
            #将相同索引的数据筛选出来
            dataset_i=dataset[np.where(result[:,0]==i)[0]]
            #按行计算均值
            print('qqqqqqqq')
            print(cent_k[i,:])
            print(dataset_i)
            cent_k[i,:]=np.mean(dataset_i,axis=0)
    return cent_k,result

if __name__=='__main__':
    #print(dis_eclud(a,b))
    data=np.mat([[1,2],[3,4],[3,4.2],[1.2,2.2]])
    #print(rand_cent_k(data,2))
    a,b=k_means(data,2)
    print('a',a)
    print('b',b)
