#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @File: adaboost.py
# @Author: dengzj
# @E-mail: 1104397829@qq.com

#事实上adaboost的弱分类器可应用于任意分类器，本次实现仅仅用单层的决策树来实现弱分类器

import numpy as np
#第一步建立单层的决策树，作为弱分类器
#这种情况的单层分类器跟决策树的分类器稍微有所不同，可看作简化版本的决策树，
#假设数据的类别用【1，-1】表示，由于不是统计熵，直接跟结果比较，而结果有两类，所以要分别
# 遍历每一可能的类别。那么首先定义对数据的单个特征，单个阈值，单个方向的结果进行比较统计结果的函数
def single_classify(data,feature,threashval,oriention):
    res=np.ones((data.shape[0],1))
    #print(res[:,feature]<=threashval)
    for i in range(data.shape[0]):
        if oriention == 'l':
            if data[i][feature]<=threashval:

                res[i,0]=-1.0
        else:
            if data[i][feature] > threashval:
                res[i, 0] = -1.0
    #res=res[res[:,feature]<=threashval]=-0.1
    #print(res)
    return res
#开始建立单层决策树的弱分类器
#输入数据，标签，权重矩阵
def weak_classifer(data,labels,mataix_w):
    m,n=data.shape
    #print(m,n)
    num_steps=10
    dict_rec={}
    #定义最小误差值
    min_err=float('inf')
    #逐一遍历每个特征
    for feature in range(n):
        # 计算阈值
        list_threashval=data[:,feature]
        max_threashval=list_threashval.max()
        min_threashval=list_threashval.min()
        step_size=(max_threashval-min_threashval)/num_steps
        #print(num_steps)

        #逐一遍历每个方向
        for orient in ['l','r']:

            #逐一遍历阈值
            for i in range(-1,num_steps+1):
                #计算single_classify
                threashval=min_threashval + float(i) * step_size
                res_classify=single_classify(data,feature,threashval,orient)

                err=np.ones(labels.shape)
                for j in range(labels.shape[0]):
                    if labels[j][0]==res_classify[j][0]:
                        err[j][0]=0

                #根据不同数据权重计算，统计误差
                #print(mataix_w.T.shape,err.shape)
                err=np.sum(mataix_w*err)
                #print(err)
                if err<min_err:
                    min_err=err
                    #mataix_w
                    dict_rec['feature']=feature
                    dict_rec['orient']=orient
                    dict_rec['threashval']=threashval
    #记录最小误差时的feature，orient,threashval,以及误差的大小，样本的计算结果，并返回
    # 其中记录dict_rec是用来作为分类器对输入进行预测
    return dict_rec,err,res_classify
#定义adaboost模型
#iters为弱分类器的个数
def model_adaboost(data,labels,iters,matrix_w):
    #存储全部的弱分类器
    res_strong_classifer=[]

    for i in range(iters):
        print(matrix_w)
        dict_rec, err, res_classify=weak_classifer(data,labels,matrix_w)
        alpha=1/2*np.log((1-err)/max(err,np.exp(-16)))
        dict_rec['alpha']=alpha
        res_strong_classifer.append(dict_rec)
        #根据公示计算规范化因子

        Z=np.sum(matrix_w*np.exp(-alpha*labels*res_classify))
        matrix_w=matrix_w*np.exp(-alpha*labels*res_classify)/Z

    return res_strong_classifer

def cal_by_adaboost(data,strong_classifer):
    result=np.zeros((data.shape[0],1))
    for i in range(len(strong_classifer)):
        result_i=single_classify(data,strong_classifer[i]['feature'],strong_classifer[i]['threashval'],strong_classifer[i]['orient'])
        result+=strong_classifer[i]['alpha']*result_i
        print(result)
        return np.sign(result)


if __name__=='__main__':
    data=np.array([[1.0,2.1]
                   ,[2.0,1.1]
                   ,[1.3,1]
                   ,[1.0,1.0]
                   ,[2.0,1.0]])
    labels=np.array([1.0,1.0,-1.0,-1.0,1.0]).reshape(-1,1)
    #print(labels)
    matrix_w=np.ones((5,1))/5
    #print(matrix_w)
#single_classify(data,0,1.5,'left')
#print(weak_classifer(data,labels,matrix_w))
    #print(np.exp(-16))
    strong_classifer=model_adaboost(data,labels,9,matrix_w)
    result=cal_by_adaboost(data,strong_classifer)
    print(result)