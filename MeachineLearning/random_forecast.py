#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @File: random_forecast.py
# @Author: dengzj
# @E-mail: 1104397829@qq.com

#以分类决策树作为弱分类器

import numpy as np
#ID3分类树
import math
#计算给定数据集的香农熵值
def cal_shannon_ent(dataset):
    len_data=len(dataset)
    dict_label={}
    shannonEnt=0.0
    #for循环遍历数据元素，统计所有的标签总数
    for i in range(len_data):
        #拿到当前数据的标签
        label_i=dataset[i][-1]
        if dict_label.get(label_i):
            dict_label[label_i]+=1
        else:
            dict_label[label_i]=1
    #统计每种标签的占比，计算当前数据集的香农熵值
    for label in dict_label:
        pi_label=dict_label[label]/len_data
        shannonEnt-=pi_label*math.log(pi_label,2)
    return shannonEnt


#按照指定特征的特定值划分数据集
#输入'数据集、划分数据集的特征、对应的值',feature为对应特征所在的列
#输出根据指定特征的特定值划分出得数据集
#注意：按当前划分完后，返回集合中所有当前特征值都是相同的，在返回的结果中去除掉该特征值.注意不能直接对原数据进行删除操作
# 便于后续操作中循环遍历特征，不用再遍历计算过的特征，计算最好的数据集划分方式

def split_dataset(dataset,feature,value):
    split_data_value=[]
    for i in range(len(dataset)):
        if dataset[i][feature]==value:
            #这样操作引起接下来错误，所以不能直接删除原数据集中的值
            # dataset[i].pop(feature)
            # split_data_value.append(dataset[i])
            temp=dataset[i][:feature]
            temp.extend(dataset[i][feature+1:])
            split_data_value.append(temp)
    return split_data_value


#选择最优的划分特征
def choose_best_splitfeature(dataset):
    #计算特征总数
    nums_features=len(dataset[0])-1
    base_shannon_ent=cal_shannon_ent(dataset)
    best_shannon_ent_gain=0.0
    best_feature=-1
    for i in range(nums_features):
        #将该特征中所有元素存入列表

        list_label_i=[data[i] for data in dataset]
        print(list_label_i)
        #去重
        set_label_i=set(list_label_i)
        shannon_ent_i=0.0
        for label_value in set_label_i:
            #计算按照当前特征和特征值进行划分得到的集合
            split_i=split_dataset(dataset,i,label_value)
            shannon_ent_i_value=cal_shannon_ent(split_i)
            shannon_ent_i+=len(split_i)/len(dataset)*shannon_ent_i_value
        shannon_ent_gain=base_shannon_ent-shannon_ent_i
        if shannon_ent_gain>best_shannon_ent_gain:
            best_shannon_ent_gain=shannon_ent_gain
            best_feature=i
    return best_feature


#当决策结束时，返回当前集合中最多的标签值
def max_labels(list_labels):
    dict_labels={}
    for i in list_labels:
        if dict_labels.get(i):
            dict_labels[i]+=1
        else:
            dict_labels[i]=1
    max_label=list_labels[0]
    max_nums=0
    for i in dict_labels:
        if dict_labels[i]>max_nums:
            max_label=i
    return max_label

#labels可看作是所有特征所对应的名称
#递归实现决策树
#递归终止条件：1.所有类别完全相同2.遍历完所有特征
def create_decision_tree(dataset,labels):
    #首先判断递归终止条件
    list_labels=[data[-1] for data in dataset]
    #所有类别都相同,即只有一个相同标签元素时，直接返回该标签
    if len(list(set(list_labels)))==1:
        return list_labels[0]
    if len(dataset[0])==1:
        return max_labels(list_labels)

    #找到最好的划分特征
    best_feature=choose_best_splitfeature(dataset)
    label_best=labels[best_feature]
    res_decision_tree={label_best:{}}
    # 统计该特征下面的所有值
    list_value_best_feature = [data[best_feature] for data in dataset]
    set_value_best_feature=set(list_value_best_feature)

    labels.pop(best_feature)
    for value in set_value_best_feature:
        #[:]将labels重新赋值给新的变量，可防止影响到labels的值，=则把labels的地址赋值过去
        sub_labels=labels[:]
        res_decision_tree[label_best][value]=create_decision_tree(split_dataset(dataset,best_feature,value),sub_labels)

    return res_decision_tree
# 参数意思分别 是从a 中以概率P，随机选择3个, p没有指定的时候相当于是一致的分布，a为一维数组或int，如果是int，则生成随机样本，就好像a是np.arange(N)一样。
# a1 = np.random.choice(a=5, size=3, replace=False, p=None)
# 非一致的分布，会以多少的概率提出来
# a2 = np.random.choice(a=5, size=3, replace=False, p=[0.2, 0.1, 0.3, 0.4, 0.0])
# replacement 代表的意思是抽样之后还放不放回去，如果是False的话，那么出来的三个数都不一样，如果是
# True的话， 有可能会出现重复的，因为前面的抽的放回去了。

#print(np.random.choice(['a','b','c','d'],3,replace=True))
#dataset为数据，包括特征数据和标签。labels为特征名称的集合
def model_rf(dataset,labels,n_trees):
    trees=[]
    for i in range(n_trees):
        nums_datas,nums_features=np.array(dataset).shape
        nums_features-=1
        #随机选取数据和特征
        idxs_d = np.random.choice(nums_datas, nums_datas-1, replace=True)
        idxs_f = np.random.choice(nums_features, nums_features-1, replace=False)
        #print(idxs_d)
        #print(idxs_f)
        #np转list
        labels_list = labels[idxs_f].tolist()
        idxs_fl=np.append(idxs_f,-1)
        #print(dataset[idxs_d][:,idxs_fl])
        #a=dataset[idxs_d].tolist()
        dataset_list=dataset[idxs_d][:,idxs_fl].tolist()
        #labels_list=labels[idxs_f]
        #print('qq',dataset_list,labels_list)
        trees.append(create_decision_tree(dataset_list,labels_list))
    return trees
#单条数据的预测
def predict_rf(model,data_predict):
    res=[]
    for i in range(len(model)):
        if isinstance(model[i],str):
            res.append(model[i])
            continue
        # print(model[i].keys())
        # print(model[i][list(model[i].keys())[0]])
        value=model[i][list(model[i].keys())[0]]
        dict_index={'no surfing':0,'flippers':1,'others':2}
        # print(list(value.keys()))
        # print(value)
        temp=model[i]

        print(value)
        print(list(temp.keys())[0])
        print(dict_index[list(temp.keys())[0]])
        print(data_predict[dict_index[list(temp.keys())[0]]])
        print(value[str(data_predict[dict_index[list(temp.keys())[0]]])])
        #print(dict_index[list(model[i].keys())[0]])
        temp_value=value[str(data_predict[dict_index[list(temp.keys())[0]]])]
        #从模型存储的字典中遍历到最终的预测结果
        while temp_value not in ['yes','no']:
            value=temp_value[list(temp_value.keys())[0]]
            feature_value=data_predict[dict_index[list(temp_value.keys())[0]]]
            temp_value=value[str(feature_value)]
        res.append(temp_value)

    #返回列表中个数最多的元素，即为随机森林最终预测的标签
    return max(res, key=res.count)

if __name__=='__main__':
    #测试数据选取不当可能会存在报错，原因是随机选取数据时会导致某个特征不能完全包括所有的可能值，由此导致错误
    data_test = [
        [1, 1, 1,'yes'],
        [1, 1, 0,'yes'],
        [1, 0, 0,'no'],
        [0, 1, 0,'no'],
        [0, 1, 0,'no'],
        [1, 1, 1, 'yes'],
        [1, 1, 0, 'yes'],
        [1, 0, 0, 'no'],
        [0, 1, 0, 'no'],
        [0, 1, 0, 'no'],
        [1, 1, 1, 'yes'],
        [1, 1, 0, 'yes'],
        [1, 0, 0, 'no'],
        [0, 1, 0, 'no'],
        [0, 1, 0, 'no']
    ]
    labels = ['no surfing', 'flippers','others']
    res=model_rf(np.array(data_test),np.array(labels),10)
    #print(predict_rf(res,data_test))
    print('res',res)
    print(predict_rf(res,[1, 1, 1]))



