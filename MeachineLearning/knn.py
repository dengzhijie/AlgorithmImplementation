import numpy as np
def knn_classify(X,dataset,labels,k):
    #print(dataset.shape)
    dis_europe=[]
    for i in dataset:

        #计算欧式距离
        dis_europe.append(np.sqrt(((i-X)**2).sum()))
    #print(dis_europe)
    #np.argsort可建立排序后的索引列表
    dis_europe_sort=np.argsort(dis_europe)
    #print(dis_europe_sort)
    #建立前k个索引对应的标签统计
    dict_labels={}
    for i in range(k):
        if dict_labels.get(labels[i]):
            dict_labels[labels[i]]+=1
        else:
            dict_labels[labels[i]]=1
    max_labes_nums=0
    #定义返回值，即预测标签
    #res_label=''
    for i in dict_labels:
        if dict_labels[i]>max_labes_nums:
            max_labes_nums,res_label=dict_labels[i],i
    return res_label

if __name__=='__main__':
    # 测试数据
    data = np.array([
        [0.8, 0.5, 0.2],
        [0.7, 0.5, 0.1],
        [0.8, 0.4, 0.2],
        [0.2, 0.1, 0.2],
        [0.3, 0.1, 0.2],
        [0.5, 0.4, 0.2],
        [0.4, 0.5, 0.2],
        [0.5, 0.2, 0.4],

    ])
    labels = np.array(['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C'])

    input = np.array([0.6, 0.6, 0.3])

    predict_label=knn_classify(input,data,labels,5)
    print(predict_label)
