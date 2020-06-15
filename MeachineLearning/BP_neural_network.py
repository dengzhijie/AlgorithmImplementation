#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @File: BP_neural_network.py
# @Author: dengzj
# @E-mail: 1104397829@qq.com

#实现简单的三层神经网络
#假设输入是个两维的变量，隐含层包括两个神经元，输出层为1个神经元

import numpy as np
#输入维度
input_nn_nums=2
#隐层神经元个数
hide_nn_nums=2
#输出层神经元个数
output_nn_nums=1


#定义相关函数
def sigmoid(x):
    return 1/(1+np.exp(-x))

def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length
    return ((y_true - y_pred) ** 2).mean()

#定义神经网络前向传播架构
#输入为要预测的输入值和模型参数,字典格式
def forword(X,model):
    #len(X)*hide_nn_nums
    W1=model['W1']
    #b1的维度跟第一个隐含层神经元个数一致
    b1=model['b1']
    #hide_nn_nums*output_nn_nums
    W2=model['W2']
    #三层神经网络，b2维度跟输出层神经元个数一致
    b2=model['b2']
    h=np.dot(W1,X)+b1
    #print(h)
    h_sigmoid=sigmoid(h)
    #print(h_sigmoid)
    output=np.dot(W2,h_sigmoid)+b2
    #print(output)
    #前向预测结果,这里讨论的输出结果为单个神经元
    pred=sigmoid(output)
    #print(pred)
    return pred[0]

#in 代表输入神经元个数，inner为中间隐层神经元个数，out为输出神经元，默认1
#采用随机梯度下降
def train(X,Y,input,inner,model,epoch_nums,rate,out=1):
    W1=model['W1']
    W2 = model['W2']
    b1 = model['b1']
    b2 = model['b2']

    for epoch in range(epoch_nums):
        #print(W1,b1,W2,b2,forword())
        #print(model)
        for index in range(len(X)):
            # model['W1']=W1
            # model['W2']=W2
            # model['b2']=b1
            # model['b2']=b2
            y_pred=forword(X[index],model)

            #用于后续迭代计算
            d_ypred = -2 * (Y[index] - y_pred)
            #遍历w1中每个参数
            sum_h=np.dot(W1,X[index])+b1
            h_sigmoid = sigmoid(sum_h)
            sum_o=np.dot(W2,h_sigmoid)+b2
            o1 = sigmoid(sum_o)

            d_h1_d_w = deriv_sigmoid(sum_h)

            #对应W2
            d_ypred_w=h_sigmoid*deriv_sigmoid(sum_o[0])
            d_ypred_b = deriv_sigmoid(sum_o[0])

            d_ypred_h=W2*deriv_sigmoid(sum_o[0])
            d_h1_w=[]
            d_h1_b=[]
            for i in range(inner):

                d_h1_w.append(X[index] * deriv_sigmoid(sum_h[i]))
                d_h1_b.append(deriv_sigmoid(sum_h[i]))
            #更新W1和b1
            for i in range(inner):

                #第一层神经元对应的第i个神经元输出结果
                #'sum_h1_'+str(i) =np.dot(W1[i],X[index])+b1[i]
                #W1[i]
                for j in range(input):
                    #sum_h1 = W1[i]   self.w1 * x[0] + self.w2 * x[1] + self.b1
                    W1[i][j]-=rate*d_ypred*d_ypred_h[i]*d_h1_w[i][j]
                    b1[i]-=rate*d_ypred*d_ypred_h[i]*d_h1_b[i]
            #更新w2和b2
            for i in range(inner):
                #print(d_ypred,d_ypred_w[i],d_ypred_b)
                W2[i]-=rate*d_ypred*d_ypred_w[i]
                b2-=rate*d_ypred*d_ypred_b
        if epoch % 2 == 0:
            y_preds = np.apply_along_axis(forword, 1, X,model)
            loss = mse_loss(Y, y_preds)
            print("Epoch %d loss: %.3f", (epoch, loss))





    return model

if __name__=='__main__':
    x=np.array([[2,3],[-2,-4],[-3,-5],[-4,-2],[3,4],[2,4],[-3,-9],[3,5]])
    y=np.array([1,0,0,0,1,1,0,1])
    model={
        'W2':np.array([0.1,0.2]),
        'W1': np.array([[0.1, 0.2],[0.3,0.4]]),
        'b1':np.array([0.30,0.40]),
        'b2':np.array([0.50])

    }
    #print(np.array([[1, 2],[3,4]]).transpose())
    #print(forword(x,model))
    model=train(x,y,2,2,model,200,0.1,1)
    for i in x:
        print(forword(i,model))




