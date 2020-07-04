#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @File: test.py
# @Author: dengzj
# @E-mail: 1104397829@qq.com

#仅模拟实现单个卷积层，单个全连接层的卷积神经网络
import skimage
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.data
# Reading the image
#chelsea方法读出来的是小猫图片
from numpy.core._multiarray_umath import ndarray

img = skimage.data.chelsea()
# Converting the image into gray.
img = skimage.color.rgb2gray(img)
#skimage.io.imshow(img)
#cv2.waitKey(0)
#img=np.array(img)
#不加plt.shaow不会有窗口
#plt.show()


#定义单个滤波对输入图片的卷积计算
#image为(64,64),filter为(3,3)这样的结构
#输出为（62，62）这种结构的feature_map
def conv_filter(image,filter,b):
    conv_res=np.zeros((image.shape[0]-filter.shape[0]+1,image.shape[1]-filter.shape[1]+1))
    for i in range(image.shape[0]-filter.shape[0]+1):
        for j in range(image.shape[1]-filter.shape[1]+1):
            image_region_cur=image[i:i+filter.shape[0],j:j+filter.shape[1]]
            #print(j,j+filter.shape[1],image.shape,image_region_cur.shape,filter.shape)
            #print(image_region_cur.shape,filter.shape)
            cur_sum=np.sum(image_region_cur*filter)
            #cur_sum = np.dot(image_region_cur ,filter)
            #print(cur_sum.shape,b.shape)
            conv_res[i][j]=cur_sum+b
    return conv_res



#filter结构为（3，3，3）或者（3，3，3，3）/最后一位对应于输入图片的通道数，如果是单通道的则对应前者结构，第一位为滤波的个数
# image为（224，224）/灰度,或者（224，224，3）最后一位代表有多个通道
#b偏置项，（1，3）
#卷积函数要保证图片的通道和滤波的通道一致
def conv(image,filter,b):
    #假设没有padding项，步长为1
    #定义储存卷积的结果
    feature_map=np.zeros((image.shape[0]-filter.shape[1]+1,image.shape[1]-filter.shape[2]+1,filter.shape[0]))
    for filter_num in range(filter.shape[0]):
        cur_filter=filter[filter_num]
        if len(cur_filter.shape)>2:
            conv_f_res=0
            for filter_channel in cur_filter.shape[2]:
                conv_f_res+=conv_filter(image[:,:,filter_channel],cur_filter[:,:,filter_channel])
        else:
            conv_f_res=conv_filter(image,cur_filter,b[0,filter_num])
        feature_map[:,:,filter_num]=conv_f_res+b[0,filter_num]
    return feature_map

#
def backward_conv(dz,rate,model,X):

    w=model['w_conv']
    b=model['b_conv']
    print(dz.shape)
    #填充0
    #
    temp=np.zeros((X.shape[0]+(w.shape[1]-1),X.shape[1]+(w.shape[2]-1),dz.shape[-1]))
    print('temp',temp.shape)
    for i in range(dz.shape[-1]):
        #dz[:,:,i]=np.pad(dz[:,:,i],((1,dz.shape[0]-1),(2,dz.shape[1]-1)),'constant', constant_values=(0,3))
        temp[w.shape[1]-1:w.shape[1]-1+dz.shape[0],w.shape[2]-1:w.shape[2]-1+dz.shape[1],:]=dz
    print('aaa',temp.shape)
    #for i in range(w.shape[0]):
    for j in range(w.shape[0]):
        w[j,:,:]=np.flip(w[j,:,:],0)
        w[j, :,  :] = np.flip(w[j, :,  :], 1)
    print('bbb',w.shape)
    dx=np.zeros(X.shape)
    for channel in range(temp.shape[-1]):

        for channel_w in range(w.shape[0]):
            for h in range(dx.shape[0]-(w.shape[1]-1)):
                for j in range(dx.shape[1]-(w.shape[2]-1)):
                    #dx[h,w,channel]+=conv_filter(temp[h:h+dz.shape[0],w:w+dz.shape[1],:],w[channel:,:,:,channel_w])
                    # print(temp[h:h+w.shape[1], j:j+w.shape[1], channel].shape,
                    #                                  w[channel_w, :, :].shape,b[0,channel_w].shape,b.shape)
                    dx[h,j ] += conv_filter(temp[h:h+w.shape[1], j:j+w.shape[1], channel],
                                                     w[channel_w, :, :],b[0,channel_w])
    print('cccc',dx.shape)
    print(X.shape,dz.shape)
    #dw=np.dot(X,dz)
    dw = np.zeros(w.shape)
    for i in range(dw.shape[0]):
        for j in range(dw.shape[1]):
            for k in range(dw.shape[2]):
                #print(X[j:j+dz.shape[0],k:k+dz.shape[1]].T[5].shape,dz[:,:,i][5].shape)
                #print(i,j,k)
                dw[i,j,k]=print(np.sum(X[j:j+dz.shape[0],k:k+dz.shape[1]]*dz[:,:,i]))
    #dw=X*dz
    print('ddddd',dw.shape)
    #dw=X*dz

    w-=rate*dw
    #
    #db=np.zeros((1,X.shape[-1]))
    col_dz = dz.reshape((dz.shape[-1], -1))
    col_dz = col_dz.T
    db = np.sum(col_dz, axis=0, keepdims=True)
    db=db.reshape((-1,dz.shape[-1]))
    b-=rate*db

    model['w_conv']=w
    model['b_conv']=b
    return dx




#激活函数
#测试通过
def relu(feature_map):
    relu_res=np.zeros(feature_map.shape)
    for k in range(feature_map.shape[-1]):
        for i in range(feature_map.shape[0]):
            for j in range(feature_map.shape[1]):
                relu_res[i][j][k]=max(feature_map[i][j][k],0)

    #或者可以用下列方式一行搞定
    #relu_res=np.where(feature_map<0,0,feature_map)
    return relu_res

def backward_relu(dz):
    dx=dz
    return dz


#默认stride的行列数相同
#池化层
def max_pooling(feature_map,size=10,stride=10):
    pool_res=np.zeros(((feature_map.shape[0]-size+1)//stride,(feature_map.shape[1]-size+1)//stride,feature_map.shape[-1]))

    remainder=((feature_map.shape[0]-size+1)%stride,(feature_map.shape[1]-size+1)%stride)
    #本次初步实现暂时不添加
    #argmax = np.zeros(out_h, out_w, fh * fw * C)

    for k in range(feature_map.shape[-1]):
        r=0
        for i in range((feature_map.shape[0]-size+1)//stride):
            for j in range((feature_map.shape[1]-size+1)//stride):
                pool_res[i,j,k]=np.max(feature_map[i*stride:i*stride+size,j*stride:j*stride+size])
    return pool_res,remainder


def backward_pooling(X,dz,remainder,size=10,stride=10):
    """
    反向传播
    Arguments:
    dz-- out的导数，shape与out 一致

    Return:
    返回前向传播是的input_X的导数

    进行上采样
    """
    #pool_size = h * w
    # dmax = np.zeros((dz.size, pool_size))
    # dmax[np.arange(arg_max.size), arg_max.flatten()] = dz.flatten()
    #
    # dx = col2im2(dmax, out_shape=X.shape, fh=h, fw=w, stride=stride)

    dx=np.zeros((dz.shape[0]*stride+remainder[0]+size-1,dz.shape[1]*stride+remainder[1]+size-1,dz.shape[-1]))
    print(dx.shape)

    for c in range(dz.shape[-1]):
        for h in range(dz.shape[0]):
            for w in range(dz.shape[1]):
                dx[size*h+remainder[0]][size*w+remainder[1]][c]=dz[h][w][c]


    return dx



#单个的连接层的实现
def full_connection(inputs,W,b):
    outputs=np.dot(inputs,W)+b
    return outputs
def backward_full_connection(X,dz,model,rate):
    '''

    :param dz:上层的导数
    :return:
    '''
    W=model['W_full_con']
    b=model['b_full_con']
    print(X.T.shape,dz.shape)

    dw=np.dot(X.T,dz)
    #按列相加，保持其二维特性
    #db=np.sum(dz, axis=0, keepdims=True)
    db=dz

    #print('aaaa')

    dx = np.dot(dz, W.T)
    #print('bbbb')
    print('asd',dx.shape,W.T.shape,X.shape)

    #dx的输出要对应上一层的shape

    dx = dx.reshape(X.shape)

    W = W - rate * dw
    b = b -  rate* db

    model['W_full_con']=W
    model['b_full_con']=b

    return dx





#多分类的softmax函数
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))
#两个softmax的效果等同
# def softmax1(input_X):
#     """
#     Arguments:
#         input_X -- a numpy array
#     Return :
#         A: a numpy array same shape with input_X
#     """
#     exp_a = np.exp(input_X)
#     sum_exp_a = np.sum(exp_a,axis=1)
#     sum_exp_a = sum_exp_a.reshape(input_X.shape[0],-1)
#     ret = exp_a/sum_exp_a
#     return ret

#softmax函数的反向传播，返回的是反向传播中下一层的输入
def backward_softmax(preds,labels):
    dx=preds-labels
    #dx 为len(labels)行，nclass列
    return dx


def loss_cross_entropy(label,preds):
    return -np.sum(label*np.log(preds))


def predict(input,model):

    out_conv=conv(input,model['w_conv'],model['b_conv'])
    out_relu=relu(out_conv)
    out_max_pooling=max_pooling(out_relu)
    full_connection_in = out_max_pooling.reshape(1, -1)
    out_full_connection=full_connection(full_connection_in,model['W_full_con'],model['b_full_con'])
    res=softmax(out_full_connection)
    return res
def single_convnet(input,model,labels,rate):

    #forward
    out_conv = conv(input, model['w_conv'], model['b_conv'])
    out_relu = relu(out_conv)
    out_max_pooling,remainder = max_pooling(out_relu)
    out_max_pooling1 = out_max_pooling.reshape(1, -1)
    out_full_connection = full_connection(out_max_pooling1, model['W_full_con'], model['b_full_con'])
    forward_res = softmax(out_full_connection)

    #back_forward

    output_back_softmax=backward_softmax(forward_res, labels)
    output_back_full_con=backward_full_connection(out_max_pooling1,output_back_softmax,  model, rate)
    output_back_full_con = output_back_full_con.reshape(out_max_pooling.shape)
    #print('1111', test_conv.shape)
    #test_conv = test_conv.reshape(output_pooling.shape)

    out_back_pooling = backward_pooling(out_relu, output_back_full_con, remainder)

    out_back_relu = backward_relu(out_back_pooling)

    res = backward_conv(out_back_relu, rate, model, input)

    return model

if __name__=='__main__':

    a=np.array([[2,3],[3,4]])
    labels=np.array([[0,0,0,0,1]])
    b=np.zeros((1,5))
    b[0,0]=a[1,1]
    print(b)
    #print(a[:,1])

    #5个中间神经元
    test=np.random.randn(2464,5)
    test_filter=np.array([[[0.2,0.3,0.3],[0.2,0.3,0.3],[0.2,0.3,0.3]],[[0.1,0.2,0.3],[0.2,0.1,0.3],[0.2,0.3,0.1]]])
    model={}
    model['W_full_con']=test
    model['b_full_con']=b

    model['w_conv']=test_filter
    model['b_conv']=np.zeros([1,2])
    #
    #
    # #img.shape (300, 451)
    # #test_filter.shape  (2, 3, 3)
    # print('img.shape',img.shape)
    # print('test_filter.shape',test_filter.shape)
    # test_conv=conv(img,model['w_conv'],model['b_conv'])
    # #out conv shape (298, 449, 2)
    # print('out conv shape',test_conv.shape)
    # input_conv=img
    # test_conv=relu(test_conv)
    # #out relu shape (298, 449, 2)
    # print('out relu shape',test_conv.shape)
    # input_pooling=test_conv
    # #print('dfg',output_pooling.shape)
    # test_conv,remainder=max_pooling(test_conv)
    # #out pooling shape (28, 44, 2)
    # print('out pooling shape',test_conv.shape)
    # output_pooling=test_conv
    #
    #
    # #print(test_conv.shape)
    # #将池化层的输出结果展开为一行n列的数组
    # #(1, 2464)
    # test_conv_in=test_conv.reshape(1,-1)
    # #test_conv=test_conv.flatten()
    # print(test_conv_in.shape)
    #
    #
    # b=np.random.randn(5)
    #
    #
    #
    # test_conv=full_connection(test_conv_in,test,b)
    # print(test_conv.shape)
    # test_conv=softmax(test_conv)
    # print(softmax(test_conv))
    # label=np.array([[0,0,0,0,1]])
    # test_conv=backward_softmax(test_conv,label)
    # #全联接层输出(1, 5)
    # print(test_conv)
    #
    #
    # test_conv=backward_full_connection(test_conv_in,test_conv,model,0.01)
    # #(1, 2464)
    # test_conv=test_conv.reshape(output_pooling.shape)
    # print('1111',test_conv.shape)
    # test_conv=test_conv.reshape(output_pooling.shape)
    #
    # test_conv=backward_pooling(input_pooling,test_conv,remainder)
    # print('2222',test_conv.shape)
    # test_conv=backward_relu(test_conv)
    # print('3333',test_conv.shape)
    # test_conv=backward_conv(test_conv,0.01,model,input_conv)


    single_convnet(img,model,labels,0.01)