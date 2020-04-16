#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @File: 顺序查找.py
# @Author: dengzj
# @E-mail: 1104397829@qq.com

#无序数组
#输入数组和要寻找的数值，返回索引
#时间复杂度为O(n)，空间复杂度为O(1)
def sequential_search(nums,target):
    for i in range(len(nums)):
        if nums[i]==target:
            return i

    return False

if __name__=='__main__':
    nums=[1,2,3,4,5]
    tar=8
    print(sequential_search(nums,tar))