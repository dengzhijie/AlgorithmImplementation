#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @File: 折半查找.py
# @Author: dengzj
# @E-mail: 1104397829@qq.com

#有序数组
#输入数组和要寻找的数值，返回索引
#时间复杂度为O(logn)，空间复杂度为O(1)
def binary_search(nums,target):
    left=0
    right=len(nums)-1

    while left<right:
        mid = (left + right) // 2
        #print(mid)
        if nums[mid]==target:
            return mid
        elif nums[mid]<target:
            left=mid+1
        else:
            right=mid

    return False


if __name__=='__main__':
    nums=[1,2,3,4,5,6]
    target=4
    print(binary_search(nums,target))
