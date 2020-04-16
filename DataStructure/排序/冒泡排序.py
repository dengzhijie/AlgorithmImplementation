#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @File: 冒泡排序.py
# @Author: dengzj
# @E-mail: 1104397829@qq.com

#输入无序数组，返回有序数组
#时间复杂度为O(n^2)，空间复杂度O(1)，是稳定排序
def bubble_sort(nums):
    for i in range(len(nums)):
        for j in range(len(nums)-i-1):
            if nums[j]>nums[j+1]:
                nums[j],nums[j+1]=nums[j+1],nums[j]
    return nums
if __name__=='__main__':
    nums=[8,5,6,2,3,4,1]
    print(bubble_sort(nums))