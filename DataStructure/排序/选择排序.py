#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @File: 选择排序.py
# @Author: dengzj
# @E-mail: 1104397829@qq.com

#输入无序数组，返回有序数组
#时间复杂度为O(n^2)，非稳定排序，空间复杂度O(1)，
def select_sort(nums):
    for i in range(len(nums)):
        temp_min=nums[0]
        index=0
        for j in range(1,len(nums)-i):
            if nums[j]<temp_min:
                temp_min,index=nums[j],j
        nums.append(nums.pop(index))
    return nums

if __name__=='__main__':
    nums=[2,1]
    print(select_sort(nums))