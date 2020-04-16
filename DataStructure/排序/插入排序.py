#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @File: 插入排序.py
# @Author: dengzj
# @E-mail: 1104397829@qq.com

#输入无序数组，返回有序数组
#时间复杂度为O(n^2)，是稳定排序
def insert_sort(nums):
    if len(nums)<2:
        return nums
    for i in range(0,len(nums)):
        for j in range(i-1,-1,-1):
            print(nums,i)
            if nums[i]>=nums[j]:
                nums.insert(j+1,nums.pop(i))
                break
            else:
                if j==0:
                    nums.insert(0,nums.pop(i))
    return nums
if __name__=='__main__':
    nums=[7,1]
    print(insert_sort(nums))