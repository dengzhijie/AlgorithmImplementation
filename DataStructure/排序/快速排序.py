#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @File: 快速排序.py
# @Author: dengzj
# @E-mail: 1104397829@qq.com

#输入无序数组，返回有序数组
#时间复杂度为O(nlogn)，非稳定排序
def quick_sort(nums):
    if len(nums)==0 or len(nums)==1:
        return nums
    list_left=[i for i in nums[1:] if i<nums[0]]
    list_right=[i for i in nums[1:] if i>=nums[0]]
    return quick_sort(list_left)+[nums[0]]+quick_sort(list_right)

if __name__=='__main__':
    nums=[8,7,9,5,2,1,4,3]
    print(quick_sort(nums))