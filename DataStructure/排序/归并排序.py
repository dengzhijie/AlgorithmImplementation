#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @File: 归并排序.py
# @Author: dengzj
# @E-mail: 1104397829@qq.com

# 归并排序：采用分治法（Divide and Conquer）的一个非常典型的应用。将已有序的子序列合并，
# 得到完全有序的序列；即先使每个子序列有序，再使子序列段间有序。若将两个有序表合并成一个有序表，称为二路归并
#
# 时间复杂度：O(nlog₂n)
# 空间复杂度：O(n)
# 稳定性：稳定
#递归实现
def merge_sort(array):
    def merge_arr(arr_l, arr_r):


        array = []
        while len(arr_l) and len(arr_r):
            if arr_l[0] <= arr_r[0]:
                array.append(arr_l.pop(0))
            elif arr_l[0] > arr_r[0]:
                array.append(arr_r.pop(0))
        if len(arr_l) != 0:
            array += arr_l
        elif len(arr_r) != 0:
            array += arr_r
        return array

    def recursive(array):
        if len(array) == 1:
            return array

        mid = len(array) // 2
        arr_l = recursive(array[:mid])
        arr_r = recursive(array[mid:])
        return merge_arr(arr_l, arr_r)
    return recursive(nums)
if __name__=='__main__':
    nums=[3,2,1]
    print(merge_sort(nums))