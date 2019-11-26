# -*- coding: utf-8 -*-

__date__ = "2019.03.22"
__author__ = "jeff"
__describe__ = "本脚本旨在完成最小编辑距离的书写"

import numpy as np



"""
    最小编辑距离：针对两个字符串 A 和 B；仅使用 delete(删除)、insert(插入) 和 substitution(替换) 三种操作，
                将字符串 A 变成和字符串 B 一样，在此过程中，所需操作的最少步骤，即为字符串 A 和 B 之间的最小编辑距离
    
    变种：可根据此三种操作的代价不同，分别定义不同的代价函数，亦即：三种操作发生的代价不同，在此可动态地寻找一个最优解
"""
def min_edit_distance(text1, text2):
    """

    :param text1:
    :param text2:
    :return:
    """
    if text1 is None:
        if isinstance(text2, (str, )):
            return len(text2)
        else:
            print("type error")
            return np.Inf

    if text2 is None:
        if isinstance(text1, (str, )):
            return len(text1)
        else:
            print("type error")
            return np.Inf

    distance_matrix = [[i + j for j in range(len(text2) + 1)] for i in range(len(text1) + 1)]

    for i in range(1, len(text1) + 1):
        for j in range(1, len(text2) + 1):
            if text1[i - 1] == text2[j - 1]:
                cost = 0
            else:
                cost = 1

            distance_matrix[i][j] = min(distance_matrix[i][j - 1] + 1,
                                        distance_matrix[i - 1][j] + 1,
                                        distance_matrix[i - 1][j - 1] + cost)
    return distance_matrix[-1][-1]


def weight_min_edit_distance(text1, text2, weights):
    """
        本函数旨在完成一个带权的最小编辑距离的编写
        注：此处的三种操作的代价不尽相同，将其代价置入weights列表中; 此处假设相同的操作代价相同，无需考虑具体操作的数值
    :param text1:
    :param text2:
    :param weights: list type, store weight of each manipulation, and ignore the same manipulation
    :return:
    """
    if text1 is None:
        if isinstance(text2, (str, )):
            return len(text2)
        else:
            print("type error")
            return np.Inf

    if text2 is None:
        if isinstance(text1, (str, )):
            return len(text1)
        else:
            print("type error")
            return np.Inf

    if not isinstance(weights, (str, list)):
        print("please given correctly weights")
        return np.Inf
    if len(weights) != 3:
        print("please given correctly length of weights")
        return np.Inf

    distance_matrix = [[i + j for j in range(len(text2) + 1)] for i in range(len(text1) + 1)]

    for i in range(1, len(text1) + 1):
        for j in range(1, len(text2) + 1):
            if text1[i - 1] == text2[j - 1]:
                substitution_cost = 0
            else:
                substitution_cost = weights[2]

            min_distance = min(distance_matrix[i][j - 1] + weights[0],
                               distance_matrix[i - 1][j] + weights[1],
                               distance_matrix[i - 1][j - 1] + substitution_cost)

            # distance_matrix[i][j] = min(distance_matrix[i][j - 1] + weights[0],
            #                             distance_matrix[i - 1][j] + weights[1],
            #                             distance_matrix[i - 1][j - 1] + substitution_cost)

            distance_matrix[i][j] = min_distance

    return distance_matrix[-1][-1]


def modify_weight_min_edit_distance(text1, text2, weights):
    """
        本函数旨在进一步改进带权最小编辑距离的代价函数
        可利用 26 的字母和 10 个数字的书写形式进行针对性地处理，分别建立字典，如在进行 substitution 操作时，"0" --> "o" 的代价
        小于 "a" --> "b" 的代价，这是因为在书写过程中，尤其是在手写字体中，"0" 和 "o" 很难分清，故而约定其代价小于常规的字符替换等
    :param text1:
    :param text2:
    :param weights:
    :return:
    """
    pass







if __name__ == "__main__":
    string1 = "abcabc"
    string2 = "acdbc"
    print(min_edit_distance(string1, string2))
    print(weight_min_edit_distance(string1, string2, [1, 1, 2]))






















