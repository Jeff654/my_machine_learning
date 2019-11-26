# -*- coding: utf-8 -*-


__data__ = "2019.03.22"
__author__ = "jeff"
__describe__ = """
        

"""

class Solution(object):
    """
        Given an array of integers, return indices of the two numbers such that they add up to a specific target.

        You may assume that each input would have exactly one solution, and you may not use the same element twice.

        Example:

            Given nums = [2, 7, 11, 15], target = 9,

            Because nums[0] + nums[1] = 2 + 7 = 9,
            return [0, 1].
    """
    @staticmethod
    def twoSum(nums, target):
        """
        :param nums: list type
        :param target: int or float number
        :return: list type
        """
        if not nums or target is None:
            return []

        if not isinstance(nums, (tuple, list)):
            return []
        if not isinstance(target, (float, int)):
            return []

        if len(nums) < 2:
            return []

        for index, item in enumerate(nums):
            remain_value = target - item
            if remain_value in nums:
                other_index = nums.index(remain_value)
                if index == other_index:
                    continue
                return [index, other_index]
        return []



class Solution1(object):
    """
        You are given two non-empty linked lists representing two non-negative integers. The digits are stored in
        reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.
        You may assume the two numbers do not contain any leading zero, except the number 0 itself.
        Example:
            Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
            Output: 7 -> 0 -> 8
            Explanation: 342 + 465 = 807.
    """
    @staticmethod
    def addTwoNumbers(l1, l2):
        """
            本函数旨在针对两个链表进行加减法
        :param l1:
        :param l2:
        :return:
        """
        if l1 is None or l2 is None:
            return []
        if not isinstance(l1, (tuple, list)) or not isinstance(l2, (tuple, list)):
            return []
        if len(l1) != len(l2):
            return []

        l1.reverse()
        digit1 = ""
        for item in l1:
            digit1 += str(item)

        l2.reverse()
        digit2 = ""
        for item in l2:
            digit2 += str(item)

        result_value = ""
        if digit1.isdigit() and digit2.isdigit():
            result_value = str(int(digit1) + int(digit2))
        # return [item for item in list(reversed(result_value))]
        results = []
        for index in range(len(result_value) - 1, -1, -1):
            results.append(result_value[index])
        return results












if __name__ == "__main__":
    # my_array = [3, 2, 4]
    # target_value = 6
    # print Solution.twoSum(my_array, target_value)

    a = [2, 4, 3]
    b = [5, 6, 4]
    print(Solution1.addTwoNumbers(a, b))





























