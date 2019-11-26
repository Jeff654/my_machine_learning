# -*- coding: utf-8 -*-


def order_by_negative_group_and_positive_group(list1):
    """
        subject describe: Rearrange positive and negative numbers using inbuilt sort function
                         Given an array of positive and negative numbers, arrange them such that all negative integers
                         appear before all the positive integers in the array without using any additional data structure
                         like hash table, arrays, etc. The order of appearance should be maintained
        对list进行排序，list中所有u元素都是整型，结果是将所有的负数按序（原序）放在所有正数前面（按序）

        解题思路：
            定义两个指针，一个指向负数，一个指向正数，然后遍历；
                如若满足条件且两个index相差为1，则交换着两个指针所对应的value
                否则，找出所有的连续的正数块，然后与下一个负数进行整块的交换
    :param
        list1: list类型，所有元素都是int类型的
    :return:
        list1: list类型
    """
    if list1:
        if isinstance(list1, (list, set)):
            list1 = list(list1)

        left, right = 0, 1
        while right < len(list1):
            if list1[left] > 0:
                if list1[right] < 0:
                    if right - left == 1:
                        list1[left], list1[right] = list1[right], list1[left]
                    else:
                        # list1[left], list1[left+1:right+1] = list1[right], list1[left:right+1]
                        positive_block = list1[left:right]          # save the continues positive block
                        list1[left] = list1[right]
                        for index in range(len(positive_block)):
                            list1[index + 1 + left] = positive_block[index]
                    left += 1
                    right += 1
                else:
                    while list1[right] > 0:
                        right += 1
            else:
                left += 1
                right += 1

        return list1


def collision(a, b, c, d, n):
    """
        subject describe: Given five numbers a, b, c, d and n (where a, b, c, d, n > 0). These values represent n terms of two series.
                                The two series formed by these four numbers are b, b+a, b+2a….b+(n-1)a and d, d+c, d+2c, ….. d+(n-1)c
                                These two series will collide when at any single point summation values becomes exactly the same for both
                                 the series.Print the collision point.
    :param
        a: int type or float type
        b: int type or float type
        c: int type or float type
        d: int type or float type
        n: int type or float type
    :return:
        first_collision_item: int type or float type

    example:
        a = 13, b = 23, c = 32, d = 18, n = 100
        result: 114

    """
    first_collision_item = 0
    for index in range(n):
        temp_item1 = b + index * a
        if (temp_item1 - d) % c == 0:
            first_collision_item = temp_item1
            break
    return first_collision_item


def find_next_greater_frequency_element(my_list):
    """
        subject describe: Given an array, for each element find the value of nearest element to the right which is having frequency greater
                            than as that of current element. If there does not exist an answer for a position, then make the value ‘-1’

        example:
            input: my_list = [1, 1, 2, 3, 4, 2, 1]
            output: result_list = [-1, -1, 1, 2, 2, 1, -1]

        解题思路：
            首先利用hash，对每个element进行计数，存放在一个列表中；然后遍历计数列表，找到比当前计数更大的最近的那个计数的index，
            对应于原list的index，取出原list的element即可
    :param
        my_list: list type of integer
    :return:
    """
    from collections import Counter
    if my_list:
        if not isinstance(my_list, list):
            return "input error"
        else:
            # statistic the number of element, and then convert to a dict
            counter = Counter(my_list)
            my_dict = dict(counter)
            frequency_list = [0] * len(my_list)
            for index in range(len(my_list)):
                current_count = my_dict.get(my_list[index])
                frequency_list[index] = current_count

            output_list = [0] * len(my_list)
            for index_i in range(len(frequency_list)):
                for index_j in range(index_i + 1, len(frequency_list)):
                    if frequency_list[index_i] < frequency_list[index_j]:
                        output_list[index_i] = my_list[index_j]
                        break
                    else:
                        continue
                else:
                    output_list[index_i] = -1
            return output_list



"""
    工程实践发现: copy.deepcopy()太过费时，运行时间较长，在此自定义deepcopy()函数来改写（代替）copy.deepcopy()函数实现深度拷贝
    函数_copy_list()、_copy_dict()、deepcopy()用于代替部分copy.deepcopy()函数来实现深度拷贝，copy.deepcopy()函数中参数memo
    存储了所有的拷贝对象，太过费时，在此进行改写
    注：此deepcopy()函数值实现了可变对象list和dict的拷贝，如需再扩展其他类型的拷贝功能，则再根据需求进行扩展
"""
_dispatcher = {}
def _copy_list(my_list, dispatch):
    # ret = my_list.copy()
    ret = copy.copy(my_list)
    for index, item in enumerate(ret):
        child_type = dispatch.get(type(item))
        if child_type:
            ret[index] = child_type(item, dispatch)
    return ret

def _copy_dict(my_dict, dispatch):
    # ret = my_dict.copy()
    ret = copy.copy(my_dict)
    for key, value in ret.items():
        child_type = dispatch.get(type(value))
        if child_type:
            ret[key] = child_type(value, dispatch)
    return ret

_dispatcher[list] = _copy_list
_dispatcher[dict] = _copy_dict



def deepcopy(value):
    child_type = _dispatcher.get(type(value))
    if not child_type:
        return value
    else:
        return child_type(value, _dispatcher)


def next_bigger_than_current(my_list):
    """
        subject describe: find the item bigger than current index
        eg:
            input: [2, 5, 9, 6, 3, 4, 8, 15, 12]
            output: [3, 6, 12 ,8, 4, 8, 12, 15, 12]

        解题思路：
            对每个当前的element计算与后面element的残差，并将大于0的与其index进行一起存储，将其存放在二元组里；
            然后对该残差列表按元组的第二个元素进行排序，并取出最小值所对应元组的第一个element（实际上对应着原list的index），最后将
            原list中所对应的index添加到新的list中
    :param
        my_list: list type
    :return:
    """
    if my_list:
        if isinstance(my_list, list):
            output_list = []
            for index, value in enumerate(my_list):
                local_right_residual_list = [(i, my_list[i] - my_list[index]) for i in range(index + 1, len(my_list)) if my_list[i] - my_list[index] > 0]
                if local_right_residual_list:
                    sorted_local_right_residual_list = sorted(local_right_residual_list, cmp=lambda x, y : cmp(x[1], y[1]))
                    min_index = sorted_local_right_residual_list[0][0]
                    output_list.append(my_list[min_index])
                    del local_right_residual_list
                else:
                    output_list.append(value)
            return output_list


def stack_and_queue(my_list):
    """
         subject describe: Print a singly linked list from start and end one by one.
                        Ex- 1->2->3->4->5->6
                        Output : 1,6,2,5,3,4
    :param my_list:
    :return:
    """
    from collections import deque
    if my_list:
        if isinstance(my_list, list):
            output_list = []
            double_queue = deque(my_list)
            count = 0
            while count < len(my_list):
                if count % 2 == 0:
                    output_list.append(double_queue.popleft())
                else:
                    output_list.append(double_queue.pop())
                count += 1
            return output_list


def cut_length_dynamic_programming(n_length, price_list):
    """
        Given a rod of length n inches and an array of prices that contains prices of all pieces of size smaller than n.
        Determine the maximum value obtainable by cutting up the rod and selling the pieces. For example, if length of
        the rod is 8 and the values of different pieces are given as following, then the maximum obtainable value is 22
        (by cutting in two pieces of lengths 2 and 6)

        example:
            n_length: 8
            price_list: [1, 5, 8, 9, 10, 17, 17, 20]

            output:
                value: 22
                pieces: 2 an 6

        思路：
            最优子结构：cut(n) = max(price[i] + cut(n - i))

    :param
        n_length: int type, it equals the length of price_list
    :param
        price_list: list type, store the corresponding price
    :return:
    """
    if n_length <= 0:
        return 0
    if price_list:
        max_value = -1
        for index in range(n_length):
            max_value = max(max_value, price_list[index] + cut_length_dynamic_programming(n_length - index - 1, price_list))
        return max_value


class Binary_tree_node:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data

def left_view_binary_tree(root, level, max_level):
    if not root:
        return None
    else:
        if max_level[0] < level:
            print "the left node is: ", root.data
            max_level[0] = level
        left_view_binary_tree(root.left, level + 1, max_level)
        left_view_binary_tree(root.right, level + 1, max_level)

def left_view(root):
    max_level = [0]
    left_view_binary_tree(root, 1, max_level)


def divide_by_37(number):
    """
        Given a large number n, we need to check whether it is divisible by 37. Print true if it is
        divisible by 37 otherwise False.

        解题技巧：
           将number从低位开始，每三个一组，不足三位的在开头补0，然后相加，循环递归此过程，直到最后不超过三位数
    :param number:
    :return:
    """
    if number:
        if not isinstance(number, str):
            number = str(number)
        if number.isdigit():
            length = len(number)
            if length % 3 == 0:
                pass
            elif length % 3 == 1:
                # 在开头添加两位0
                number = "00" + number
            else:
                number = "0" + number
            split_number_list = [int(number[index:index+3]) for index in range(0, len(number), 3)]
            sum_split_number = sum(split_number_list)
            if len(str(sum_split_number)) > 3:
                return divide_by_37(sum_split_number)
            else:
                if sum_split_number % 37 == 0:
                    return True
                else:
                    return False


def complement_integer(number):
    """
        以二进制补足number
    :param number:
    :return:
    """
    import math
    if number:
        if not isinstance(number, str):
            number = str(number)
        if number.isdigit():
            binary_bit_length = int(math.floor(math.log(long(number), 2))) + 1
            return ((1 << binary_bit_length) - 1) ^ long(number)




if __name__ == "__main__":
    # list1 = [-12, 11, -13, -5, 6, -7, 5, -3, -6]
    # print order_by_negative_group_and_positive_group(list1)

    # collision_item = collision(13, 23, 32, 18, 100)
    # print collision_item

    # my_list = [1, 1, 2, 3, 4, 2, 1]
    # result_list = find_next_greater_frequency_element(my_list)
    # print result_list

    import copy
    import msgpack
    import json
    import time

    import string

    # data = {'name': 'John Doe', 'ranks': {'sports': 13, 'edu': 34, 'arts': 45}, 'grade': 5}
    # shadow_copy_data = copy.copy(data)
    # self_define_copy_data = deepcopy(data)
    # deep_copy_data = copy.deepcopy(data)
    #
    # print id(data), id(shadow_copy_data), id(self_define_copy_data), id(deep_copy_data)
    #
    # data["hello"] = "jeff"
    # data["ranks"]["edu"] = 100
    # print data
    # print shadow_copy_data
    # print self_define_copy_data
    # print deep_copy_data

    # my_test_list = [2, 5, 9, 6, 3, 4, 8, 15, 12]
    # result = next_bigger_than_current(my_test_list)
    # print result

    # result = stack_and_queue(my_test_list)
    # print result

    # price = [1, 5, 8, 9, 10, 17, 17, 20]
    # value = cut_length_dynamic_programming(8, price)
    # print value


    # root = Binary_tree_node(1)
    # root.left = Binary_tree_node(2)
    # root.right = Binary_tree_node(3)
    # root.left.left = Binary_tree_node(4)
    # root.left.right = Binary_tree_node(5)
    # root.right.right = Binary_tree_node(6)
    #
    # left_view(root)


    # large_number = 123544837829489374878349
    # large_number = 8955795758
    # flag = divide_by_37(large_number)

    # print flag

    # number = 22
    # result = complement_integer(number)
    # print result









    # begin_time = time.time()
    # for index in range(100000):
    #     copy.deepcopy(data)
    # end_time1 = time.time()
    # print "deepcopy: ", end_time1 - begin_time
    # for index in range(100000):
    #     copy.copy(data)
    # end_time2 = time.time()
    # print "copy: ", end_time2 - end_time1
    # for index in range(100000):
    #     deepcopy(data)
    # end_time3 = time.time()
    # print "self define deepcopy: ", end_time3 - end_time2

    # print "-----------------------------------------------------------------------------"
    # text = "hello world welcome jeff"
    # print "the original text is: ", text, id(text)
    # time1 = time.time()
    # text1 = ""
    # for index in range(1000000):
    #     text1 = copy.copy(text)
    # time2 = time.time()
    # print "the shadow copy duration is: ", time2 - time1, text1, id(text1)
    # text2 = ""
    # for index in range(1000000):
    #     text2 = copy.deepcopy(text)
    # time3 = time.time()
    # print "the deep copy duration is: ", time3 - time2, text2, id(text2)
    # text3 = ""
    # for index in range(1000000):
    #     text3 = deepcopy(text)
    # time4 = time.time()
    # print "the self define deepcopy duration is: ", time4 - time3, text3, id(text3)

    import time

    begin_time = time.time()
    large_number = int(1e4)
    my_data = range(large_number)
    my_data = [str(item) for item in my_data]
    first_data = copy.copy(my_data)
    first_time = time.time()
    print "the first shallow copy time is: ", first_time - begin_time

    second_data = copy.deepcopy(my_data)
    second_time = time.time()
    # print "the second deep copy time is: ", second_time - first_time

    third_data = deepcopy(my_data)
    third_time = time.time()
    print "the third self copy time is: ", third_time - second_time
















