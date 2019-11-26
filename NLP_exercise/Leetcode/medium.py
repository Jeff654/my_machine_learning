# -*- coding: utf-8 -*-

# import numpy as np

import re
import copy
import collections
class Solution(object):
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        max_value, min_value = 2 << 31 - 1, -2 << 31
        text = self.preprocess(str)
        if isinstance(text, basestring):
            if len(text) == 0:
                return 0
            if str.isdigit():
                if int(text) < max_value:
                    return int(text)
            else:
                if text[0] == "+":
                    my_integer = self.get_first_integer(text[1:])
                    if my_integer > max_value:
                        return max_value
                    return my_integer
                elif text[0] == "-":
                    my_integer = self.get_first_integer(text[1:])
                    if my_integer > abs(min_value):
                        return min_value
                    return -my_integer

                elif text[0].isdigit():
                    my_integer = self.get_first_integer(text)
                    if min_value < my_integer < max_value:
                        return my_integer
                    elif my_integer > max_value:
                        return max_value
                    else:
                        return min_value
                else:
                    return 0


    def preprocess(self, text):
        """
            pre_process my data
        :param text:
        :return:
        """
        if isinstance(text, str):
            my_text = ""
            for index in range(len(text)):
                if text[index] == " ":
                    continue
                else:
                    my_text = text[index:]
                    break
            return my_text
        return text


    def get_first_integer(self, text):
        """
            slide the data from begin
        :param text:
        :return:
        """
        if isinstance(text, str):
            my_integer = 0
            my_index = -1
            for index in range(len(text)):
                if text[index].isdigit():
                    my_index = index
                    continue
                else:
                    my_index = index - 1
                    break
            if my_index > -1:
                my_integer = text[:my_index + 1]
            return int(my_integer)
        return 0



class Solution1(object):
    """
        本类旨在解决指定一个words集合和一串字符str，拼接后的words是否是str的子串
    """
    def findSubstring(self, s, words):
        """
        :type s: str
        :type words: List[str]
        :rtype: List[int]
        """
        if not s or not words:
            return []

        count_words = collections.Counter(words)
        words_total_length = len("".join(words))
        result = []

        for index in range(len(s) - words_total_length + 1):
            local_index = index
            words_dict = dict(count_words)
            for index_2 in range(len(words)):
                current_word = s[local_index: local_index + len(words[index_2])]
                if current_word in words_dict:
                    if words_dict.get(current_word, 0) == 1:
                        words_dict.pop(current_word)
                    else:
                        words_dict[current_word] -= 1
                else:
                    break
                local_index += len(current_word)
            if not words_dict:
                result.append(index)
        return result

    def findSubstring2(self, s, words):
        """

        :param s:
        :param words:
        :return:
        """
        if not s or not words:
            return []
        result = []
        from collections import Counter
        for index in range(len(s) - len("".join(words)) + 1):
            my_dict = dict(Counter(words))
            local_bool_list = self.get_bool_list(index, s, words)
            # indexes = self.get_continue_one_sum(local_bool_list, len("".join(words)))
            result_list = self.get_total_number(my_dict)
            indexes = self.get_continue_one_sum(local_bool_list, result_list, len("".join(words)), sum(result_list))
            result.extend(indexes)
        return list(set(result))

    def get_total_number(self, my_dict):
        count = 0
        result_list = []
        for key, value in my_dict.iteritems():
            count += 1
            result_list.extend([count * value] * len(key))
        return result_list

        # result_list = []
        # for index ,word in enumerate(words):
        #     local_list = [index + 1] * len(word)
        #     result_list.extend(local_list)
        # return sum(result_list)

    def get_bool_list(self, index, s, words):
        """

        :param index:
        :param s:
        :param words:
        :return:
        """
        if not s or not words:
            return []
        if len(s) < len("".join(words)):
            return []
        local_s = copy.deepcopy(s)
        bool_list = [0] * len(s)
        local_s = local_s[index:]
        count = 0
        for word in words:
            count += 1
            while local_s.find(word) > -1:
                local_index = local_s.find(word)
                # bool_list[local_index + index : local_index + index + len(word)] = [1] * len(word)
                bool_list[local_index + index: local_index + index + len(word)] = [count] * len(word)
                local_s = local_s[:local_index] + " " * len(word) + local_s[local_index + len(word):]
                s = s[:local_index + index] + " " * len(word) + s[local_index + index + len(word):]
        return bool_list

    def get_continue_one_sum(self, bool_list, result_list, total_words_length, word_number):
        """
        :param bool_list:
        :param result_list
        :param total_words_length:
        :param word_number
        :return:
        """
        if not bool_list:
            return []
        # indexes = [index for index in range(len(bool_list) - total_words_length + 1) if sum(bool_list[index: index + total_words_length]) == total_words_length]
        indexes = []
        local_index = 0
        for index in range(len(bool_list) - total_words_length + 1):
            if index < local_index:
                continue
            slice_list = bool_list[index: index + total_words_length]
            if sum(slice_list) == word_number and self.is_not_contain_zero(slice_list):
                # 并判断slice_list中的数字是否严格满足条件
                if self.is_equal_to_list(slice_list, result_list):
                    indexes.append(index)
                    local_index = index + total_words_length

        # indexes = [index for index in range(len(bool_list) - total_words_length + 1) if
        #            sum(bool_list[index: index + total_words_length]) == word_number and self.is_not_contain_zero(bool_list[index: index + total_words_length])]

        return indexes

    def is_not_contain_zero(self, my_list):
        if not my_list:
            return True
        for item in my_list:
            if item == 0:
                return False
        return True

    def is_equal_to_list(self, slice_list, result_list):
        """
            本函数判断slice_list数组是否和result_list里面元素一致，可以无序
        :param slice_list:
        :param result_list:
        :return:
        """
        if slice_list and result_list:
            my_dict_1 = collections.Counter(slice_list)
            my_dict_2 = collections.Counter(result_list)
            if not set(my_dict_1.keys()).difference(set(my_dict_2.keys())):
                count = 0
                for key in my_dict_1.keys():
                    if my_dict_1.get(key) == my_dict_2.get(key):
                        count += 1
                if count == len(my_dict_2.keys()):
                    return True
                return False
            return False
        return False



if __name__ == "__main__":
    # my_insatnce = Solution()
    # text = "   -42"
    # print my_insatnce.myAtoi(text)

    my_instance = Solution1()
    # s = "sheateateseatea"
    # words = ["sea", "tea", "ate"]

    # s = "barfoofoobarthefoobarman"
    # words = ["bar", "foo", "the"]

    s = "wordgoodgoodgoodbestword"
    words = ["word", "good", "best", "good"]
    print my_instance.findSubstring2(s, words)

    # m()
    # n()



