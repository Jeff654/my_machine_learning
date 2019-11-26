# -*- coding: utf-8 -*-

class StringPattern(object):
    def __init__(self, chr, p):
        self.chr = chr
        self.p = p
        self.p_len = len(p)
        self.pi = [0 for i in range(self.p_len)]

    def set_pattern(self, p):
        self.p = p

        self.p_len = len(p)

    def set_chr(self, chr):
        self.chr = chr


    '''KMP'''
    def __kmp_partial_match_table__(self):
        k = 0
        q = 1
        # self.pi[0] = 0
        while q < self.p_len:
            while (k > 0) and (self.p[k] != self.p[q]):
                k = self.pi[k - 1]
            if self.p[k] == self.p[q]:
                k = k + 1
            self.pi[q] = k
            q = q + 1
        return 0

    def string_pattern_kmp(self):
        self.__kmp_partial_match_table__()
        print(self.pi)
        list_size = len(self.chr)
        pi_len = len(self.pi)
        k = 0
        for q in range(list_size):
            while (k > 0) and (self.p[k] != self.chr[q]):
                k = self.pi[k - 1]
            if self.p[k] == self.chr[q]:
                k = k + 1
            if k == pi_len:
                return q - pi_len + 1
                # q = q+1
        return 0


    '''BM'''
    def __calc_match__(self, num):
        k = num
        j = 0
        while k >= 0:
            if self.p[-k] == self.p[j]:
                k = k - 1
                j = j + 1
                if k <= 0:
                    self.pi[num - 1] = num
                    return 0
            else:
                if num == 1:
                    return 0
                self.pi[num - 1] = self.pi[num - 2]
                return 0

    def __init_good_table__(self):
        i = 1
        while i <= self.p_len:
            self.__calc_match__(i)
            i = i + 1
        print (self.pi)
        return 0

    def __check_bad_table__(self, tmp_chr):
        i = 1
        while self.p_len - i >= 0:
            if self.p[-i] == tmp_chr:
                return i
            else:
                i = i + 1
        return self.p_len + 1

    def __check_good_table__(self, num):
        if not num:
            return self.p_len
        else:
            return self.pi[num]

    def string_pattern_bm(self):
        self.__init_good_table__()
        tmp_len = self.p_len
        i = 1
        while tmp_len <= len(self.chr):
            if self.p[-i] == self.chr[tmp_len - i]:
                i = i + 1
                if i > self.p_len:
                    return tmp_len - self.p_len
            else:
                tmp_bad = self.__check_bad_table__(self.chr[tmp_len - i]) - i
                tmp_good = self.p_len - self.__check_good_table__(i - 1)
                tmp_len = tmp_len + max(tmp_bad, tmp_good)
                print(tmp_bad, tmp_good, tmp_len)
                i = 1
        return 0


    '''sunday'''
    def __check_bad_shift__(self, p):
        i = 0
        while i < self.p_len:
            if self.p[i] == p:
                return i
            else:
                i = i + 1
        return -1

    def string_pattern(self):
        # self.__init_good_table__()
        tmp_len = 0
        tmp_hop = self.p_len
        i = 0
        while tmp_hop <= len(self.chr):
            if self.p[i] == self.chr[tmp_len + i]:
                i = i + 1
                if i == self.p_len:
                    return tmp_len
            else:
                tmp_len = tmp_len + self.p_len - self.__check_bad_shift__(self.chr[tmp_hop])
                tmp_hop = tmp_len + self.p_len
                i = 0
        return 0








