# -*- coding: utf-8 -*-

__date__ = "2019.03.28"
__author__ = "jeff"
__describe__ = "题目：将 1~9 着九个数字分成三组，每组三个三位数，且着三个三位数构成 1 : 2 : 3；试求解出满足条件的所有的三位数的组合"


def solution():
    for i in range(123, 329):
        d = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        i2 = str(i * 2)
        i3 = str(i * 3)
        concate_string = str(i) + str(i2) + str(i3)
        if "0" in concate_string:
            continue
        elif len(set(concate_string)) != 9:
            continue
        else:
            for n in concate_string:
                d[int(n) - 1] = 1
            if 0 not in d:
                print(i, i2, i3)




if __name__ == "__main__":
    solution()


