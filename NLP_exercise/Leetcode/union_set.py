# -*- coding: utf-8 -*-

"""
    # 并查集
    1) 可查找、合并操作
    2) 无法删除

    在此使用两个字典来实现并查集：一个字典保存当前的父节点信息、另一个字典保存父节点的大小信息(点数(内部节点 + 叶子节点数)或者树的高度)

"""

class UnionSet(object):
    def __init__(self, data_list):
        """
            初始化两个字典，一个保存结点的父节点，另一个保存父节点的大小
            初始化时，将节点的父节点设为自身(路径压缩)，size设置为1
        :param data_list:
        """
        self.father_dict = dict()
        self.father_size = dict()

        for node in data_list:
            self.father_dict[node] = node
            self.father_size[node] = 1


    def find_head(self, node):
        """
            使用递归的方式来查找父节点

        :param node:
        :return:
        """
        father = self.father_dict[node]
        if node != father:
            father = self.find_head(father)
        self.father_dict[node] = father

        return father


    def is_same_set(self, node_a, node_b):
        """
            判断两个结点是否在同一个集合内
        :param node_a:
        :param node_b:
        :return:
        """
        return self.find_head(node_a) == self.find_head(node_b)


    def union(self, node_a, node_b):
        """
            将节点 node_a 和节点 node_b 所代表的集合连接在一起
        :param node_a:
        :param node_b:
        :return:
        """
        if node_a is None or node_b is None:
            return

        father_a = self.find_head(node_a)
        father_b = self.find_head(node_b)

        if father_a != father_b:
            a_set_size = self.father_size[father_a]
            b_set_size = self.father_size[father_b]

            if a_set_size > b_set_size:
                # 为了减小迁移(合并)的代价，理当小集合并入大集合
                # 将node_b的祖先结点指向node_a的祖先节点，亦即：father_b是father_a的儿子
                self.father_dict[father_b] = father_a
                self.father_size[father_a] = a_set_size + b_set_size
            else:
                self.father_dict[father_a] = father_b
                self.father_size[father_b] = a_set_size + b_set_size
        else:
            return





if __name__ == "__main__":
    data = range(10)
    union_set = UnionSet(data)
    union_set.union(1, 2)
    union_set.union(3, 5)
    union_set.union(3, 1)
    print(union_set.father_dict)
    print(union_set.is_same_set(2, 5))








