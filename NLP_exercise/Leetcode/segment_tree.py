# -*- coding: utf-8 -*-

# 线段树的基本实现

class SegmentTree(object):
    def __init__(self, data, func):
        """
            构造函数
        :param data: list type
        :param func: function, 用于实现对两个数的操作功能，如加减乘除等
        """
        self.data = data
        self.tree = [None] * 4 * len(self.data)
        self.function = func

        self._build_segment_tree(0, 0, len(self.data) - 1)


    def get_size(self):
        """
            获取有效元素的个数
        :return:
        """
        return len(self.data)


    def get_item(self, index):
        """
            获取到索引为index的相应元素
        :param index:
        :return:
        """
        if index < 0 or index > len(self.data):
            raise Exception("index is illegal")
        return self.data[index]


    def query(self, query_l, query_r):
        """
            查找区间[query_l, query_r]闭区间上的值
        :param query_l: 左端点索引
        :param query_r: 右端点索引
        :return:
        """
        if query_l < 0 or query_r < 0 or query_r >= self.get_size():
            raise Exception("the index is illegal")

        return self._meta_query(0, 0, self.get_size() - 1, query_l, query_r)


    def set_item(self, index, e):
        """
            将数组中index位置的值设置为e
        :param index:
        :param e:
        :return:
        """
        if index < 0 or index >= self.get_size():
            raise Exception("index is illegal")

        self.data[index] = e
        self._meta_set(0, 0, self.get_size() - 1, index, e)


    def print_segment_tree_path(self):
        """
            路径打印
        :return:
        """
        pass


    # private method
    def _left_child(self, index):
        """
            和最大堆一样，由于线段树是一棵完全二叉树，通过索引方式即可
        :param index:
        :return:
        """
        return 2 * index + 1


    def _right_child(self, index):
        """

        :param index:
        :return:
        """
        return 2 * index + 2


    def _build_segment_tree(self, tree_index, left, right):
        """
            以 tree_index 为树的根节点，构造self.data在[left, right]上的线段树
        :param tree_index: 线段树的根节点索引
        :param left: 数据的左端点索引
        :param right: 数据的右端点索引
        :return:
        """
        if left == right:
            self.tree[tree_index] = self.data[left]
            return

        left_child_index = self._left_child(tree_index)
        right_child_index = self._right_child(tree_index)

        middle_index = left + (right - left) // 2
        self._build_segment_tree(left_child_index, left, middle_index)
        self._build_segment_tree(right_child_index, middle_index + 1, right)
        self.tree[tree_index] = self.function(self.tree[left_child_index], self.tree[right_child_index])



    def _meta_query(self, tree_index, left, right, query_l, query_r):
        """
            在根节点索引为 tree_index 的线段树上查找索引范围为 [query_l, query_r] 上的值，
            其中 left， right 值代表该节点所表示的索引范围（左闭右闭）
        :param tree_index: 根节点所在索引
        :param left: 根节点所代表区间的左端点索引值
        :param right: 根节点所代表区间的右端点索引值
        :param query_l: 待查询区域左端点索引值
        :param query_r: 待查询区域右端点索引值
        :return:
        """
        if left == query_l and right == query_r:
            return self.tree[tree_index]

        middle_index = left + (right - left) // 2
        left_child_index = self._left_child(tree_index)
        right_child_index = self._right_child(tree_index)

        if query_l > middle_index:
            return self._meta_query(right_child_index, middle_index + 1, right, query_l, query_r)
        elif query_r < middle_index:
            return self._meta_query(left_child_index, left, middle_index, query_l, query_r)

        # 此时一部分在[left, middle_index], 一部分在[middle_index, right]区间内
        left_result = self._meta_query(left_child_index, left, middle_index, query_l, query_r)
        right_result = self._meta_query(right_child_index, middle_index + q, right, query_l, query_r)

        return self.function(left_result, right_result)


    def _meta_set(self, tree_index, left, right, index, e):
        """
             在以索引 tree_index 为根节点的线段树中, 将索引为 index 的
             位置的元素设为 e（此时treeIndex索引处所代表的区间范围为：[left, right]
        :param tree_index:
        :param left:
        :param right:
        :param index:
        :param e:
        :return:
        """
        if left == right:
            self.tree[tree_index] = e
            return

        middle_index = left + (right - left) // 2
        left_child_index = self._left_child(tree_index)
        right_child_index = self._right_child(tree_index)

        if index <= middle_index:
            self._meta_set(left_child_index, left, middle_index, index, e)
        else:
            self._meta_set(right_child_index, middle_index + 1, right, index, e)

        self.tree[tree_index] = self.function(self.tree[left_child_index], self.tree[right_child_index])
















