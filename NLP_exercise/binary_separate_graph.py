# -*- coding: utf-8 -*-

global_matched_array = []

class DFS_binary_separate_graph:
    def __init__(self, source_vertex_set, destination_vertex_set, edge, source_flags, destination_flags, visited_list):
        """
            :param source_vertex_set: list类型，表示的是起始顶点的集合
            :param destination_vertex_set: list类型，表示的是终点顶点的集合
            :param edge: dict类型，表示的是从源顶点到终点顶点的边
            :param source_flags: dict类型，表示的是源顶点是否已匹配
            :param destination_flags: dict类型，表示的是目标顶点是否已匹配
            :param visited_list: dict类型，表示的是目标顶点是否已被访问
        """
        self.source_vertex_set = source_vertex_set
        self.destination_vertex_set = destination_vertex_set
        self.edge = edge
        self.source_flags = source_flags
        self.destination_flags = destination_flags
        self.visited_list = visited_list


    def best_match(self):
        visit_length = 0            # 表示的是匹配的对数
        for source_vertex in self.source_vertex_set:
            # 若source_vertex 没有被匹配到
            if self.source_flags[source_vertex] == -1:
                for destination_vertex in self.destination_vertex_set:
                    self.visited_list[destination_vertex] = 0
                visit_length += self.get_path(source_vertex)
        return visit_length


    def get_path(self, source_vertex):
        """
            本函数旨在获取以source_vertex作为源顶点的path，亦即，在self.destination_vertex_set中与
            source_vertex相连的边，获取其路径path
        :param source_vertex:
        :return:
        """
        for destination_vertex in self.destination_vertex_set:
            # 若源点和终点之间存在边且终点尚未被访问
            if self.edge[source_vertex][destination_vertex] and not self.visited_list[destination_vertex]:
                self.visited_list[destination_vertex] = 1
                # 若终点尚未被匹配到
                if self.destination_flags[destination_vertex] == -1:
                    self.source_flags[source_vertex] = destination_vertex
                    self.destination_flags[destination_vertex] = source_vertex
                    global_matched_array.append((source_vertex, destination_vertex))
                    return True
                else:
                    # 先回溯一波
                    global_matched_array.remove((self.destination_flags[destination_vertex], destination_vertex))
                    if self.get_path(self.destination_flags[destination_vertex]):
                        self.source_flags[source_vertex] = destination_vertex
                        self.destination_flags[destination_vertex] = source_vertex
                        global_matched_array.append((source_vertex, destination_vertex))
                        return True
        return False


if __name__ == "__main__":
    source_vertex_set = ["A", "B", "C", "D"]
    destination_vertex_set = ["E", "F", "G", "H"]
    edge = {
                "A": {"E": 1, "F": 0, "G": 1, "H": 0},
                "B": {"E": 0, "F": 1, "G": 0, "H": 1},
                "C": {"E": 1, "F": 0, "G": 0, "H": 1},
                "D": {"E": 0, "F": 0, "G": 1, "H": 0}
            }
    source_flags = {"A": -1, "B": -1, "C": -1, "D": -1}
    destination_flags = {"E": -1, "F": -1, "G": -1, "H": -1}
    visited_list = {"E": 0, "F": 0, "G": 0, "H": 0}

    print DFS_binary_separate_graph(source_vertex_set, destination_vertex_set, edge, source_flags, destination_flags, visited_list).best_match()





























