# -*- coding: utf-8 -*-



class TrieTree(object):
    def __init__(self):
        self.tree = {}

    def add(self, word):
        tree = self.tree

        for char in word:
            if char in tree:
                tree = tree[char]
            else:
                tree[char] = {}
                tree = tree[char]

        tree['exist'] = True

    def search(self, word):
        tree = self.tree

        for char in word:
            if char in tree:
                tree = tree[char]
            else:
                return False

        if "exist" in tree and tree["exist"] == True:
            return True
        else:
            return False




if __name__ == "__main__":
    # tree = TrieTree()
    # tree.add("abc")
    # tree.add("bcd")
    # print(tree.tree)
    # # Print {'a': {'b': {'c': {'exist': True}}}, 'b': {'c': {'d': {'exist': True}}}}
    # print(tree.search("ab"))
    # # Print False
    # print(tree.search("abc"))
    # # Print True
    # print(tree.search("abcd"))
    # Print False

    series_list = ["LA42", "LD11", "LD11", "LAY50", "LD11", "LD11", "LAD", "AD16", "ND16", "ND1", "ND1", "SK", "SKB4",
                   "LA22K\uff08XB2\uff09", "AD56", "PL1-D", "AD56", "AD16", "TGAD56", "LAY5", "LAY7", "AD16", "XB6E",
                   "LA42", "LA42", "ND1", "TL-703", "AD16", "XVM", "SK", "LA42\uff08S\uff09", "LA42\uff08S\uff09",
                   "AD22B", "AD16", "AD17", "AD56", "AD56", "LA42\uff08B\uff09", "AD56", "AD17-30ZH", "AD115", "AD115",
                   "AD115", "TGAD56", "AD16", "ND1", "ND16", "SK", "SK", "ND16", "AD17", "AD17", "AD17", "AD11", "LAY39",
                   "SK", "SKB1", "ND9", "LAY5", "AD11", "AD17", "NP6", "AD16", "AD11", "AD11", "AD11", "AD11", "AD16",
                   "AD56", "AD16", "LAY39", "XB7E", "ACX", "ACX", "ACX", "ACX", "ACX", "AD11", "AD16", "AD11", "EB6-AD",
                   "EB6-AD", "EB6-AD", "EB6-AD", "EB6-AD", "BA", "E", "AD17", "AD56", "AD56", "AD56", "BA", "E10", "BA",
                   "BA", "LANB4", "XB6E", "XLA2", "AD56", "AD56", "RDD16", "AD17", "AD17", "AD16", "AD11", "AD17", "AD17",
                   "LA42", "XVC", "CD16", "XVC", "AD17", "AD11", "AD11", "LA22K\uff08XB2\uff09", "HUL1", "HUL1", "TGAD51",
                   "XB2B", "", "", "LA42", "LD11", "LAY50", "LAY50", "LD11", "AD22B", "LA42", "LA42", "AD56",
                   "LA42\uff08B\uff09", "AD16", "AD16", "AD11", "AD11", "JVA21", "XVM", "AD17", "LAY5", "LA42", "LA42",
                   "LA42", "AD16", "E219", "TL-701", "AD17", "AD17", "AD17", "AD17", "AD17", "AD16", "AD16", "AD16", "ML1",
                   "AD26", "AD17", "AD16", "CL2", "AD16", "SK", "SK", "", "SK", "CD11", "LAY5", "Y090", "AD16", "ND16",
                   "ND16", "CL", "CL", "XVC", "AD17-40ZH", "LD11", "AD17", "AD17", "LAY5s", "LA39", "LD11", "AD11", "CL",
                   "LD11"]

    # tree = TrieTree()
    # for current_series in set(series_list):
    #     tree.add(current_series)
    # print tree.search("LAY50")
    # print tree.search("LA42\uff08B\uff09")
    # print tree.add(u"LA42（B）")
    # s = "LA42（B）"
    # print s
    # print tree.tree
    #
    # text = u"LA24（B）"
    # print text
    # tree.add(text)
    # tree.add("")
    # print tree.tree


    tree = TrieTree()
    tree.add("ic65n")
    # tree.add("ic65")
    print tree.tree

    print tree.search("ic65")









