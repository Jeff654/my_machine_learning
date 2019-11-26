# -*- coding: utf-8 -*-


# 观察者模式


def printInfo(info):
    # print unicode(info, 'utf-8').encode('gbk')
    print info

# 抽象的通知者
class Informer():
    observers = []
    action = ''

    def Attach(self, observer):
        self.observers.append(observer)

    def Notify(self):
        for o in self.observers:
            o.Update()


# 秘书
class Secretary(Informer):
    observers = []
    action = "老板回来了！！！"


# 老板
class Boss(Informer):
    observers = []
    update = []  # 更新函数接口列表
    action = "我胡汉三回来了！！！"

    def AddEventCB(self, eventCB):
        self.update.append(eventCB)

    def Notify(self):
        for o in self.update:
            o()


# 抽象的观察者
class Observer():
    name = ''
    nformer = None;

    def __init__(self, name, secretary):
        self.name = name
        self.secretary = secretary

    def Update(self):
        pass


# 看股票的同事
class StockObserver(Observer):
    name = ''
    secretary = None;

    def __init__(self, name, secretary):
        Observer.__init__(self, name, secretary)

    def Update(self):
        printInfo("%s, %s, 不要看股票了，继续工作" % (self.secretary.action, self.name))

    def CloseStock(self):
        printInfo("%s, %s, 不要看股票了，快点工作" % (self.secretary.action, self.name))


# 看NBA的同事
class NBAObserver(Observer):
    name = ''
    secretary = None;

    def __init__(self, name, secretary):
        Observer.__init__(self, name, secretary)

    def Update(self):
        printInfo("%s, %s, 不要看NBA了，继续工作" % (self.secretary.action, self.name))


def clientUI():
    secretary = Secretary()
    stockObserver1 = StockObserver('张三', secretary)
    nbaObserver1 = NBAObserver('王五', secretary)

    secretary.Attach(stockObserver1)
    secretary.Attach(nbaObserver1)

    secretary.Notify()

    huHanShan = Boss()
    stockObserver2 = StockObserver('李四', huHanShan)
    huHanShan.AddEventCB(stockObserver2.CloseStock)
    huHanShan.Notify()
    return


if __name__ == '__main__':
    clientUI();








