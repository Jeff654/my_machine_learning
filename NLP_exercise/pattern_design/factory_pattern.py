# -*- coding: utf-8 -*-


__time__ = "2018-03-21"
__author__ = "jeff"


# 抽象产品角色
class PenCore(object):
    color = ""
    def writeWord(self, s):
        pass

# 具体产品角色1
class RedPenCore(PenCore):
    def __init__(self):
        self.color = "红色"
    def writeWord(self, s):
        print "写出" + self.color + "的字：" + s

# 具体产品角色2
class BluePenCore(PenCore):
    def __init__(self):
        self.color = "蓝色"
    def writeWord(self, s):
        print "写出" + self.color + "的字：" + s

# 具体产品角色3
class BlackPenCore(PenCore):
    def __init__(self):
        self.color = "黑色"
    def writeWord(self, s):
        print "写出" + self.color + "的字：" + s


# 抽象工厂
class BallPen(object):
    def __init__(self):
        # color = self.getPenCore().color
        print "生产了一只装有" + self.getPenCore().color + "笔芯的圆珠笔"

    def getPenCore(self):
        return PenCore()


# 具体工厂1
class RedBallPen(BallPen):
    def getPenCore(self):
        return RedPenCore()

# 具体工厂2
class BlueBallPen(BallPen):
    def getPenCore(self):
        return BluePenCore()

# 具体工厂3
class BlackBallPen(BallPen):
    def getPenCore(self):
        return BlackPenCore()

if __name__ == "__main__":
    ballPen = BlueBallPen()
    penCore = ballPen.getPenCore()
    penCore.writeWord("hello, jeff")



















