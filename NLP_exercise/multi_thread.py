# -*- coding: utf-8 -*-


import time
import thread
import threading



def timer1(n, ID):
    count = 0
    while count < n:
        print "Thread: (%d) Time: %s\n" %(ID, time.ctime())
        count += 1
    thread.exit_thread()


class timer2(threading.Thread):
    def __init__(self, ID):
        threading.Thread.__init__(self)
        self.m_ID = ID
        self.m_stop = False

    def run(self):
        while not self.m_stop:
            time.sleep(2)
            print "Thread Object(%d), Time: %s\n" %(self.m_ID, time.ctime())

    def stop(self):
        self.m_stop = True

# @profile
def do_something():
    thread1 = timer2(1)
    thread2 = timer2(2)
    thread1.start()
    thread2.start()
    time.sleep(5)
    thread1.stop()
    thread2.stop()


from multiprocessing import Pool
from threading import Thread
from multiprocessing import Process

def loop():
    while True:
        pass


def f(x):
    return x + 1


def my_add(local_tuple):
    return local_tuple[0] + local_tuple[1], local_tuple[0] * local_tuple[1]

def local_add(x, y):
    return x + y

import unicodedata


if __name__ == "__main__":
    # do_something()
    # for index in range(3):
    #     t = Thread(target=loop)
    #     # t = Process(target=loop)
    #     print "the index is: ", index, time.time()
    #     t.start()
    # while True:
    #     pass

    import time
    # start_time = time.time()
    # my_list = range(100000000)
    # result = [f(x) for x in my_list]
    # end_time = time.time()
    # print "before: ", end_time - start_time


    from multiprocessing import pool
    p = pool.Pool(4)
    # lst = range(100000000)
    # p.map(f, lst)
    # print "after: ", time.time() - end_time

    my_parameter_list = [(index, index + 1) for index in range(100)]
    start_time = time.time()
    result = p.map(my_add, my_parameter_list)
    # # result = p.starmap(local_add, my_parameter_list)
    print(result)
    print time.time() - start_time

    # print "*****************************************************"
    # p = Pool(4)
    # result = p.starmap(local_add, my_parameter_list)        # 3.3 version new characteristic
    # print result
















