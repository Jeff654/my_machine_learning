# -*- coding: utf-8 -*-

import reflect_mechanism

def run():
    inp = input("请输入url: ").strip()
    print "the input is: ", inp
    if inp == "login":
        reflect_mechanism.login()
    elif inp == "logout":
        reflect_mechanism.logout()
    elif inp == "home":
        reflect_mechanism.home()
    else:
        pass

def run1():
    inp = input("请输入URL: ").strip()
    # func = getattr(reflect_mechanism, inp)
    # func()
    if hasattr(reflect_mechanism, inp):
        func = getattr(reflect_mechanism, inp)
        func()
    else:
        print "{0} module has no {1} attribute".format(reflect_mechanism, inp)





if __name__ == "__main__":
    # run()
    run1()






