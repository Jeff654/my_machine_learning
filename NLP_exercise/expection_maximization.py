# -*- coding: utf-8 -*-


import numpy as np
import math
import copy
import matplotlib.pyplot as plt

is_debug = True


def init_data(sigma, mean_1, mean_2, k, n):
    """

    :param sigma: float type, the mean square standard error
    :param mean_1: float type, the mean of first Gaussian Distribution
    :param mean_2: float type, the mean of second Gaussian Distribution
    :param k: int type, represent k Mixture Gaussian Distribution
    :param n: int type, the number of data
    :return:
    """
    global X
    global Mu
    global Expectation

    X = np.zeros((1, n))
    Mu = np.random.random(k)
    Expectation = np.zeros((n, k))

    for index in xrange(0, n):
        if np.random.random(1) > 0.5:
            X[0, index] = np.random.normal(mean_1, sigma)
        else:
            X[0, index] = np.random.normal(mean_2, sigma)

    if is_debug:
        print "---------------------"
        print "the initialized data..."
        print X


# compute Expectation step
def expectation_step(sigma, k, n):
    global Expectation
    global Mu
    global X
    epsilon = 1e-6

    for index_i in xrange(0, n):
        demon = 0
        number = [0.0] * k
        for index_j in xrange(0, k):
            number[index_j] = math.exp((-1 / (2 * (float(sigma**2)))) * (float(X[0, index_i] - Mu[index_j]))**2)
            demon += number[index_j]
        for index_j in xrange(0, k):
            Expectation[index_i, index_j] = number[index_j] / (demon + epsilon)
    if is_debug:
        print "---------Expectation step---------"
        print "the expectation step variable update is: ", Expectation


# compute Maximization step
def maximization_step(k, n):
    global Expectation
    global X
    global Mu
    epsilon = 1e-6

    for index_j in xrange(0, k):
        number = 0
        demon = 0
        for index_i in xrange(0, n):
            number += Expectation[index_i, index_j] * X[0, index_i]
            demon += Expectation[index_i, index_j]
        Mu[index_j] = number / (demon + epsilon)


# excute
def excute(sigma, mean_1, mean_2, k, n, iter_number, epsilon):
    init_data(sigma, mean_1, mean_2, k, n)
    print "the init vector <v1, v2> is: ", Mu
    for index in range(iter_number):
        old_Mu = copy.deepcopy(Mu)
        expectation_step(sigma, k, n)
        maximization_step(k, n)
        print "the {} step update parameter is: {}".format(index, Mu)
        if sum(abs(Mu - old_Mu)) < epsilon:
            break


if __name__ == "__main__":
    sigma = 6.0
    mean_1 = 40.0
    mean_2 = 20.0
    k = 2
    n = 50
    iter_number = 1000
    epsilon = 1e-10
    excute(sigma, mean_1, mean_2, k, n, iter_number, epsilon)

    plt.hist(X[0, :], 50)
    plt.show()





























