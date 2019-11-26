# -*- coding: utf-8 -*-

__author__ = "jeff"
__date__ = "2018.12.16"
__describe__ = """
    generate this with one of the Markov Chain Carlo methods called Metropolis
    Hastings algorithm; and assumed transition probabilities would follow 
    normal distribution x2 ~ N(x1, Covariance = [[0.2, 0], [0, 0.2]])
    
    马尔科夫蒙特卡洛采样：
        传统的MC方法：在样本空间内均匀的采样

        MCMC方法：原则：在样本稠密空间花费更多的时间采样，亦即：在稠密空间采取更多的样本。一旦进入样本稠密(高概率)区域，
                则尝试着在此稠密空间内去搜集更多样本；在生成下一个样本点时，采取基于当前样本点，给与与当前样本点邻近的
                样本更大的概率(更容易采样到邻近样本点)，给予远处的样本点更小的概率值，这样就能保证在稠密空间内采样更多
                的样本点；然而，也并非所有的样本点都是在稠密空间进行采样(这样会造成样本有偏)，在此给出一个随机的跳跃
                (从一个稠密样本子空间跳跃到另一个稠密样本子空间，当然，这里也是有一定几率的，给予其他的稠密样本子空间更大
                的概率跳跃值，相应的给予稀疏样本子空间更小的概率跳跃值)
        
        具体步骤：
            1、在样本空间中随机选取一个样本点 x1
            2、生成下一个样本点 x2 的方法：以样本点 x1 的均值为均值，选取一个固定的方差 S; 以参数对
               (x1, S)为正态分布；此处的方差 S 如若过大，则会导致正态分布密度函数比较扁平，峰值较小，
               则意味着当前采样的样本点 x2 距离图像均值处 x1 较远的概率增加，则 x2 落在非当前稠密空间
               的概率增加，不太利于上述 MCMC 分析的原理；如若方差 S 过小，则会导致正态分布函数图像十分
               "尖锐"，表示当前稠密空间内的样本点十分紧凑，采样点不易"逃逸"，此种情形下，所有的采样样本点
               则基本上会"仅仅"落在此稠密空间内，极端情形下，采样"仅仅"只选取此稠密空间内的样本，而对其他
               子空间内的样本"置之不理"，显然会导致采样数据严重倾斜，不智
            3、由于步骤2的分析，在此采用启发式的方法来判断下一个样本点是"接受"还是"拒绝"
                3.1、如若比值 P(x2)/P(x1) >= 1, 则接受该样本点 x2;
                3.2、如若比值 P(x2)/P(x1) < 1, 在此生成一个随机数r(0~1之间的均匀分布产生的随机数), 
                     如果此比值大于r，则接受该样本；否则，则拒绝该样本点
"""


import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()


num_samples = 100000
prob_density = 0

# 二元高斯变量
mean = np.array([0, 0])
cov = np.array([[1, 0.7], [0.7, 1]])

cov1 = np.matrix(cov)
mean1 = np.matrix(mean)

x_list, y_list = [], []
accepted_samples_count = 0
normalizer = np.sqrt(((2 * np.pi) ** 2) * np.linalg.det(cov))
x_initial, y_initial = 0, 0
x1, y1 = x_initial, y_initial

for index in range(num_samples):
    mean_trans = np.array([x1, y1])
    # mean_trans = np.mean([x1, y1])
    cov_trans = np.array([[0.2, 0], [0, 0.2]])
    x2, y2 = np.random.multivariate_normal(mean_trans, cov_trans).T
    X = np.array([x2, y2])
    X2 = np.matrix(X)
    X1 = np.matrix(mean_trans)

    mahalnobis_dist2 = (X2 - mean1) * np.linalg.inv(cov) * (X2 - mean1).T
    prob_density2 = (1 / float(normalizer)) * np.exp(-0.5 * mahalnobis_dist2)
    mahalnobis_dist1 = (X1 - mean1) * np.linalg.inv(cov) * (X1 - mean1).T
    prob_density1 = (1 / float(normalizer)) * np.exp(-0.5 * mahalnobis_dist1)

    # core algorithm code
    acceptance_ratio = prob_density2[0, 0] / float(prob_density1[0, 0])
    if (acceptance_ratio >= 1.0) or ((acceptance_ratio < 1.0) and (acceptance_ratio >= np.random.uniform(0, 1))):
        x_list.append(x2)
        y_list.append(y2)
        x1 = x2
        y1 = y2
        accepted_samples_count += 1


end_time = time.time()
print("time taken to sample " + str(accepted_samples_count) + " points ==> "
      + str(end_time - start_time) + " seconds")
print("acceptance ratio ==> ", accepted_samples_count / float(num_samples))

plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(x_list, y_list, color="black")

print("mean of sample points")
print(np.mean(x_list), np.mean(y_list))

print("covariance matrix of sample points")
print(np.cov(x_list, y_list))

plt.show()














