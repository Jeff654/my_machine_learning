# -*- coding: utf-8 -*-

from keras.optimizers import Optimizer
from keras.legacy import interfaces
import keras.backend as K


__date__ = "2019.08.19"
__describe__ = "本脚本旨在实现paper: on the variance of the adaptive learning rate and beyond"

"""
	# References
        - [RAdam - A Method for Stochastic Optimization]
          (https://arxiv.org/abs/1908.03265)
        - [On The Variance Of The Adaptive Learning Rate And Beyond]
          (https://arxiv.org/abs/1908.03265)
"""

class RAdam(Optimizer):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, **kwargs):
        super(RAdam, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype="int64", name="iterations")
            self.lr = K.variable(lr, name="lr")
            self.beta_1 = K.variable(beta_1, name="beta_1")
            self.beta_2 = K.variable(beta_2, name="beta_2")
            self.decay = K.variable(decay, name="decay")

        if epsilon is None:
            epsilon = K.epsilon()

        self.epsilon = epsilon
        self.initial_decay = decay


    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        # 衰减速率，学习率随着迭代次数的增加而下降
        if self.initial_decay > 0:
            lr = lr * (1.0 / (1.0 + self.decay * K.cast(self.iterations, K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1

        # 第 t 轮迭代的参数
        beta_1_t = K.pow(self.beta_1, t)
        beta_2_t = K.pow(self.beta_2, t)

        # Adam采用的是指数加权平均(exponential moving average)，在此利用简单移动平均(simple moving average)
        # 当 t 趋近于无穷大时，其 SMA 的值趋近于 2 / (1 - beta_2) - 1
        rho = 2 / (1 - self.beta_2) - 1

        # 表示的是 rho 在第 t 轮迭代中的值
        rho_t = rho - 2 * t * beta_2_t / (1 - beta_2_t)

        # 计算方差修正项
        r_t = K.sqrt(K.relu(rho_t - 4) * K.relu(rho_t - 2) * rho / (K.relu(rho - 4) * K.relu(rho - 2) * rho_t))

        # 论文中算法的推导过程需要保证 rho_4 > 0 才能进行
        flag = K.cast(rho_t > 4, K.floatx())


        # 初始化指数移动平均中的一次动量项和二次动量项
        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs


        for p, g, m, v in zip(params, grads, ms, vs):
            # 第 t 轮迭代中，指数移动的一次动量项
            m_t = (self.beta_1 * m) + (1.0 - self.beta_1) * g

            # 第 t 轮迭代中，指数移动的二次动量项
            v_t = (self.beta_2 * v) + (1.0 - self.beta_2) * K.square(g)

            # compute the bias-corrected moving average 1st moment
            m_hat_t = m_t / (1 - beta_1_t)

            # compute the bias-corrected moving average 2nd moment
            v_hat_t = K.sqrt(v_t / (1 - beta_2_t))

            # compute parameters with adaptive momentum
            p_t = p - lr * m_hat_t * (flag * r_t / (v_hat_t + self.epsilon) + (1 - flag))

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t


            # apply constraints
            if getattr(p, "constraint", None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))

        return self.updates


    def get_config(self):
        config = {"lr": float(K.get_value(self.lr)),
                  "beta_1": float(K.get_value(self.beta_1)),
                  "beta_2": float(K.get_value(self.beta_2)),
                  "decay": float(K.get_value(self.decay)),
                  "epsilon": self.epsilon}

        base_config = super(RAdam, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))



































































