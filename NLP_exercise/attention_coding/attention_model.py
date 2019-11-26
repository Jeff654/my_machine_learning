# -*- coding: utf-8 -*-

from keras.layers import *
import keras.backend as K


def to_mask(x, mask, mode="mul"):
    """
        通用的 mask 函数
    :param x:
    :param mask:
    :param mode:
    :return:
    """
    if mask is None:
        return x

    for _ in range(K.ndim(x) - K.ndim(mask)):
        mask = mask.expand_dims(mask, K.ndim(mask))

    if mode == "mul":
        return x * mask
    else:
        return x - (1 - mask) * 1e10




def extract_seq_patches(x, kernel_size, rate):
    """
        x.shape = [None, seq_len, seq_dim]
        滑动地将每个窗口里面的 x 提取出来，为局部 attention 做准备
    :param x:
    :param kernel_size:
    :param rate:
    :return:
    """
    seq_dim = K.int_shape(x)[-1]
    seq_len = K.shape(x)[1]
    k_size = kernel_size + (rate - 1) * (kernel_size - 1)

    p_right = (k_size - 1) // 2
    p_left = k_size - 1 - p_right

    x = K.temporal_padding(x, (p_right, p_left))
    xs = [x[:, i: i + seq_len] for i in range(0, k_size, rate)]
    x = K.concatenate(xs, 2)

    return K.reshape(x, (-1, seq_len, kernel_size, seq_dim))



class OurLayer(Layer):
    """
        定义新的Layer层，增加reuse方法，允许在定义Layer时调用现成的层
    """
    def reuse(self, layer, *args, **kwargs):
        if not layer.build:
            if len(args) > 0:
                inputs = args[0]
            else:
                inputs = kwargs["inputs"]

            if isinstance(inputs, list):
                input_shape = [K.int_shape(x) for x in inputs]
            else:
                input_shape = K.int_shape(inputs)

            layer.build(input_shape)

        outputs = layer.call(*args, **kwargs)
        for w in layer.trainable_weights:
            if w not in self._trainable_weights:
                self._trainable_weights.append(w)

        for w in layer.non_trainable_weights:
            if w not in self._non_trainable_weights:
                self._non_trainable_weights.append(w)

        return outputs



class Attention(OurLayer):
    """
        实现多注意力机制
    """
    def __init__(self, heads, size_per_head, key_size=None, mask_right=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right


    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        self.q_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.k_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.v_dense = Dense(self.out_dim, use_bias=False)


    def call(self, inputs):
        q, v, k = inputs[: 3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]

        # 线性变换
        qw = self.reuse(self.q_dense, q)
        kw = self.reuse(self.k_dense, k)
        vw = self.reuse(self.v_dense, v)


        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.heads, self.size_per_head))

        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))

        # attention
        a = K.batch_dot(qw, kw, [3, 3]) / self.key_size ** 0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = to_mask(a, v_mask, "add")
        a = K.permute_dimensions(a, (0, 3, 2, 1))

        if (self.mask_right is not False) or (self.mask_right is not None):
            if self.mask_right is True:
                ones = K.ones_like(a[: 1, : 1])
                mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
                a = a - mask
            else:
                # 此种case，表示 mask_right 是由外部传入的 0/1 矩阵， shape = [q_len, k_len]
                mask = (1 - K.constant(self.mask_right)) * 1e10
                mask = K.expand_dims(K.expand_dims(mask, 0), 0)
                self.mask = mask
                a = a - mask

        a = K.softmax(a)
        self.a = a

        # 输出
        o = K.batch_dot(a, vw, [3, 2])
        o = K.permute_dimensions(0, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(0)[1], self.out_dim))
        o = to_mask(o, q_mask, "mul")

        return o


    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)






