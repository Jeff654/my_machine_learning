# -*-coding: utf-8 -*-

from keras import backend as K


class Lookahead(object):
    def __init__(self, k=5, alpha=0.5):
        self.k = k
        self.alpha = alpha
        self.count = 0


    def insert_wrapper(self, model):
        """
            本函数主要是针对model进行优化
            调用 Lookahead 算法针对指定的 model 进行优化
            本函数只要是仿照 _make_train_function() 函数进行改写
        :param model:
        :return:
        """
        if not hasattr(model, "train_function"):
            raise RuntimeError("you must compile your model before using")


        model._check_trainable_weights_consistency()

        if model.train_function is None:
            inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)

            if model._uses_dynamic_learning_phase():
                inputs += [K.learning_phase()]

            fast_params = model._collected_trainable_weights

            with K.name_scope("training"):
                with K.name_scope(model.optimizer.__class__.__name__):
                    training_updates = model.optimizer.get_updates(
                        params=fast_params,
                        loss=model.total_loss
                    )
                    slow_params = [K.variable(p) for p in fast_params]

                fast_updates = (model.updates + training_updates + model.metrics_updates)

                slow_updates, copy_updates = [], []
                for p, q in zip(fast_params, slow_params):
                    slow_updates.append(K.update(q, q + self.alpha * (p - q)))
                    copy_updates.append(K.update(p, q))


                # get the loss and metrics, updates weights at each call
                fast_train_function = K.function(inputs,
                                                 [model.total_loss] + model.metrics_tensors,
                                                 updates=fast_updates,
                                                 name="fast_train_function",
                                                 **model._function_kwargs)


                def F(inputs):
                    self.count += 1
                    R = fast_train_function(inputs)
                    if self.count % self.k == 0:
                        K.batch_get_value(slow_updates)
                        K.batch_get_value(copy_updates)
                    return R

                model.train_function = F





















