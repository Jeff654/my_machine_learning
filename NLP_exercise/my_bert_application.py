# -*- coding: utf-8 -*-

import os
import datetime
import pickle
import warnings

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import bert
from bert import run_classifier, optimization, tokenization

flags = tf.flags
FLAGS = flags.FLAGS


warnings.filterwarnings("ignore")

path = "\\".join(os.getcwd().split("\\")[:-1]) + "\\resource_code_reading\\my_bert_data"


def pretty_print(result):
    df = pd.DataFrame([result]).T
    df.columns = ["values"]

    return df


def create_tokenizer_from_hub_module(bert_model_hub):
    """
        get the vocab file and casting info from the Hub module
    :param bert_model_hub:
    :return:
    """
    with tf.Graph().as_default():
        bert_module = hub.Module(bert_model_hub)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)



def make_features(dataset, label_list, max_seq_length, tokenizer, data_column, label_column):
    input_example = dataset.apply(lambda x: bert.run_classifier.InputExample(
        guid=None, text_a=x[data_column], text_b=None, label=x[label_column], axis=1
    ))

    features = bert.run_classifier.convert_examples_to_features(input_example,
                                                                label_list,
                                                                max_seq_length,
                                                                tokenizer)

    return features



def create_model(bert_model_hub, is_predict, input_ids, input_mask, segment_ids,
                 labels, num_labels):
    """
        create a classification model
    :param bert_model_hub:
    :param is_predict:
    :param input_ids:
    :param input_mask:
    :param segment_ids:
    :param labels:
    :param num_labels:
    :return:
    """
    bert_module = hub.Module(bert_model_hub, trainable=True)
    bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
    bert_outputs = bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)

    """
        use "pooled_output" for classification tasks on an entire sentence, 
        use "sequence_outputs" for token-level output
    """
    output_layer = bert_outputs["pooled_output"]
    hidden_size = output_layer.shape[-1].value

    # create own layer to tune for politeness data
    output_weights = tf.get_variable("output_weights", [num_labels, hidden_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))


        # if we are predicting, we want the predicted_labels and its probabilities
        if is_predict:
            return (predicted_labels, log_probs)


        # if train or eval, compute loss between predicted and actual labels
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, predicted_labels, log_probs)



def model_fn_builder(bert_model_hub, num_labels, learning_rate,
                     num_train_steps, num_warmup_steps):
    """
        it aims to build our model function, using the passed parameters
        for num_labels, learning_rate...
    :param bert_model_hub:
    :param num_labels:
    :param learning_rate:
    :param num_train_steps:
    :param num_warmup_steps:
    :return:
    """
    def model_fn(features, labels, mode, params):
        """
            return closure for TPUEstimator
        :param features:
        :param labels:
        :param mode:
        :param params:
        :return:
        """
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_predict = (mode == tf.estimator.ModeKeys.PREDICT)

        # train & eval
        if not is_predict:
            (loss, predicted_labels, log_probs) = create_model(
                bert_model_hub, is_predict, input_ids, input_mask, segment_ids, label_ids, num_labels
            )

            train_op = bert.optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps
            )

            # calculate evaluate metrics
            def metrics_fn(label_ids, predicted_labels):
                accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                f1_score = tf.contrib.metrics.f1_scaoe(label_ids, predicted_labels)
                auc = tf.metrics.auc(label_ids, predicted_labels)
                recall = tf.metrics.recall(label_ids, predicted_labels)
                precision = tf.metrics.precision(label_ids, predicted_labels)

                true_pos = tf.metrics.true_positives(label_ids, predicted_labels)
                true_neg = tf.metrics.true_negatives(label_ids, predicted_labels)
                false_pos = tf.metrics.false_positives(label_ids, predicted_labels)
                false_neg = tf.metrics.false_negatives(label_ids, predicted_labels)

                return {
                    "eval_accuracy": accuracy,
                    "f1_score": f1_score,
                    "auc": auc,
                    "recall": recall,
                    "precision": precision,
                    "true_pos": true_pos,
                    "true_neg": true_neg,
                    "false_pos": false_pos,
                    "false_neg": false_neg
                }


            eval_metrics = metrics_fn(label_ids, predicted_labels)

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
            else:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics)

        else:
            (predicted_labels, log_probs) = create_model(
                bert_model_hub, is_predict, input_ids, input_mask, segment_ids, label_ids, num_labels
            )
            predictions = {
                "probabilities": log_probs,
                "labels": predicted_labels
            }

            return tf.estimator.EstimatorSpec(mode, predictions=predictions)


    return model_fn


def estimator_builder(bert_model_hub, output_dir, save_summary_steps,
                      save_checkpoints_steps, label_list, learning_rate,
                      num_train_steps, num_warmup_steps, batch_size):
    run_config = tf.estimator.RunConfig(model_dir=output_dir,
                                        save_summary_steps=save_summary_steps,
                                        save_checkpoints_steps=save_checkpoints_steps)

    model_fn = model_fn_builder(bert_model_hub=bert_model_hub,
                                num_labels=len(label_list),
                                learning_rate=learning_rate,
                                num_train_steps=num_train_steps,
                                num_warmup_steps=num_warmup_steps)
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config,
                                       params={"batch_size": batch_size})

    return estimator, model_fn, run_config



def run_on_dfs(train, test, data_column, label_column, max_seq_length=128,
               batch_size=32, learning_rate=2e-5, num_train_epochs=3,
               warmup_proportion=0.1, save_summary_steps=100, save_checkpoint_steps=10000,
               bert_model_hub="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"):

    label_list = train[label_column].unique().tolist()
    tokenizer = create_tokenizer_from_hub_module(bert_model_hub)
    train_features = make_features(train, label_list, max_seq_length, tokenizer,
                                   data_column, label_column)
    test_features = make_features(test, label_list, max_seq_length, tokenizer,
                                  data_column, label_column)

    num_train_steps = int(len(train_features) / batch_size * num_train_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)

    estimator, model_fn, run_config = estimator_builder(
        bert_model_hub, output_dir, save_summary_steps, save_checkpoint_steps,
        label_list, learning_rate, num_train_steps, num_warmup_steps, batch_size
    )

    train_input_fn = bert.run_classifier.input_fn_builder(
        features=train_features, seq_length=max_seq_length, is_training=True,
        drop_remainder=False
    )
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)


    test_input_fn = bert.run_classifier.input_fn_builder(
        features=test_features, seq_length=max_seq_length, is_training=False,
        drop_remainder=False
    )

    result_dict = estimator.evaluate(input_fn=test_input_fn, steps=None)

    return result_dict, estimator







if __name__ == "__main__":
    OUT_DIR = path + "\\output"
    with open(path + "\\dianping_train_test.pickle", "rb") as f:
        train, test = pickle.load(f)

    train = train.sample(len(train))
    my_param = {
        "data_column": "comment",
        "label_column": "sentiment",
        "learning_rate": 2e-5,
        "num_train_epochs": 3,
        "bert_model_hub": "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    }

    result, estimator = run_on_dfs(train, test, **my_param)

    print(pretty_print(result))











