"""
This file will contain a complete example how to use Estimator API to train Cifar100
"""

import sys

sys.path.extend(['..'])

import pickle

import numpy as np
import tensorflow as tf

from utils.metrics import top_k_error


class Config:
    # Model params
    reg = 1e-4
    dropout = 0.4

    # train params
    learning_rate = 1e-4

    # data params
    numpy_data = 'cifar-10-batches-py/data_batch_1_dir/numpy_data'
    data_len = 10000
    img_h = 32
    img_w = 32
    img_c = 3


def network_fn(features, training_flag, params):
    assert features['x'].shape.as_list() == [None, 32, 32, 3]

    with tf.variable_scope("my_network"):
        with tf.variable_scope("conv1"):
            conv1 = tf.layers.conv2d(features['x'], filters=4, kernel_size=[3, 3], padding="same", name="conv",
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=params.reg))
            conv1 = tf.layers.batch_normalization(conv1, training=training_flag, fused=True, name="bn")
            conv1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)
            conv1 = tf.nn.relu(conv1)

        with tf.variable_scope("conv2"):
            conv2 = tf.layers.conv2d(conv1, filters=8, kernel_size=[3, 3], padding="same", name="conv",
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=params.reg))
            conv2 = tf.layers.batch_normalization(conv2, training=training_flag, fused=True, name="bn")
            conv2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)
            conv2 = tf.nn.relu(conv2)

        flattened = tf.layers.flatten(conv2)

        with tf.variable_scope("dense1"):
            dense1 = tf.layers.dense(flattened, units=32, name="dense",
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=params.reg))
            dense1 = tf.layers.batch_normalization(dense1, training=training_flag, fused=True, name="bn")
            dense1 = tf.nn.relu(dense1)
            dense1 = tf.layers.dropout(dense1, rate=params.dropout, training=training_flag)

        logits = tf.layers.dense(dense1, units=10, name="logits")

    return logits


def model_fn(features, labels, mode, params):
    training_flag = mode == tf.estimator.ModeKeys.TRAIN

    logits = network_fn(features, training_flag, params)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1, name="classes"),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Add evaluation metrics
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"]),
        "top_1_error": tf.metrics.mean(top_k_error(labels, logits, 1)),
        "top_5_error": tf.metrics.mean(top_k_error(labels, logits, 5)),
    }

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        tf.summary.scalar("train/loss/main", loss)
        tf.summary.scalar("train/loss/regularization", tf.add_n(reg_losses))

        total_loss = tf.add_n([loss] + reg_losses)
        tf.summary.scalar("train/loss/total", total_loss)

        optimizer = tf.train.AdamOptimizer(params.learning_rate)

        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step)

        tensors_to_log = {"acc": eval_metric_ops['accuracy'][1]}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

        return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op,
                                          eval_metric_ops=eval_metric_ops, training_hooks=[logging_hook])

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    params = Config()

    # Create the Estimator
    cifar10_classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir="./cifar10_convnet_model",
        config=None,
        params=params
    )

    # load the data
    with open(params.numpy_data, "rb") as f:
        data_class = pickle.load(f)
    data_x = data_class['x']
    data_y = data_class['y']

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        {'x': data_x.astype(np.float32)},
        y=data_y.astype(np.int32),
        batch_size=16,
        num_epochs=5,
        shuffle=True,
        queue_capacity=1000,
        num_threads=5)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        data_x,
        y=data_y,
        batch_size=16,
        num_epochs=1,
        shuffle=False,
        queue_capacity=1000,
        num_threads=1)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        data_x,
        y=None,
        batch_size=16,
        num_epochs=1,
        shuffle=False,
        queue_capacity=1000,
        num_threads=1)

    # Train the Model
    train_results = cifar10_classifier.train(train_input_fn)

    # Evaluate the Model
    evaluation_results = cifar10_classifier.evaluate(eval_input_fn)

    # test the Model
    test_predictions = cifar10_classifier.predict(test_input_fn)


if __name__ == "__main__":
    main()
