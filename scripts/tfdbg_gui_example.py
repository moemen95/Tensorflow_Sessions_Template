import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python import debug as tf_debug


def main(config):
    # Import data
    mnist = input_data.read_data_sets(config.data_dir,
                                      one_hot=True,
                                      fake_data=config.fake_data)

    def feed_dict(train):
        if train or config.fake_data:
            xs, ys = mnist.train.next_batch(config.batch_size,
                                            fake_data=config.fake_data)
        else:
            xs, ys = mnist.test.images, mnist.test.labels

        return {x: xs, y_: ys}

    sess = tf.InteractiveSession()

    # Create the MNIST neural network graph.

    # Input placeholders.
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, config.image_size ** 2], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, config.num_classes], name="y-input")

    hidden = tf.layers.dense(x, config.hidden_size, activation=tf.nn.relu,
                             kernel_initializer=tf.initializers.truncated_normal(stddev=0.1,seed=config.seed),
                             name="hidden")
    logits = tf.layers.dense(hidden, config.num_classes,
                             kernel_initializer=tf.initializers.truncated_normal(stddev=0.1, seed=config.seed),
                             name="logits")
    y = tf.nn.softmax(logits)

    with tf.name_scope("cross_entropy"):
        # The following line is the culprit of the bad numerical values that appear
        # during training of this graph. Log of zero gives inf, which is first seen
        # in the intermediate tensor "cross_entropy/Log:0" during the 4th run()
        # call. A multiplication of the inf values with zeros leads to nans,
        # which is first in "cross_entropy/mul:0".
        #
        # You can use the built-in, numerically-stable implementation to fix this
        # issue:
        #   diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits)

        diff = -(y_ * tf.log(y))
        with tf.name_scope("total"):
            cross_entropy = tf.reduce_mean(diff)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(config.learning_rate).minimize(cross_entropy)

    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    if config.debug and config.tensorboard_debug_address:
        raise ValueError("The --debug and --tensorboard_debug_address config are mutually exclusive.")
    if config.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type=config.ui_type)
    elif config.tensorboard_debug_address:
        sess = tf_debug.TensorBoardDebugWrapperSession(
            sess, config.tensorboard_debug_address)

    # Add this point, sess is a debug wrapper around the actual Session if
    # config.debug is true. In that case, calling run() will launch the CLI.
    for i in range(config.max_steps):
        acc = sess.run(accuracy, feed_dict=feed_dict(False))
        print("Accuracy at step %d: %s" % (i, acc))

        sess.run(train_step, feed_dict=feed_dict(True))


class Config:
    image_size = 28
    hidden_size = 500
    num_classes = 10
    seed = 42
    max_steps = 10
    batch_size = 100
    learning_rate = 0.025
    data_dir = '../data/mnist_data'
    ui_type = 'curses'
    fake_data = False
    debug = False
    tensorboard_debug_address = 'localhost:6064'


if __name__ == "__main__":
    main(Config)
