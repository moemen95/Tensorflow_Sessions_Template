"""
This file is to profile everything in that example using Python API for the profiler.
I love you ALL.
I wanna ask you something ?!
Learning Tensorflow does it worth ?!
"""
import sys
from tqdm import tqdm
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def main(config):
    # clear the graph
    tf.reset_default_graph()

    # Import data
    mnist = input_data.read_data_sets(config.data_dir, one_hot=False)

    def feed_dict(train):
        if train:
            xs, ys = mnist.train.next_batch(config.batch_size)
        else:
            xs, ys = mnist.test.next_batch(config.batch_size)

        return {x: xs, y: ys}

    sess = tf.InteractiveSession()

    # Create the MNIST neural network graph.

    # Input placeholders.
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [config.batch_size, config.image_size ** 2], name="x-input")
        y = tf.placeholder(tf.int64, [config.batch_size], name="y-input")

    # The Network
    hidden = tf.layers.dense(x, config.hidden_size, activation=tf.nn.relu,
                             kernel_initializer=tf.initializers.truncated_normal(),
                             name="hidden")
    logits = tf.layers.dense(hidden, config.num_classes,
                             kernel_initializer=tf.initializers.truncated_normal(),
                             name="logits")
    # probabilities
    probs = tf.nn.softmax(logits)

    # out
    out = tf.argmax(logits, axis=1)

    with tf.name_scope("loss_cross_entropy"):
        print(y.shape)
        print(logits.shape)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

    with tf.name_scope("train_Step"):
        train_step = tf.train.AdamOptimizer(config.learning_rate).minimize(loss)

    with tf.name_scope("accuracy"):
        # Talk with them about argmax for logits not softmax
        accuracy = tf.reduce_mean(tf.cast(tf.equal(out, y), tf.float32))

    # initialize the variables of the network YA LO2Y <3 ba7ebak.
    sess.run(tf.global_variables_initializer())

    # Say WELCOME TO OUR PROFILER!!! YAY
    profiler = tf.profiler.Profiler(sess.graph)
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

    with tf.contrib.tfprof.ProfileContext('./profile_dir') as pctx:
        # train on some steps
        for i in tqdm(range(config.n_iterations)):
            # Enable tracing for next session.run.
            pctx.trace_next_step()
            # Dump the profile to '/tmp/train_dir' after the step.
            pctx.dump_next_step()
            # run the session
            sess.run(train_step,
                     options=options,
                     feed_dict=feed_dict(True))

        # Enable tracing for next session.run.
        pctx.trace_next_step()
        # Dump the profile to '/tmp/train_dir' after the step.
        pctx.dump_next_step()
        print("Accuracy on test_data {}".format(sess.run(accuracy,
                                                         options=options,
                                                         feed_dict=feed_dict(False))))




class Config:
    # network design
    image_size = 28
    hidden_size = 500
    num_classes = 10

    # training design
    n_iterations = 500
    batch_size = 32
    learning_rate = 1e-3
    data_dir = '../data/mnist_data'

    # profiler config
    file_output = 'v1'


if __name__ == "__main__":
    main(Config)
