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
    # train on some steps
    for i in tqdm(range(config.n_iterations)):
        run_metadata = tf.RunMetadata()
        # run the session
        sess.run(train_step,
                 options=options,
                 run_metadata=run_metadata,
                 feed_dict=feed_dict(True))
        # We collect profiling infos for each step.
        profiler.add_step(i, run_metadata)

    run_metadata = tf.RunMetadata()
    print("Accuracy on test_data {}".format(sess.run(accuracy,
                                                     options=options,
                                                     run_metadata=run_metadata,
                                                     feed_dict=feed_dict(False))))
    # I collect profiling infos for last step, too.
    profiler.add_step(config.n_iterations, run_metadata)

    # OPEN Where did i get the run_metadata .. the file ?!
    # Print the timings of each operation that executed.
    print("Print the timings of each operation that executed")
    print("************************************************")
    print(run_metadata.step_stats)



    # THE Option builder
    profile_option_builder = tf.profiler.ProfileOptionBuilder

    # Default profiler for the command line
    tf.profiler.write_op_log(sess.graph, './', run_meta=run_metadata)

    # Parameters and shape
    param_stats = tf.profiler.profile(
        tf.get_default_graph(),
        options=profile_option_builder.trainable_variables_parameter())

    # Float operations
    float_stats = tf.profiler.profile(
        tf.get_default_graph(),
        options=profile_option_builder.float_operation())

    # profile time and memory
    time_n_memory_opts = (profile_option_builder(profile_option_builder.time_and_memory()).
                          with_step(-1).  # with -1, should compute the average of all registered steps.
                          with_file_output('profiling-%s.txt' % config.file_output).
                          select(['micros', 'bytes', 'occurrence']).order_by('micros').
                          build())

    # Profiling Time and Memory about ops are saved in 'profiling-%s.txt' % config.file_output
    profiler.profile_operations(options=time_n_memory_opts)

    time_n_memory_stats = tf.profiler.profile(
        tf.get_default_graph(),
        run_meta=run_metadata,
        cmd='code',
        options=time_n_memory_opts)


    sys.stdout.write('**************************************************')
    sys.stdout.write('total_params: \n{}\n'.format(param_stats.total_parameters))
    sys.stdout.write('**************************************************')
    sys.stdout.write('float_params: \n{}\n'.format(float_stats.total_parameters))
    sys.stdout.write('**************************************************')
    sys.stdout.write('time_n_memory_params: \n{}\n'.format(time_n_memory_stats.total_parameters))
    sys.stdout.write('**************************************************')




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
