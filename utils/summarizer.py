import os

import tensorflow as tf


class DefinedSummarizer:
    def __init__(self, sess, summary_dir, scalar_tags=None, images_tags=None):
        """
        :param sess: The Graph tensorflow session used in your graph.
        :param summary_dir: the directory which will save the summaries of the graph
        :param scalar_tags: The tags of summaries you will use in your training loop
        :param images_tags: The tags of image summaries you will use in your training loop
        """
        self.sess = sess

        self.scalar_tags = scalar_tags
        self.images_tags = images_tags

        self.summary_tags = []
        self.summary_placeholders = {}
        self.summary_ops = {}

        self.init_summary_ops()

        self.summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

    def init_summary_ops(self):
        with tf.variable_scope('summary_ops'):
            if self.scalar_tags is not None:
                for tag in self.scalar_tags:
                    pass
                    # TODO
            if self.images_tags is not None:
                for tag, shape in self.images_tags:
                    pass
                    # TODO

    def summarize(self, step, summaries_dict=None, summaries_merged=None):
        """
        Add the summaries to tensorboard
        :param step: the number of iteration in your training
        :param summaries_dict: the dictionary which contains your summaries .
        :param summaries_merged: Merged summaries which they come from your graph
        :return:
        """
        if summaries_dict is not None:
            # TODO
            pass
        if summaries_merged is not None:
            # TODO
            pass

    def finalize(self):
        pass
    # TODO
