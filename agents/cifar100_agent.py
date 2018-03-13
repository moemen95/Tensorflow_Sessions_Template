"""
Hello Everyone That's will be our cifar100 agent to use it in training and testing
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from models import *
from data_loaders import *
from utils.summarizer import DefinedSummarizer
from utils.metrics import AverageMeter, FPSMeter


class Cifar100Agent:

    def __init__(self, config):
        self.config = config

        # initialize the graph
        tf.reset_default_graph()

        # Session
        confg_sess = tf.ConfigProto(allow_soft_placement=True)
        confg_sess.gpu_options.allow_growth = True
        self.sess = tf.Session(config=confg_sess)

        # dataloader
        data_loader_class = globals()[config.data_loader]
        self.data_loader = data_loader_class(config)
        x, y = self.data_loader.get_input()

        # model
        model_class = globals()[config.model]
        self.model = model_class(config)
        self.model.build(x, y)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Saver
        self.saver = tf.train.Saver(max_to_keep=5, save_relative_paths=True)

        # load the model
        self.load()

        # Summarizer
        self.summarizer = DefinedSummarizer(self.sess, self.config.summary_dir,
                                            ['train/loss_per_epoch', 'train/acc_per_epoch'])

    def run(self):
        self.train()

    def train(self):
        """

        :return:
        """

        x, y, is_training = tf.get_collection('inputs')
        train_step, loss_node, acc_node = tf.get_collection('train')

        for epoch in range(self.model.global_epoch_tensor.eval(self.sess), self.config.max_epoch, 1):

            # initialize dataset
            self.data_loader.initialize(self.sess, is_train=True)

            # initialize tqdm
            tt = tqdm(range(self.data_loader.num_iterations_train), total=self.data_loader.num_iterations_train,
                      desc="epoch-{}-".format(epoch))

            loss_per_epoch = AverageMeter()
            acc_per_epoch = AverageMeter()

            # Iterate over batches
            for cur_it in tt:
                _, loss, acc, _ = self.sess.run([train_step, loss_node, acc_node, self.model.global_step_inc],
                                                feed_dict={is_training: True})

                loss_per_epoch.update(loss)
                acc_per_epoch.update(acc)

            self.sess.run(self.model.global_epoch_inc)

            # summarize
            summaries_dict = {'train/loss_per_epoch': loss_per_epoch.val,
                              'train/acc_per_epoch': acc_per_epoch.val}
            self.summarizer.summarize(self.model.global_step_tensor.eval(self.sess), summaries_dict)

            self.save()

            tt.close()

    def test(self):
        pass

    def save(self):
        print("saving a checkpoint..")
        self.saver.save(self.sess, self.config.checkpoint_dir, self.model.global_step_tensor)
        print("Checkpoint saved")

    def load(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            self.saver.restore(self.sess, latest_checkpoint)
            print("Loading Model checkpoint {} ..".format(latest_checkpoint))
        else:
            print("The Training is initializing itself")

    def finalize(self):
        self.save()
        self.summarizer.finalize()
