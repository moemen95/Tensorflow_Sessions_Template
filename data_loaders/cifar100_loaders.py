"""
This class will contain different loaders for cifar 100 dataset
# Techniques
# FIT in ram
# - load numpys in the graph
# - generator python

# Doesn't fit in ram
# - load files but in tfrecords format
# - load files from disk using dataset api
"""
import pickle

from tqdm import tqdm

import tensorflow as tf
import numpy as np


class BaselineCifar100Loader:
    """
    Manual Loading
    Using placeholders and python generators
    """

    def __init__(self, config):
        self.config = config

        self.x_train = np.load(config.x_train)
        self.y_train = np.load(config.y_train)
        self.x_test = np.load(config.x_test)
        self.y_test = np.load(config.y_test)

        print("x_train shape: {} dtype: {}".format(self.x_train.shape, self.x_train.dtype))
        print("y_train shape: {} dtype: {}".format(self.y_train.shape, self.y_train.dtype))
        print("x_test shape: {} dtype: {}".format(self.x_test.shape, self.x_test.dtype))
        print("y_test shape: {} dtype: {}".format(self.y_test.shape, self.y_test.dtype))

        self.len_x_train = self.x_train.shape[0]
        self.len_x_test = self.x_test.shape[0]

        self.num_iterations_train = self.len_x_train // self.config.batch_size
        self.num_iterations_test = self.len_x_test // self.config.batch_size

    def get_input(self):
        x = tf.placeholder(tf.float32, [None, self.config.img_h, self.config.img_w, 3])
        y = tf.placeholder(tf.int64, [None, ])

        return x, y

    def generator_train(self):
        start = 0
        idx = np.random.choice(self.len_x_train, self.len_x_train, replace=False)
        while True:
            mask = idx[start:start + self.config.batch_size]
            x_batch = self.x_train[mask]
            y_batch = self.y_train[mask]

            start += self.config.batch_size

            yield x_batch, y_batch

            if start >= self.len_x_train:
                return

    def generator_test(self):
        start = 0
        idx = np.random.choice(self.len_x_test, self.len_x_test, replace=False)
        while True:
            mask = idx[start:start + self.config.batch_size]
            x_batch = self.x_test[mask]
            y_batch = self.y_test[mask]

            start += self.config.batch_size

            yield x_batch, y_batch

            if start >= self.len_x_test:
                return


class Cifar100IMGLoader:
    """
    DataSetAPI - Load Imgs from the disk
    """

    def __init__(self, config):
        self.config = config

        self.train_imgs_files = []
        self.test_imgs_files = []

        with open(config.x_train_filenames, "rb") as f:
            self.train_imgs_files = pickle.load(f)

        with open(config.x_test_filenames, "rb") as f:
            self.test_imgs_files = pickle.load(f)

        self.train_labels = np.load(config.y_train)
        self.test_labels = np.load(config.y_test)

        self.imgs = tf.convert_to_tensor(self.train_imgs_files, dtype=tf.string)

        self.dataset = tf.data.Dataset.from_tensor_slices((self.imgs, self.train_labels))
        self.dataset = self.dataset.map(Cifar100IMGLoader.parse_train, num_parallel_calls=self.config.batch_size)
        self.dataset = self.dataset.shuffle(1000, reshuffle_each_iteration=False)
        self.dataset = self.dataset.batch(self.config.batch_size)
        # self.dataset = self.dataset.repeat(1)

        # ['','',''] , [0,1,2]
        #      -          -
        #       preprocess -> map
        #       shuffle
        #       batch_size

        #### RAM -> Buffer -> Batch

        ### .,.,.,.,.,.,.,.,.,.,
        ### i

        self.iterator = tf.data.Iterator.from_structure((tf.float32, tf.int64), ([None, 32, 32, 3], [None, ]))
        self.training_init_op = self.iterator.make_initializer(self.dataset)

    @staticmethod
    def parse_train(img_path, label):
        # load img
        img = tf.read_file('data/cifar-100-python/' + img_path)
        img = tf.image.decode_png(img, channels=3)

        return tf.cast(img, tf.float32), tf.cast(label, tf.int64)

    def initialize(self, sess):
        sess.run(self.training_init_op)

    def get_input(self):
        return self.iterator.get_next()


class Cifar100TFRecord:
    """
        DataSetAPI - Load TFRecords from the disk
    """

    def __init__(self, config):
        self.config = config

        # initialize the dataset
        self.dataset = tf.data.TFRecordDataset(self.config.tfrecord_data)
        self.dataset = self.dataset.map(Cifar100TFRecord.parser, num_parallel_calls=self.config.batch_size)
        self.dataset = self.dataset.shuffle(1000)
        self.dataset = self.dataset.batch(self.config.batch_size)

        self.iterator = tf.data.Iterator.from_structure(self.dataset.output_types,
                                                        self.dataset.output_shapes)
        self.init_op = self.iterator.make_initializer(self.dataset)

    @staticmethod
    def parser(record):
        keys_to_features = {
            'label': tf.FixedLenFeature((), tf.int64),
            'image_raw': tf.FixedLenFeature((), tf.string)
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        image = tf.decode_raw(parsed['image_raw'], tf.uint8)
        image = tf.reshape(image, [32,32,3])
        label = parsed['label']
        image = tf.cast(image, tf.float32)

        return image, label

    def initialize(self, sess):
        sess.run(self.init_op)

    def get_input(self):
        return self.iterator.get_next()


class Cifar100Numpy:
    """
        DataSetAPI - Load Numpys from the disk
    """

    def __init__(self, config):
        self.config = config

        # TODO

    def initialize(self, sess, is_train):
        # TODO
        pass

    def get_input(self):
        return self.iterator.get_next()