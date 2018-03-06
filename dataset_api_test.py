import tensorflow as tf
import numpy as np
from tqdm import tqdm
import time

from data_loaders.session_data_loaders import BaselineCifar100Loader
from data_loaders.session_data_loaders import Cifar100IMGLoader
from models.baseline_cifar import Cifar
from utils.metrics import FPSMeter


class Config:
    img_h = 32
    img_w = 32
    img_c = 3
    num_classes = 100

    learning_rate = 1e-3
    summary_dir = './exp_1'

    x_train = 'data/cifar-100-python/x_train.npy'
    y_train = 'data/cifar-100-python/y_train.npy'
    x_test = 'data/cifar-100-python/x_test.npy'
    y_test = 'data/cifar-100-python/y_test.npy'

    x_train_filenames = 'data/cifar-100-python/x_train_filenames.pkl'
    x_test_filenames = 'data/cifar-100-python/x_test_filenames.pkl'

    batch_size = 8


def test():
    tf.reset_default_graph()

    sess = tf.Session()

    # data_loader = BaselineCifar100Loader(Config)
    data_loader = Cifar100IMGLoader(Config)

    model = Cifar(Config)

    x, y = data_loader.get_input()

    model.build(x, y)

    print('Model is built successfully')

    init = tf.global_variables_initializer()
    sess.run(init)

    print(tf.get_collection('inputs'))
    print(tf.get_collection('out'))

    is_training, x_pl, y_pl = tf.get_collection('inputs')

    out = tf.get_collection('out')
    out = out[0]

    train_step, loss, acc = tf.get_collection('train')

    meter = FPSMeter(Config.batch_size)

    # tt = tqdm(data_loader.generator_train(), total=data_loader.num_iterations_train, desc="Epoch-{}".format(1))
    data_loader.initialize(sess)
    for _ in tqdm(range(6250)):
        start = time.time()
        try:
            _, loss_ret, acc_ret = sess.run([train_step, loss, acc],
                                            feed_dict={is_training: True})
        except tf.errors.OutOfRangeError:
            print("END OF DATASET")
        meter.update(time.time() - start)

    meter.print_statistics()


if __name__ == '__main__':
    test()
