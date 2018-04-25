import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from anomalygan import AnomalyGAN
from anomaly_utils import *
from time import time
import tensorflow as tf
import datetime

import sys

sys.path.append('../Thesis_Utilities/')
from utilities import load_datasets

x_train, y_train, x_val, y_val, x_test, y_test = load_datasets(test_size=10000, val_size=5000, omniglot_bool=True,
                                                               name_data_set='data_omni_seed1337.h5', force=False,
                                                               create_file=True, r_seed=1337)

# Load data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True, reshape=False, validation_size=5000)

train = False

if train:
    # Create Gan
    tf.reset_default_graph()
    sess = tf.Session()
    net = AnomalyGAN(sess)

    # Preprocess
    train_set = tf.image.resize_images(mnist.train.images[0:10], [64, 64]).eval(session=sess)
    train_set = (train_set - 0.5) / 0.5  # normalization; range: -1 ~ 1

    # Train Gan
    im, _, _ = net.train_model(train_set, epochs=20, batch_size=5, learning_rate=2e-4)

    # Display generated images
    rows, cols = 1, 10
    fig, axes = plt.subplots(figsize=(4, 4), nrows=rows, ncols=cols, sharex=True, sharey=True, squeeze=False)
    k = 0
    for ax_row in axes:
        for ax in ax_row:
            img = im[k]
            img = img[0][0][:, :, :]
            k = k + 1
            ax.imshow(np.squeeze(img), cmap='Greys_r')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
    plt.show()

# Anomaly detection

# Shuffle
"""
idx = np.random.permutation(len(mnist.validation.images))
x_test = mnist.validation.images[idx]
y_test = mnist.validation.labels[idx]
"""
idx = np.random.permutation(len(x_test))
x_test = x_test[idx]
y_test = y_test[idx]
loss_omni = []
loss_mnist = []
loss_all = list()
plot = False
n_querys = 200
tf.reset_default_graph()

with tf.Session() as sess:
    query_set = tf.image.resize_images(x_test, [64, 64]).eval(session=sess)
    query_set = (query_set - 0.5) / 0.5
    net = AnomalyGAN(sess)
    optim, loss, resloss, discloss, w, samples, query, grads = init_anomaly(sess, net)
    t = -1
    print()
    for i in range(n_querys):
        print(
            '\r{0}/{1} querys prosessed, ETA: {2}'.format(i, n_querys, datetime.timedelta(seconds=t*(n_querys-i))),
            flush=True, end='')
        query_img = (query_set[i, :, :, :])
        query_img = query_img[np.newaxis, :, :, :]
        start = time()
        im, losses, r_loss, d_loss, noise = anomaly(sess, query_img, optim, loss, resloss, discloss, w, query,
                                                    grads, samples, np.arange(0, 12))

        end = time()
        t = end - start
        if np.argmax(y_test[i]) == 10:
            label = 'Omni'
            loss_omni.append(np.min(losses))
        else:
            if np.argmax(y_test[i]):
                label = ''
                loss_mnist.append(np.min(losses))
        loss_all.append(losses)
        if plot:
            rows, cols = 1, 9
            fig, axes = plt.subplots(figsize=(15, 5), nrows=rows, ncols=cols, sharex=True, sharey=True, squeeze=False)
            plot_query = True
            k = 0
            for ax_row in axes:
                for ax in ax_row:
                    if plot_query:
                        ax.imshow(np.squeeze(query_img), cmap='Greys_r')
                        ax.xaxis.set_visible(False)
                        ax.yaxis.set_visible(False)
                        ax.set_title('Q ' + label)
                        plot_query = False
                    else:
                        ax.imshow(np.squeeze(im[k, :, :, 0]), cmap='Greys_r')
                        ax.xaxis.set_visible(False)
                        ax.yaxis.set_visible(False)
                        ax.set_title('R \n {0:.3f} \n {0:.3f} \n {0:.3f}'.format(losses[k][0], r_loss[k][0], d_loss[k][0]))
                        k = k + 1
            plt.suptitle('{} \n \n'.format(np.mean(losses)))
            plt.show()

fig, axes = plt.subplots(figsize=(15, 5), nrows=1, ncols=2, sharex=True, sharey=True, squeeze=True)
axes[0].hist(loss_mnist)
axes[0].set_title('Mnist')
axes[1].hist(loss_omni)
axes[1].set_title('Omniglott')
plt.show()
print()
print(loss_mnist)
print(loss_omni)
