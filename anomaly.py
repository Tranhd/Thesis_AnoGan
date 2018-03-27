from anomalygan import AnoGan
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt



def init_anomaly(sess, anogan):
    learning_rate = 0.07  # Latent space learning rate.
    beta1 = 0.6
    n_seed = 1
    # Create placeholders and variables.
    w = tf.Variable(initial_value=tf.random_normal([n_seed, 1, 1, anogan.z_dim], 0, 1),
                         name='qnoise')

    samples = anogan.sampler(w)
    # print(self.samples.get_shape())
    query = tf.placeholder(shape=[1, 64, 64, 1], dtype=tf.float32)
    _, real = anogan.discrimimnator_mnist(query, reuse=True)
    _, fake = anogan.discrimimnator_mnist(samples, reuse=True)

    resloss = tf.reduce_mean(tf.abs(samples - query))
    discloss = tf.reduce_mean(tf.abs(real - fake))
    loss = 0.9 * resloss + 0.1 * discloss
    grads = tf.gradients(resloss, w)

    # Optimizer
    optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(loss, var_list=w)
    adam_init = [var.initializer for var in tf.global_variables() if 'qnoise/Adam' in var.name]
    sess.run(adam_init)
    beta_init = [var.initializer for var in tf.global_variables() if 'beta1_power' in var.name]
    sess.run(beta_init)
    beta_init = [var.initializer for var in tf.global_variables() if 'beta2_power' in var.name]
    sess.run(beta_init)

    return optim, loss, w, samples, query, grads


def anomaly(sess, query_image, optim, loss, w, query, grads, samples, seeds):
    l = len(seeds)
    G = np.zeros((l, 64, 64, 1))
    losses = np.zeros((l,1))
    noise = np.zeros((l, 1, 1, w.get_shape()[-1]))
    for i, seed in enumerate(seeds):
        tf.set_random_seed(seed)
        sess.run(w.initializer)
        G[i, :, :, 0], losses[i], noise[i, :] = anomaly_score(sess, samples, query_image, query, optim, loss, w, grads)
    return G, losses, noise


def anomaly_score(sess, sample, query_image, query, optim, loss, w, grads):
    for r in range(100):
        _, losses, noise, gradients, samples = sess.run(
            [optim, loss, w, grads, sample],
            feed_dict={query: query_image})
    z_sample = noise
    samples = sample.eval(session=sess, feed_dict={w: z_sample})
    return samples, losses, z_sample

mnist = input_data.read_data_sets('MNIST_data', one_hot=True, reshape=False, validation_size=5000)
with tf.Session() as sess:
    train_set = tf.image.resize_images(mnist.train.images, [64, 64]).eval(session=sess)
    train_set = (train_set - 0.5) / 0.5  # normalization; range: -1 ~ 1
    net = AnoGan(sess)
    rows, cols = 1, 12
    fig, axes = plt.subplots(figsize=(15, 5), nrows=rows, ncols=cols, sharex=True, sharey=True, squeeze=False)
    k = 0
    optim, loss, w, samples, query, grads = init_anomaly(sess, net)
    query_img = (train_set[np.random.randint(0, 9000), :, :, :])
    query_img = query_img[np.newaxis, :, :, :]
    for ax_row in axes:
        for ax in ax_row:
            if k == 0:
                ax.imshow(np.squeeze(query_img), cmap='Greys_r')
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.set_title('Query image')
            else:
                im, losses, noise = anomaly(sess, query_img, optim, loss, w, query, grads, samples)
                ax.imshow(np.squeeze(im[0, :, :, 0]), cmap='Greys_r')
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.set_title('Restored')
            k = k + 1
    plt.show()
