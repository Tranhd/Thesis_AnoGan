#from anomalygan import AnomalyGAN
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
#import matplotlib.pyplot as plt

"Functions to run the anomaly score"


def init_anomaly(sess, anomalygan):
    learning_rate = 0.07  # Latent space learning rate.
    beta1 = 0.7
    n_seed = 1
    # Create placeholders and variables.
    w = tf.Variable(initial_value=tf.random_normal([n_seed, 1, 1, anomalygan.z_dim], 0, 1),
                         name='qnoise')

    samples = anomalygan.sampler(w)
    # print(self.samples.get_shape())
    query = tf.placeholder(shape=[1, 64, 64, 1], dtype=tf.float32)
    _, real = anomalygan.discrimimnator_mnist(query, reuse=True)
    _, fake = anomalygan.discrimimnator_mnist(samples, reuse=True)

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

    return optim, loss, resloss, discloss, w, samples, query, grads


def anomaly(sess, query_image, optim, loss, resloss, discloss, w, query, grads, samples, seeds):
    l = len(seeds)
    G = np.zeros((l, 64, 64, 1))
    losses = np.zeros((l,1))
    r_loss = np.zeros((l,1))
    d_loss = np.zeros((l,1))
    noise = np.zeros((l, 1, 1, w.get_shape()[-1]))
    for i, seed in enumerate(seeds):
        tf.set_random_seed(seed)
        sess.run(w.initializer)
        G[i, :, :, :], losses[i], r_loss, d_loss, noise[i, :] = anomaly_score(sess, samples, query_image, query, optim, loss, resloss, discloss,w, grads)
    return G, losses, r_loss, d_loss, noise


def anomaly_score(sess, sample, query_image, query, optim, loss, resloss, discloss, w, grads):
    for r in range(100):
        _, losses, r_loss, d_loss, noise, gradients, samples = sess.run(
            [optim, loss, resloss, discloss, w, grads, sample],
            feed_dict={query: query_image})
    z_sample = noise
    samples = sample.eval(session=sess, feed_dict={w: z_sample})
    return samples, losses, r_loss, d_loss, z_sample
