import tensorflow as tf
import numpy as np
from time import time
import datetime

# The Deep Convolutional GAN class, for mnist.

class AnomalyGAN(object):

    def __init__(self, sess, input_width=64, input_height=64, channels=1,
                 z_dim = 100, save_dir='./AnomalyGAN_save/'):
        """
        Inititates the AnomalyGAN class.

        :param sess: tensorflow session
            Tensorflow session assigned to the AnomalyGAN
        :param input_width: Int
            Width of input images
        :param input_height: Int
            Height of input images
        :param channels: Int
            Color channels of input images
        :param z_dim: Int
            Dimension of latent space
        :param save_dir: String
            Save directory for the GAN
        :param dataset: String
            Name of dataset
        """
        self.sess = sess
        self.input_width = input_width
        self.input_height = input_height
        self.save_dir = save_dir
        self.z_dim = z_dim
        self.channels = channels
        self.build_model() # Build model
        try:
            # Restore weights if checkpoint exists
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_dir))
            print('Restored')
        except:
            pass

    def lrelu(self, x, th=0.2):
        return tf.maximum(th * x, x)

    def discrimimnator_mnist(self, x, reuse=False, isTrain=True, name='Discriminator'):
        """
        Disrciminator network for Mnist

        :param x: tensor
            Input to descriminator (image)
        :param reuse: Bool
            Reuse parameters
        :param name: String
            Name of discriminator
        :return: tensor
            Probability of real image
        :return: tensor
            Logits
        """

        with tf.variable_scope(name, reuse=reuse):
            # 1st hidden layer
            conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same')
            lrelu1 = self.lrelu(conv1, 0.2)

            # 2nd hidden layer
            conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
            lrelu2 = self.lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

            # 3rd hidden layer
            conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same')
            lrelu3 = self.lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

            # 4th hidden layer
            conv4 = tf.layers.conv2d(lrelu3, 1024, [4, 4], strides=(2, 2), padding='same')
            lrelu4 = self.lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

            # output layer
            conv5 = tf.layers.conv2d(lrelu4, 1, [4, 4], strides=(1, 1), padding='valid')
            o = tf.nn.sigmoid(conv5)
        return o, conv5


    def generator_mnist(self, z, reuse=False, isTrain=True, name='Generator'):
        """
        Generator network for Mnist

        :param z: tensor
            latent vector
        :param reuse:
            Reuse parameters
        :param name:
            Name of Generator
        :return: tensor
            Generated image
        """

        with tf.variable_scope(name, reuse=reuse):
            # 1st hidden layer
            conv1 = tf.layers.conv2d_transpose(z, 1024, [4, 4], strides=(1, 1), padding='valid')
            lrelu1 = self.lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

            # 2nd hidden layer
            conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [4, 4], strides=(2, 2), padding='same')
            lrelu2 = self.lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

            # 3rd hidden layer
            conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [4, 4], strides=(2, 2), padding='same')
            lrelu3 = self.lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

            # 4th hidden layer
            conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding='same')
            lrelu4 = self.lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

            # output layer
            conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding='same')
            o = tf.nn.tanh(conv5)
            return o

    def build_model(self):
        """
        Function that builds the DCGAN

        """
        with tf.variable_scope('Placeholders'):
            self.x = tf.placeholder(
                tf.float32, [None, self.input_width, self.input_height, self.channels])
            self.z = tf.placeholder(tf.float32, [None, 1, 1, self.z_dim])
            self.learning_rate = tf.placeholder(tf.float32)
            self.isTrain = tf.placeholder(dtype=tf.bool)

        # networks : generator
        self.G_z = self.generator_mnist(self.z, isTrain=self.isTrain, reuse=False)

        # networks : discriminator
        D_real, D_real_logits = self.discrimimnator_mnist(self.x, isTrain=self.isTrain, reuse=False)
        D_fake, D_fake_logits = self.discrimimnator_mnist(self.G_z, isTrain=self.isTrain, reuse=True)

        with tf.variable_scope('Loss'):
            D_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
            D_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_real)))
            self.D_loss = D_loss_real + D_loss_fake
            self.G_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_real)))

        vars = tf.trainable_variables()

        D_vars = [var for var in vars if var.name.startswith('Discriminator')]
        G_vars = [var for var in vars if var.name.startswith('Generator')]

        # optimizer for each network
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.D_optim = tf.train.AdamOptimizer(self.learning_rate/2, beta1=0.5).minimize(self.D_loss, var_list=D_vars)
            self.G_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.G_loss, var_list=G_vars)

        self.saver = tf.train.Saver()

    def train_model(self, images, batch_size=64, epochs=50, learning_rate=2e-4, verbose=1):
        """

        :param images: numpy array
            Training images
        :param batch_size: Int
            Batch size
        :param epochs: Int
            Number of epochs to train
        :param learning_rate: float
            Learning rate
        :param verbose: Int
            Level of logging
        :return: List
            List of images generated, 4 for each epoch.
        """
        N = len(images) // batch_size # Number of iterations per epoch
        im = list()
        generator_loss = list()
        discriminator_loss = list()
        self.sess.run(tf.global_variables_initializer()) # Otherwise initialize.
        print('Starting GAN training ...')
        for epoch in range(epochs):
            idx = np.random.permutation(len(images))
            images = images[idx]
            if verbose: print('='*30 + ' Epoch {} '.format(epoch+1) + '='*30)
            batch_start = 0
            batch_end = batch_size
            t=-1
            for i in range(N):
                print('\r{0}/{1} iterations, ETA: {2}'.format(i, N, datetime.timedelta(
                    seconds=t * (N - i))),
                    flush=True, end='')
                start = time()
                if batch_end <= len(images):
                    batch = images[batch_start:batch_end, :, :, :]
                    batch_z = np.random.normal(0, 1, size=(batch_size, 1, 1, self.z_dim))
                    if epoch > 5:
                        _ = self.sess.run([self.G_optim],
                                                  feed_dict={self.x: batch, self.z: batch_z,
                                                             self.learning_rate: learning_rate,
                                                             self.isTrain: True})

                    _, d_loss = self.sess.run([self.D_optim, self.D_loss],
                                               feed_dict={self.x: batch, self.z: batch_z,
                                                          self.learning_rate: learning_rate,
                                                          self.isTrain: True})
                    _, g_loss = self.sess.run([self.G_optim , self.G_loss], feed_dict={self.x: batch, self.z: batch_z,
                                                                               self.learning_rate: learning_rate,
                                                                                     self.isTrain: True})
                    batch_start = batch_end
                    batch_end = batch_end + batch_size
                if i % 100 == 0: t = time() - start
            if verbose:
                print('Generator loss {}'.format(g_loss))
                print('Discriminator loss {}'.format(d_loss))
            generator_loss.append(g_loss)
            discriminator_loss.append(d_loss)
            z = np.random.normal(0, 1, size=(1, 1, 1, self.z_dim))
            G = self.sess.run([self.G_z], feed_dict={self.z: z, self.isTrain: False})
            im.append(G)
            self.saver.save(self.sess, save_path=self.save_dir + 'AnomalyGAN.ckpt') # Save parameters.
        return im, generator_loss, discriminator_loss

    def sampler(self, z):
        """
        Copy of generator to generate images for anomaly detection

        :param z: tensor
            Latent vector
        :return: tensor
            Generated Image
        """
        with tf.variable_scope('Generator', reuse=True):
            # 1st hidden layer
            conv1 = tf.layers.conv2d_transpose(z, 1024, [4, 4], strides=(1, 1), padding='valid')
            lrelu1 = self.lrelu(tf.layers.batch_normalization(conv1, training=False), 0.2)

            # 2nd hidden layer
            conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [4, 4], strides=(2, 2), padding='same')
            lrelu2 = self.lrelu(tf.layers.batch_normalization(conv2, training=False), 0.2)

            # 3rd hidden layer
            conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [4, 4], strides=(2, 2), padding='same')
            lrelu3 = self.lrelu(tf.layers.batch_normalization(conv3, training=False), 0.2)

            # 4th hidden layer
            conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding='same')
            lrelu4 = self.lrelu(tf.layers.batch_normalization(conv4, training=False), 0.2)

            # output layer
            conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding='same')
            o = tf.nn.tanh(conv5)
            return o



def init_anomaly(sess, anomalygan):
    """
    Initiates graph for anomaly detection

    :param sess: Tensorflow session
    :param anomalygan: AnomalyGan object
    :return optim: Tensorflow optimizer
    :return loss: Tensorflow placeholder
        The loss
    :return resloss: Tensorflow placeholder
        Residual loss
    :return discloss: Tensorflow placeholder
        Discriminator loss
    :return w: Tensorflow variable
        Latent space vector
    :return samples: Tensorflow placeholder
        generated image from latent space vector
    :return query: Tensorflow placeholder
        Query
    :return grads: Tensorflow placeholder
        Gradients
    """
    learning_rate = 0.25  # Latent space learning rate.
    beta1 = 0.5
    n_seed = 1
    # Create placeholders and variables.
    w = tf.Variable(initial_value=tf.random_normal([n_seed, 1, 1, anomalygan.z_dim], 0, 1),
                         name='qnoise')

    samples = anomalygan.sampler(w)
    query = tf.placeholder(shape=[1, 64, 64, 1], dtype=tf.float32)
    _, real = anomalygan.discrimimnator_mnist(query, reuse=True)
    _, fake = anomalygan.discrimimnator_mnist(samples, reuse=True)

    resloss = tf.reduce_sum(tf.abs(samples - query))
    discloss = tf.reduce_sum(tf.abs(real - fake))
    count = tf.reduce_sum(tf.cast(tf.equal(query, -1), tf.int32))
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
    """
    Calculates anomaly scores on query image using len(seeds) latent vectors

    :param sess: Tensorflow session
    :param query_image: numpy array
    :param optim: Tensorflow optimizer
    :param loss: Tensorflow placeholder
        The loss
    :param resloss: Tensorflow placeholder
        Residual loss
    :param discloss: Tensorflow placeholder
        Discriminator loss
    :param w: Tensorflow variable
        Latent space vector
    :param query: Tensorflow placeholder
        Query
    :param grads: Tensorflow placeholder
        Gradients
    :param samples: Tensorflow placeholder
        Generated image from latent space vector
    :param seeds: int
        seed
    :return G: numpy array
        reconstructed images
    :return losses: numpy array
        losses
    :return r_loss: numpy array
        reconstruction losses
    :return d_loss: numpy array
        discriminator losses
    :return noise: numpy array
        latent vectors
    """
    l = len(seeds)
    G = np.zeros((l, 64, 64, 1))
    losses = np.zeros((l,1))
    r_loss = np.zeros((l,1))
    d_loss = np.zeros((l,1))
    noise = np.zeros((l, 1, 1, w.get_shape()[-1]))
    for i, seed in enumerate(seeds):
        tf.set_random_seed(seed)
        sess.run(w.initializer)
        G[i, :, :, :], losses[i], r_loss[i], d_loss[i], noise[i, :] = anomaly_score(sess, samples, query_image, query, optim, loss, resloss, discloss,w, grads)
    return G, losses, r_loss, d_loss, noise


def anomaly_score(sess, sample, query_image, query, optim, loss, resloss, discloss, w, grads):
    """
    Calculates anomaly score/loss on one latent vector

    :param sess: Tensorflow session
    :param sample: Tensorflow placeholder
        Generated image from latent space vector
    :param query_image: numpy array
    :param query: Tensorflow placeholder
        Query
    :param optim: Tensorflow optimizer
    :param loss: Tensorflow placeholder
        The loss
    :param resloss: Tensorflow placeholder
        Residual loss
    :param discloss: Tensorflow placeholder
        Discriminator loss
    :param w: Tensorflow variable
        Latent space vector
    :param grads: Tensorflow placeholder
        Gradients
    :return samples: numpy array
        reconstructed image
    :return losses: numpy array
        loss
    :return r_loss: numpy array
        reconstruction loss
    :return d_loss: numpy array
        discriminator loss
    :return z_sample: numpy array
        latent vector
    """
    for r in range(5):
        _, losses, r_loss, d_loss, noise, gradients, samples = sess.run(
            [optim, loss, resloss, discloss, w, grads, sample],
            feed_dict={query: query_image})
    z_sample = noise
    samples = sample.eval(session=sess, feed_dict={w: z_sample})
    return samples, losses, r_loss, d_loss, z_sample
