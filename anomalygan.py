import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from time import time



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
            self.D_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.D_loss, var_list=D_vars)
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
        try:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_dir
                                                                     )) # Restore if checkpoint exists.
        except:
            self.sess.run(tf.global_variables_initializer()) # Otherwise initialize.
        print('Starting GAN training ...')
        for epoch in range(epochs):
            idx = np.random.permutation(len(images))
            images = images[idx]
            if verbose: print('='*30 + f' Epoch {epoch+1} ' + '='*30)
            batch_start = 0
            batch_end = batch_size
            for i in range(N):
                if batch_end <= len(images):
                    batch = images[batch_start:batch_end, :, :, :]
                    batch_z = np.random.normal(0, 1, size=(batch_size, 1, 1, self.z_dim))
                    _, d_loss = self.sess.run([self.D_optim, self.D_loss],
                                               feed_dict={self.x: batch, self.z: batch_z,
                                                          self.learning_rate: learning_rate,
                                                          self.isTrain: True})
                    _, g_loss = self.sess.run([self.G_optim , self.G_loss], feed_dict={self.x: batch, self.z: batch_z,
                                                                               self.learning_rate: learning_rate,
                                                                                     self.isTrain: True})
                    batch_start = batch_end
                    batch_end = batch_end + batch_size
            if verbose:
                print(f'Generator loss {g_loss}')
                print(f'Discriminator loss {d_loss}')
            z = np.random.normal(0, 1, size=(1, 1, 1, self.z_dim))
            G = self.sess.run([self.G_z], feed_dict={self.z: z, self.isTrain: False})
            im.append(G)
            self.saver.save(self.sess, save_path=self.save_dir + 'AnomalyGAN.ckpt') # Save parameters.
        return im

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


tf.reset_default_graph()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True, reshape=False, validation_size=5000)
sess = tf.Session()
net = AnomalyGAN(sess)
train_set = tf.image.resize_images(mnist.train.images, [64, 64]).eval(session=sess)
train_set = (train_set - 0.5) / 0.5 # normalization; range: -1 ~ 1

im = net.train_model(train_set, epochs=20, batch_size=100, learning_rate=2e-4)


rows, cols = 1, 10
fig, axes = plt.subplots(figsize=(10,4), nrows=rows, ncols=cols, sharex=True, sharey=True, squeeze=False)
k = 0
for ax_row in axes:
    for ax in ax_row:
        img = im[k]
        img = img[0][0][:,:,:]
        k = k+1
        ax.imshow(np.squeeze(img), cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
plt.show()



