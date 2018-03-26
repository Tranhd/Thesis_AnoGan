import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from time import time



# The Deep Convolutional GAN class, for mnist.

class AnoGan(object):

    def __init__(self, sess, input_width=64, input_height=64, channels=1,
                 z_dim = 100, save_dir='./AnoGan_save/'):
        """
        Inititates the AnoGan class.

        :param sess: tensorflow session
            Tensorflow session assigned to the AnoGan
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
            print('restored')
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
            batch_size = -1
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
            z = np.random.normal(0, 1, size=(batch_size, 1, 1, self.z_dim))
            G = self.sess.run([self.G_z], feed_dict={self.z: z, self.isTrain: False})
            im.append(G)
            self.saver.save(self.sess, save_path=self.save_dir + 'AnoGan.ckpt') # Save parameters.
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


# tf.reset_default_graph()
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True, reshape=False, validation_size=5000)
# sess = tf.Session()
# net = AnoGan(sess)
# train_set = tf.image.resize_images(mnist.train.images, [64, 64]).eval(session=sess)
# train_set = (train_set - 0.5) / 0.5 # normalization; range: -1 ~ 1
#
# im = net.train_model(train_set, epochs=4, batch_size=100, learning_rate=2e-4)
#
#
# rows, cols = 2, 5
# fig, axes = plt.subplots(figsize=(10,4), nrows=rows, ncols=cols, sharex=True, sharey=True, squeeze=False)
# k = 0
# for ax_row in axes:
#     for ax in ax_row:
#         img = im[k]
#         img = img[0][0][:,:,:]
#         k = k+1
#         ax.imshow(np.squeeze(img), cmap='Greys_r')
#         ax.xaxis.set_visible(False)
#         ax.yaxis.set_visible(False)
# plt.show()



#     def init_anomaly(self):
#         """
#         To initialize anomaly detection
#         """
#         learning_rate = 0.07 # Latent space learning rate.
#         beta1 = 0.4
#         self.n_seed = 1
#         # Create placeholders and variables.
#         self.w = tf.Variable(initial_value=tf.random_uniform(minval=-1, maxval=1, shape=[self.n_seed, self.z_dim]), name='qnoise')
#
#         #self.w = tf.Variable('qnoise', [1, self.z_dim], tf.float32,
#         #                    tf.random_normal_initializer(stddev=0.01))
#         self.samples = self.sampler(self.w)
#         #print(self.samples.get_shape())
#         self.query = tf.placeholder(shape=[1, 28, 28, 1], dtype=tf.float32)
#         _, real = self.discrimimnator_mnist(self.query, reuse=True)
#         _, fake = self.discrimimnator_mnist(self.samples, reuse=True)
#
#         # Loss
#         self.loss_w = 0.9 * tf.reduce_sum(tf.abs(self.samples - self.query), axis=[1, 2, 3]
#                                           )+0.1*tf.reduce_sum(tf.abs(real - fake), axis=1)
#         self.resloss = tf.reduce_mean(tf.abs(self.samples - self.query))
#         self.discloss = tf.reduce_mean(tf.abs(real - fake))
#         self.loss = 0.9 * self.resloss + 0.1 * self.discloss
#         self.grads = tf.gradients(self.resloss, self.w)
#         #print(tf.abs(self.samples - self.query).get_shape())
#         #print(tf.abs(real - fake).get_shape())
#
#         # Optimizer
#         self.optim =tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(self.loss, var_list=self.w)
#         adam_init = [var.initializer for var in tf.global_variables() if 'qnoise/Adam' in var.name]
#         self.sess.run(adam_init)
#         beta_init = [var.initializer for var in tf.global_variables() if 'beta1_power' in var.name]
#         self.sess.run(beta_init)
#         beta_init = [var.initializer for var in tf.global_variables() if 'beta2_power' in var.name]
#         self.sess.run(beta_init)
#
#     def anomaly(self, query_image):
#         """
#
#         :param query_image: numpy array
#             Input image for anomaly score
#         :return samples: numpy array
#             The generated images for query image
#         :return losses: float
#             The loss for all generated images
#         :return best_index: Int
#             The index for the best match in samples for the query image
#         :return w_loss: numpy array
#             The losses for each of the generated images in samples.
#         """
#         self.sess.run(self.w.initializer)
#
#         samples, losses, best_index, w_loss, w = self.anomaly_score(query_image)
#
#         return samples, losses, best_index, w_loss, w
#
#     def anomaly_score(self, query_image):
#         """
#         :param query_image: numpy array
#             Input image
#         :return samples: numpy array
#             The generated images for query image
#         :return losses: float
#             The loss for all generated images
#         :return best_index: Int
#             The index for the best match in samples for the query image
#         :return w_loss: numpy array
#             The losses for each of the generated images in samples.
#         """
#         #print(self.w.eval(session=self.sess))
#         for r in range(100):
#             _, losses, noise, discloss, resloss, sampl, grads = self.sess.run([self.optim, self.loss, self.w, self.discloss, self.resloss, self.samples, self.grads],
#                                                           feed_dict={self.query: query_image})
#             #print(grads)
#         samples, w_loss, losses, w, discloss, resloss = self.sess.run([self.samples, self.loss_w, self.loss, self.w, self.discloss, self.resloss],
#                                         feed_dict={self.query: query_image})
#         z_sample = noise
#         samples = self.sampler(self.w)
#         samples = self.sess.run(samples, feed_dict={self.w: z_sample})
#         best_index = np.argmin(w_loss)
#         #print(z_sample)
#         #print(f'discloss {discloss}')
#         #print(f'resloss {resloss}')
#         return samples, losses, best_index, w_loss, w
#
#
#
# # query_img = (mnist.test.images[19,:,:,:])
# # query_img = query_img[np.newaxis,:,:,:]
# # fig, axes = plt.subplots(figsize=(12,10), nrows=1, ncols=2, sharex=True, sharey=True, squeeze=False)
# # net.init_anomaly()
# # img, loss, best, _, w = net.anomaly(query_img)
# # print(w)
# # print(np.shape(img[0,:,:,0]))
# # axes[0,0].imshow(query_img[0,:,:,0], cmap='Greys_r')
# # axes[0,0].set_title('Input image')
# # axes[0,1].imshow(img[0,:,:,0], cmap='Greys_r')
# # axes[0,1].set_title('Reconstructed image')
# # plt.suptitle(f'loss: {loss}')
# # plt.show()
#
# rows, cols = 5, 5
# fig, axes = plt.subplots(figsize=(12,10), nrows=rows, ncols=cols, sharex=True, sharey=True, squeeze=False)
# k = 0
# net.init_anomaly()
# for ax_row in axes:
#     for ax in ax_row:
#         query_img = (mnist.test.images[np.random.randint(0, 9000), :, :, :])
#         query_img = query_img[np.newaxis, :, :, :]
#         im, loss, _,_, w = net.anomaly(query_img)
#         if k%2:
#             ax.imshow(np.squeeze(im[0,:,:,0]), cmap='Greys_r')
#             ax.xaxis.set_visible(False)
#             ax.yaxis.set_visible(False)
#         else:
#             ax.imshow(np.squeeze(query_img), cmap='Greys_r')
#             ax.xaxis.set_visible(False)
#             ax.yaxis.set_visible(False)
#             ax.set_title('Query image')
#         k = k + 1
# plt.show()
