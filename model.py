import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


# The Deep Convolutional GAN class, for mnist.

class AnoGan(object):

    def __init__(self, sess, input_width=28, input_height=28, channels=1,
                 z_dim = 100, save_dir='./AnoGan_save/', log_dir=None, dataset = 'mnist',
                 lrelu_alpha = 0.2):
        self.sess = sess
        self.input_width = input_width
        self.input_height = input_height
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.z_dim = z_dim
        self.channels = channels
        self.dataset = dataset
        self.lrelu_alpha = lrelu_alpha
        self.build_model()

    def discrimimnator_mnist(self, x, reuse=False, name='Discriminator'):

        def lrelu(x, alpha=0.2):
            return tf.maximum(alpha * x, x)

        with tf.variable_scope(name, reuse=reuse):
            print(x.get_shape())

            conv0 = tf.layers.conv2d(x, 64, [5, 5], strides=(2,2), padding='same')
            a0 = lrelu(conv0, alpha = self.lrelu_alpha)
            print(a0.get_shape())

            conv1 = tf.layers.conv2d(a0, 128, [4, 4], strides=(2,2), padding='same')
            a1 = lrelu(tf.layers.batch_normalization(conv1, training=self.isTrain), self.lrelu_alpha)
            print(a1.get_shape())

            conv2 = tf.layers.conv2d(a1, 1, [4, 4], strides=(2,2), padding='same')
            a2 = lrelu(tf.layers.batch_normalization(conv2, training=self.isTrain), self.lrelu_alpha)
            print(a2.get_shape())

            conv3 = tf.layers.dense(tf.layers.flatten(a2), 1)
            a3 = tf.nn.sigmoid(conv3)
            print(a3.get_shape())

            return a3, conv3

    def generator_mnist(self, z, reuse=False, name='Generator'):
        def lrelu(x, alpha=0.2):
            return tf.maximum(alpha * x, x)

        with tf.variable_scope(name, reuse=reuse):
            # 1st hidden layer

            dense0 = tf.layers.dense(z, 128*7*7)
            batch0 = lrelu(tf.layers.batch_normalization(dense0, training=self.isTrain))
            print(batch0.get_shape())
            x = tf.reshape(batch0, [-1, 7, 7, 128])
            print(x.get_shape())

            conv0 = tf.layers.conv2d_transpose(x, 64, [4, 4], strides=(1, 1), padding='same')
            a0 = lrelu(tf.layers.batch_normalization(conv0, training=self.isTrain), 0.2)
            print(a0.get_shape())

            conv1 = tf.layers.conv2d_transpose(a0, 32, [4, 4], strides=(1, 1), padding='same')
            a1 = lrelu(tf.layers.batch_normalization(conv1, training=self.isTrain), 0.2)
            print(a1.get_shape())

            conv2 = tf.layers.conv2d_transpose(a1, 32, [3, 3], strides=(2, 2), padding='same')
            a2 = lrelu(tf.layers.batch_normalization(conv2, training=self.isTrain), 0.2)
            print(a2.get_shape())

            conv3 = tf.layers.conv2d_transpose(a2, 1, [4, 4], strides=(2, 2), padding='same')
            a3 = tf.nn.tanh(conv3)
            print(a3.get_shape())

            return a3

    def build_model(self):
        with tf.variable_scope('Placeholders'):
            self.inputs = tf.placeholder(
                tf.float32, [None, self.input_width, self.input_height, self.channels])
            self.z = tf.placeholder(tf.float32, [None, self.z_dim])
            self.isTrain = tf.placeholder(tf.bool)
            self.learning_rate = tf.placeholder(tf.float32)

        if self.dataset == 'mnist':
            self.G = self.generator_mnist(self.z, reuse=False)
            self.D, self.D_logits = self.discrimimnator_mnist(self.inputs, reuse=False)
            self.D_, self.D_logits_ = self.discrimimnator_mnist(self.G, reuse=True)


        with tf.variable_scope('Loss'):
            self.d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)*(1 - 0.1)))
            self.d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
            self.g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

            self.d_loss = self.d_loss_real + self.d_loss_fake

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)


        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.g_image = tf.summary.image("Generated_image", self.G)

        vars = tf.trainable_variables()

        self.d_vars = [var for var in vars if var.name.startswith('Discriminator')]
        self.g_vars = [var for var in vars if var.name.startswith('Generator')]

        with tf.variable_scope('Optimizers'):
            self.d_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                self.d_loss, var_list=self.d_vars)
            self.g_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                self.g_loss, var_list=self.g_vars)

        self.saver = tf.train.Saver()
        self.grads = tf.gradients(self.g_loss, self.g_vars)

    def train_model(self, images, batch_size=64, epochs=50, learning_rate=1e-3, beta1 = 0.002, verbose=1):
        N = len(images) // batch_size # Number of iterations per epoch
        im = list()
        try:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_dir)) # Restore if checkpoint exists.
        except:
            self.sess.run(tf.global_variables_initializer()) # Otherwise initialize.
        print('Starting GAN training ...')
        for epoch in range(epochs):
            idx = np.random.permutation(len(images))
            images = images[idx]
            if verbose: print('='*30 + f' Epoch {epoch+1} ' + '='*30)
            d_loss = 0
            g_loss = 0
            batch_start = 0
            batch_end = batch_size
            for i in range(N):
                if batch_end <= len(images):
                    batch = images[batch_start:batch_end, :, :, :]
                    batch_z = np.random.uniform(-1, 1, size=(batch_size, self.z_dim))
                    _, g_loss_= sess.run([self.g_opt, self.g_loss], feed_dict={self.z: batch_z, self.isTrain: True,
                                                                               self.learning_rate: learning_rate})
                    _, d_loss_ = self.sess.run([self.d_opt, self.d_loss],
                                               feed_dict={self.inputs: batch, self.z: batch_z, self.isTrain: True,
                                                          self.learning_rate: learning_rate})
                    d_loss = d_loss + d_loss_
                    g_loss = g_loss + g_loss_
                    batch_start = batch_end
                    batch_end = batch_end + batch_size
                    #print(grads)
            if verbose:
                print(f'Average generator loss {g_loss/N}')
                print(f'Average discriminator loss {d_loss/N}')
            G = self.sess.run([self.G], feed_dict={self.z: np.random.uniform(-1, 1, size=(1, self.z_dim)), self.isTrain: False})
            im.append(G)
        self.saver.save(self.sess, save_path=self.save_dir + 'AnoGan.ckpt') # Save parameters.
        return im




tf.reset_default_graph()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True, reshape=False, validation_size=5000)
sess = tf.Session()
net = AnoGan(sess)
im = net.train_model(((mnist.train.images[1:400,:,:,:])-0.5)/0.5, epochs=100, batch_size=100, learning_rate=0.02)
rows, cols = 5, 5
fig, axes = plt.subplots(figsize=(8,8), nrows=rows, ncols=cols, sharex=True, sharey=True, squeeze=False)
k = 0
print(np.shape(im[2]))
for ax_row in axes:
    for ax in ax_row:
        img = im[k+74]
        img = img[0][0][:,:,:]
        k = k+1
        ax.imshow(np.squeeze(img), cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
plt.show()