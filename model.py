import tensorflow as tf
import numpy as np


# The Deep Convolutional GAN class, for mnist.

class DCGAN(object):

    def __init__(self, sess, input_width=28, input_height=28, channels=1, batch_size=64,
                 z_dim = 100,checkpoint_dir=None, log_dir=None, dataset = 'mnist',
                 lrelu_alpha = 0.2):
        self.sess = sess
        self.batch_size = batch_size
        self.input_width = input_width
        self.input_height = input_height
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.z_dim = z_dim
        self.channels = channels
        self.dataset = dataset
        self.lrelu_alpha = lrelu_alpha
        self.dataset = dataset
        self.build_model()

    def build_model(self):
        image_dims = [self.input_width, self.input_height, self.channels]
        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        inputs = self.inputs

        def sigmoid_cross_entropy_with_logits(x, y):
          try:
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
          except:
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        if self.dataset == 'mnist':
            self.G = self.generator_mnist(self.z)
            self.D, self.D_logits = self.discrimimnator_mnist(inputs, reuse=False)
            self.D_, self.D_logits_ = self.discrimimnator_mnist(self.G, reuse=True)

        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.g_image = tf.summary.image("Generated_image", self.G)

        vars = tf.trainable_variables()

        self.d_vars = [var for var in vars if var.name.startswith('Discriminator')]
        self.g_vars = [var for var in vars if var.name.startswith('Generator')]

        self.saver = tf.train.Saver()

    def train(self, config):
        lr = config.learning_rate
        beta1 = config.beta1
        d_opt = tf.train.AdamOptimizer(learning_rate=lr,
                                       beta1=beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_opt = tf.train.AdamOptimizer(learning_rate=lr,
                                       beta1=beta1).minimize(self.g_loss, var_list=self.g_vars)




    def discrimimnator_mnist(self, x, reuse=False, isTrain=True, name='Discriminator'):

        def lrelu(x, alpha=0.2):
            return tf.maximum(alpha * x, x)

        with tf.variable_scope(name, reuse=reuse):
            conv0 = tf.layers.conv2d(x, 128, [4, 4], strides=(2,2), padding='same', name='d_conv0')
            a0 = lrelu(conv0, alpha = self.lrelu_alpha)

            conv1 = tf.layers.conv2d(a0, 256, [4, 4], strides=(2,2), padding='same', name='d_conv1')
            a1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), self.lrelu_alpha)

            conv2 = tf.layers.conv2d(a1, 512, [4, 4], strides=(2,2), padding='same', name='d_conv2')
            a2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), self.lrelu_alpha)

            conv3 = tf.layers.conv2d(a2, 1024, [4, 4], strides=(2,2), padding='same', name='d_conv3')
            a3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), self.lrelu_alpha)

            conv4 = tf.layers.conv2d(a3, 1, [4, 4], strides=(2,2), padding='same', name='d_conv4')
            a4 = tf.nn.sigmoid(conv4)

            return a4, conv4

    def generator_mnist(self, z, reuse=False, isTrain=True, name='Generator'):
        def lrelu(x, alpha=0.2):
            return tf.maximum(alpha * x, x)

        with tf.variable_scope(name, reuse=reuse):
            # 1st hidden layer

            dense0 = tf.layers.dense(z, 1024*7*7, name='g_dense0')
            batch0 = tf.layers.batch_normalization(dense0, training=isTrain)

            x = tf.reshape(batch0, [-1, 7, 7, 1024])

            conv0 = tf.layers.conv2d_transpose(x, 512, [4, 4], strides=(1, 1), padding='valid')
            a0 = lrelu(tf.layers.batch_normalization(conv0, training=isTrain), 0.2)

            conv1 = tf.layers.conv2d_transpose(a0, 256, [4, 4], strides=(2, 2), padding='same')
            a1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

            conv2 = tf.layers.conv2d_transpose(a1, 128, [4, 4], strides=(2, 2), padding='same')
            a2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

            conv3 = tf.layers.conv2d_transpose(a2, 1, [4, 4], strides=(2, 2), padding='same')
            a3 = tf.nn.tanh(conv3)

            return a3

sess = tf.get_default_session()
flags = tf.app.flags
flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate')
flags.DEFINE_float('beta1', 0.5, 'Momentum')
FLAGS = flags.FLAGS
dc_mnist = DCGAN(sess).train(FLAGS)

