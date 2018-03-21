import tensorflow as tf
import numpy as np


# The Deep Convolutional GAN class, for mnist.

class AnoGan(object):

    def __init__(self, sess, input_width=28, input_height=28, channels=1,
                 z_dim = 100, save_dir='./AnoGan_save/', dataset = 'mnist'):
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
        self.dataset = dataset
        self.build_model() # Build model
        try:
            # Restore weights if checkpoint exists
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.save_dir))
        except:
            pass

    def convolution2d(self, input, output_dim=64, kernel=(5, 5), strides=(2, 2), stddev=0.2, name='conv_2d'):
        """
        Convolutional layer with target output size.

        :param input: tensor
            Input to convolutional layer [batch_size x width x height x channels]
        :param output_dim: Int
            Target output dimension of feature maps
        :param kernel: tuple
            Size of convolutional kernel
        :param strides: tuple
            Strides
        :param stddev: float
            Standard deviation for initialization
        :param name: String
            Name of layer
        :return: tensor
            Output from convolutional layer
        """

        with tf.variable_scope(name):
            # Weights
            W = tf.get_variable('Conv2dW', [kernel[0], kernel[1], input.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            b = tf.get_variable('Conv2db', [output_dim], initializer=tf.zeros_initializer())

            return tf.nn.conv2d(input, W, strides=[1, strides[0], strides[1], 1], padding='SAME') + b

    def deconvolution2d(self, input, output_dim, kernel=(5, 5), strides=(2, 2), stddev=0.2, name='deconv_2d'):
        """
        Deconvolutional layer with target output size.

        :param input: tensor
            Input to deconvolutional layer [batch_size x width x height x channels]
        :param output_dim: Int
            Target output dimension
        :param kernel: tuple
            Size of deconvolutional kernel
        :param strides: tuple
            Strides
        :param stddev: float
            Standard deviation for initialization
        :param name: String
            Name of layer
        :return: tensor
            Output from deconvolutional layer
        """

        with tf.variable_scope(name):
            # Weights
            W = tf.get_variable('Deconv2dW', [kernel[0], kernel[1], output_dim, input.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            b = tf.get_variable('Deconv2db', [output_dim], initializer=tf.zeros_initializer())
            batch_size = tf.shape(input)[0]
            input_shape = input.get_shape().as_list()
            # Output shape
            output_shape = tf.stack([batch_size,
                            int(input_shape[1] * strides[0]),
                            int(input_shape[2] * strides[1]),
                            output_dim])

            deconv = tf.nn.conv2d_transpose(input, W, output_shape=output_shape,
                                            strides=[1, strides[0], strides[1], 1])

            return deconv + b

    def dense_layer(self, input, output_dim, stddev=0.02, name='dense'):
        """
        Dense layer.

        :param input: tensor
            Input to dense layer
        :param output_dim: Int
            Output dimension of dense later
        :param stddev: float
            Standard deviation for initialization
        :param name: String
            Name of layer
        :return: tensor
            Output from dense layer
        """

        with tf.variable_scope(name):
            shape = input.get_shape()
            W = tf.get_variable('DenseW', [shape[1], output_dim], tf.float32,
                                tf.random_normal_initializer(stddev=stddev))
            b = tf.get_variable('Denseb', [output_dim],
                                initializer=tf.zeros_initializer())

            return tf.matmul(input, W) + b

    def batchnormalization(self, input, name='bn'):
        """
        Batch normalization layer

        :param input: tensor
            Input to dense layer
        :param name: String
            Name of layer
        :return: tensor
            Output from batch normalization layer
        """

        with tf.variable_scope(name):

            output_dim = input.get_shape()[-1]
            beta = tf.get_variable('BnBeta', [output_dim],
                                   initializer=tf.zeros_initializer())
            gamma = tf.get_variable('BnGamma', [output_dim],
                                    initializer=tf.ones_initializer())

            if len(input.get_shape()) == 2:
                mean, var = tf.nn.moments(input, [0])
            else:
                mean, var = tf.nn.moments(input, [0, 1, 2])
            return tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-5)

    def leakyReLU(self, input, leak=0.2, name='lrelu'):
        """
        Leaky relu activation layer

        :param input: tensor
            Input to activation layer
        :param leak: float
            Leak parameter
        :param name: String
            Name of activation layer
        :return: tensor
            Activation
        """
        return tf.maximum(input, leak * input)

    def discrimimnator_mnist(self, x, reuse=False, name='Discriminator'):
        """
        Disrciminator network for Mnist

        :param x: tensor
            Input to discriminator (image)
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

            D_conv1 = self.convolution2d(x, output_dim=64, name='D_conv1')
            D_h1 = self.leakyReLU(D_conv1)
            D_conv2 = self.convolution2d(D_h1, output_dim=128, name='D_conv2')
            D_h2 = self.leakyReLU(D_conv2)
            D_h3 = tf.contrib.layers.flatten(D_h2)
            D_h4 = tf.nn.dropout(D_h3, 0.5)
            D_h5 = self.dense_layer(D_h4, output_dim=1, name='D_h5')
            return tf.nn.sigmoid(D_h5), D_h5


    def generator_mnist(self, z, reuse=False, name='Generator'):
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

            G_1 =self. dense_layer(z, output_dim=1024, name='G_1')
            G_bn1 = self.batchnormalization(G_1, name='G_bn1')
            G_h1 = tf.nn.relu(G_bn1)
            G_2 = self.dense_layer(G_h1, output_dim=7 * 7 * 128, name='G_2')
            G_bn2 = self.batchnormalization(G_2, name='G_bn2')
            G_h2 = tf.nn.relu(G_bn2)
            G_r2 = tf.reshape(G_h2, [-1, 7, 7, 128])
            G_conv3 = self.deconvolution2d(G_r2, output_dim=64, name='G_conv3')
            G_bn3 = self.batchnormalization(G_conv3, name='G_bn3')
            G_h3 = tf.nn.relu(G_bn3)
            G_conv4 = self.deconvolution2d(G_h3, output_dim=1, name='G_conv4')
            return tf.nn.sigmoid(G_conv4)



    def build_model(self):
        """
        Function that builds the DCGAN

        """
        with tf.variable_scope('Placeholders'):
            self.inputs = tf.placeholder(
                tf.float32, [None, self.input_width, self.input_height, self.channels])
            self.z = tf.placeholder(tf.float32, [None, self.z_dim])
            self.learning_rate = tf.placeholder(tf.float32)

        if self.dataset == 'mnist':
            self.G = self.generator_mnist(self.z, reuse=False)
            self.D, self.D_logits = self.discrimimnator_mnist(self.inputs, reuse=False)
            self.D_, self.D_logits_ = self.discrimimnator_mnist(self.G, reuse=True)

        with tf.variable_scope('Loss'):
            self.d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D))
            self.d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_))
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

            self.d_loss = tf.reduce_mean(self.d_loss_real + self.d_loss_fake)

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)


        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.g_image = tf.summary.image("Generated_image", self.G)

        vars = tf.trainable_variables()

        self.d_vars = [var for var in vars if var.name.startswith('Discriminator')]
        self.g_vars = [var for var in vars if var.name.startswith('Generator')]

        with tf.variable_scope('Optimizers'):
            self.d_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate/2, beta1=0.1).minimize(
                self.d_loss, var_list=self.d_vars)
            self.g_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.3).minimize(
                self.g_loss, var_list=self.g_vars)

        self.saver = tf.train.Saver()
        self.grads = tf.gradients(self.g_loss, self.g_vars)

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
            d_loss = 0
            g_loss = 0
            batch_start = 0
            batch_end = batch_size
            for i in range(N):
                if batch_end <= len(images):
                    batch = images[batch_start:batch_end, :, :, :]
                    batch_z = np.random.uniform(-1, 1, size=(batch_size, self.z_dim))
                    _, g_loss_ = self.sess.run([self.g_opt, self.g_loss], feed_dict={self.z: batch_z,
                                                                               self.learning_rate: learning_rate})
                    _, d_loss_ = self.sess.run([self.d_opt, self.d_loss],
                                               feed_dict={self.inputs: batch, self.z: batch_z,
                                                          self.learning_rate: learning_rate})
                    d_loss = d_loss + d_loss_
                    g_loss = g_loss + g_loss_
                    batch_start = batch_end
                    batch_end = batch_end + batch_size
            if verbose:
                print(f'Average generator loss {g_loss/N}')
                print(f'Average discriminator loss {d_loss/N}')
            G = self.sess.run([self.G], feed_dict={self.z: np.random.uniform(-1, 1, size=(4, self.z_dim))})
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
            G_1 =self. dense_layer(z, output_dim=1024, name='G_1')  # [-1, 1024]
            G_bn1 = self.batchnormalization(G_1, name='G_bn1')
            G_h1 = tf.nn.relu(G_bn1)
            G_2 = self.dense_layer(G_h1, output_dim=7 * 7 * 128, name='G_2')  # [-1, 7*7*128]
            G_bn2 = self.batchnormalization(G_2, name='G_bn2')
            G_h2 = tf.nn.relu(G_bn2)
            G_r2 = tf.reshape(G_h2, [-1, 7, 7, 128])
            G_conv3 = self.deconvolution2d(G_r2, output_dim=64, name='G_conv3')
            G_bn3 = self.batchnormalization(G_conv3, name='G_bn3')
            G_h3 = tf.nn.relu(G_bn3)
            G_conv4 = self.deconvolution2d(G_h3, output_dim=1, name='G_conv4')
            return tf.nn.sigmoid(G_conv4)

    def init_anomaly(self):
        """
        To initialize anomaly detection
        """
        learning_rate = 0.0007 # Latent space learning rate.
        beta1 = 0.7

        # Create placeholders and variables.
        self.w = tf.Variable(initial_value=tf.random_uniform(minval=-1, maxval=1, shape=[49, self.z_dim]), name='qnoise')
        self.samples = self.sampler(self.w)
        self.query = tf.placeholder(shape=[1, 28, 28, 1], dtype=tf.float32)
        _, real = self.discrimimnator_mnist(self.query, reuse=True)
        _, fake = self.discrimimnator_mnist(self.samples, reuse=True)

        # Loss
        self.loss_w = 0.9 * tf.reduce_sum(tf.abs(self.samples - self.query), axis=[1, 2, 3]) + tf.reduce_sum(
            0.1 * tf.abs(real - fake), axis=1)
        resloss = tf.reduce_sum(tf.abs(self.samples - self.query))
        discloss = tf.reduce_sum(tf.abs(real - fake))
        self.loss = 0.9 * resloss + 0.1 * discloss

        # Optimizer
        self.optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(self.loss, var_list=[self.w])

    def anomaly(self, query_image):
        """

        :param query_image: numpy array
            Input image for anomaly score
        :return samples: numpy array
            The generated images for query image
        :return losses: float
            The loss for all generated images
        :return best_index: Int
            The index for the best match in samples for the query image
        :return w_loss: numpy array
            The losses for each of the generated images in samples.
        """
        self.sess.run(self.w.initializer)
        adam_init = [var.initializer for var in tf.global_variables() if 'qnoise/Adam' in var.name]
        self.sess.run(adam_init)
        beta_init = [var.initializer for var in tf.global_variables() if 'beta1_power' in var.name]
        self.sess.run(beta_init)
        beta_init = [var.initializer for var in tf.global_variables() if 'beta2_power' in var.name]
        self.sess.run(beta_init)
        samples, losses, best_index, w_loss = self.anomaly_score(query_image)

        return samples, losses, best_index, w_loss

    def anomaly_score(self, query_image):
        """
        :param query_image: numpy array
            Input image
        :return samples: numpy array
            The generated images for query image
        :return losses: float
            The loss for all generated images
        :return best_index: Int
            The index for the best match in samples for the query image
        :return w_loss: numpy array
            The losses for each of the generated images in samples.
        """
        for r in range(2000):
            _, losses, noise, current_sample = self.sess.run([self.optim, self.loss, self.w, self.samples],
                                                              feed_dict={self.query: query_image})

        w_loss, losses, samples = self.sess.run([self.loss_w, self.loss, self.samples],
                                        feed_dict={self.query: query_image})
        best_index = np.argmin(w_loss)
        return samples, losses, best_index, w_loss

tf.reset_default_graph()
sess = tf.Session()
net = AnoGan(sess)
"""
tf.reset_default_graph()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True, reshape=False, validation_size=5000)
sess = tf.Session()
net = AnoGan(sess)

#im = net.train_model(((mnist.train.images-0.5)/0.5), epochs=100, batch_size=64, learning_rate=0.0002)


rows, cols = 10, 10
fig, axes = plt.subplots(figsize=(12,10), nrows=rows, ncols=cols, sharex=True, sharey=True, squeeze=False)
k = 0
print(np.shape(im))
for ax_row in axes:
    for ax in ax_row:

        img = im[k]
        img = img[0][np.random.randint(0,4)][:,:,:]
        k = k+1
        ax.imshow(np.squeeze(img), cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
plt.show()

query_img = (mnist.test.images[np.random.randint(0,9000),:,:,:]-0.5)/0.5
query_img = query_img[np.newaxis,:,:,:]
fig, axes = plt.subplots(figsize=(12,10), nrows=1, ncols=2, sharex=True, sharey=True, squeeze=False)
img, loss = net.anomaly(query_img)
print(np.shape(img[0,:,:,0]))
axes[0,0].imshow(query_img[0,:,:,0], cmap='Greys_r')
axes[0,0].set_title('Input image')
axes[0,1].imshow(img[0,:,:,0], cmap='Greys_r')
axes[0,1].set_title('Reconstructed image')
plt.suptitle(f'loss: {loss}')
plt.show()


rows, cols = 5, 5
fig, axes = plt.subplots(figsize=(12,10), nrows=rows, ncols=cols, sharex=True, sharey=True, squeeze=False)
k = 0
query_img = (mnist.test.images[np.random.randint(0,9000),:,:,:]-0.5)/0.5
query_img = query_img[np.newaxis,:,:,:]
im, loss = net.anomaly(query_img)
print(np.shape(im))
for ax_row in axes:
    for ax in ax_row:
        if k == 24:
            ax.imshow(np.squeeze(query_img), cmap='Greys_r')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_title('Query image')

        else:
            img = im[k]
            k = k+1
            ax.imshow(np.squeeze(img), cmap='Greys_r')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
plt.show()
"""
