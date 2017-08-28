#-------------------------------------------------------------------------------
# Author: Alexey Simonov <alexey.simonov@gmail.com>
# Date:   28.08.17
#-------------------------------------------------------------------------------

"""
Traffic Lights Classifier using Conv Nets

Class implementing a multi-class classifier for traffic lights images.
Inspired by CIFAR-related convolutional neural networks.
Using pure tensorflow implementation.
Developed with tensorflow 1.1 (GPU) on python 2.7.13 (conda).

Takes an image of size 32x32 with 3 channels in OpenCV convention (BGR uint8)

Trained on Bosch Small Traffic Lights dataset

Todo:
    *
"""

import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import math
from datetime import datetime


class TLClassifierCNN:
  # initialization defaults as class variables
  _trunc_normal_stddev = 0.05
  _bias_init = 0.1
  _L1_kernel_size = 5
  _L1_out_channels = 64
  _kernel1_strides = [1, 1, 1, 1]
  _pool1_kernel = [1, 2, 2, 1]
  _pool1_strides = [1, 2, 2, 1]
  _L2_kernel_size = 5
  _L2_out_channels = 32
  _kernel2_strides = [1, 1, 1, 1]
  _pool2_kernel = [1, 2, 2, 1]
  _pool2_strides = [1, 2, 2, 1]

  def __init__(self):
    self._global_step = None
    # input placeholders
    self._images = None
    self._labels = None
    # transform input
    self._images_float = None
    self._images_std = None
    # layer setup
    # Layer 1
    self._kernel1 = None
    self._conv1 = None
    self._bias1 = None
    self._biased1 = None
    self._conv1r = None
    self._pool1 = None
    # Layer 2
    self._kernel2 = None
    self._conv2 = None
    self._bias2 = None
    self._biased2 = None
    self._conv2r = None
    self._pool2 = None
    # reshape
    self._reshape2 = None
    # dropout
    self._keep_prob = None
    self._dropout = None
    # Level 3 FC
    self._weights3 = None
    self._biases3 = None
    self._local3 = None
    # softmax
    self._prediction = None
    self._cross_entropy = None
    self._loss = None
    self._true_class = None
    self._predicted_class = None
    self._accuracy = None
    # savers
    self._model_param_file = None
    self._saver = None
    self._summary_dir = None
    self._summaries = None
    self._summary_writer = None
    # optimizer
    self._optimizer = None
    # session
    self._session = None

  def variable_summaries(self, var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean/' + str(var.name).replace(":", "_"), mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev/' + str(var.name).replace(":", "_"), stddev)
      tf.summary.scalar('max/' + str(var.name).replace(":", "_"), tf.reduce_max(var))
      tf.summary.scalar('min/' + str(var.name).replace(":", "_"), tf.reduce_min(var))
      tf.summary.histogram('histogram/' + str(var.name).replace(":", "_"), var)

  def define_model(self, features_shape, labels_shape):
    """
    Create calculation graph

    Takes batch of OpenCV images of type uint8.
    Does per-image normalization internally.
    Defines CNN.
    Defines summaries.
    """
    self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    # placeholders
    self._images = tf.placeholder(tf.uint8, shape=features_shape, name='images')
    self._labels = tf.placeholder(tf.float32, name='labels')
    # convert type and standardise image to [0,1] values
    self._images_float = tf.image.convert_image_dtype(self._images, tf.float32)
    self._images_std = tf.map_fn(lambda img: tf.image.per_image_standardization(img), self._images_float)
    # layer 1
    k1_params = [self._L1_kernel_size, self._L1_kernel_size, features_shape[3], self._L1_out_channels]
    self._kernel1 = tf.Variable(tf.truncated_normal(k1_params, stddev=self._trunc_normal_stddev), name='L1_kernel')
    self.variable_summaries(self._kernel1)
    self._conv1 = tf.nn.conv2d(self._images_std, self._kernel1, strides=self._kernel1_strides, padding='SAME')
    tf.summary.histogram('conv1', self._conv1)
    self._bias1 = tf.Variable(tf.constant(np.ones(self._L1_out_channels, np.float32) * self._bias_init), name='L1_bias')
    self.variable_summaries(self._bias1)
    self._biased1 = tf.nn.bias_add(self._conv1, self._bias1)
    self._conv1r = tf.nn.relu(self._biased1)
    tf.summary.histogram('conv1r', self._conv1r)
    self._pool1 = tf.nn.max_pool(self._conv1r,
                                 ksize=self._pool1_kernel,
                                 strides=self._pool1_strides,
                                 padding='SAME')
    tf.summary.histogram('pool1', self._pool1)
    # layer 2
    k2_params = [self._L2_kernel_size, self._L2_kernel_size, self._L1_out_channels, self._L2_out_channels]
    self._kernel2 = tf.Variable(tf.truncated_normal(k2_params, stddev=self._trunc_normal_stddev), name='L2_kernel')
    self.variable_summaries(self._kernel2)
    self._conv2 = tf.nn.conv2d(self._pool1, self._kernel2, strides=self._kernel2_strides, padding='SAME')
    tf.summary.histogram('conv2', self._conv2)
    self._bias2 = tf.Variable(tf.constant(np.ones(self._L2_out_channels, np.float32) * self._bias_init), name='L2_bias')
    self.variable_summaries(self._bias2)
    self._biased2 = tf.nn.bias_add(self._conv2, self._bias2)
    self._conv2r = tf.nn.relu(self._biased2)
    tf.summary.histogram('conv2r', self._conv2r)
    self._pool2 = tf.nn.max_pool(self._conv2r,
                                 ksize=self._pool2_kernel,
                                 strides=self._pool2_strides,
                                 padding='SAME')
    tf.summary.histogram('pool2', self._pool2)
    # reshape
    shape = self._pool2.get_shape().as_list()
    dim = np.prod(shape[1:])
    self._reshape2 = tf.reshape(self._pool2, [-1, dim])
    # DROPOUT
    self._keep_prob = tf.placeholder(tf.float32, name='dropout_keep_probability')
    self._dropout = tf.nn.dropout(self._reshape2, self._keep_prob)
    tf.summary.histogram('dropout', self._dropout)
    tf.summary.scalar('dropout_keep_probability', self._keep_prob)
    # layer 3 fully connected
    init_range = math.sqrt(6.0 / (dim + labels_shape[1]))
    self._weights3 = tf.Variable(tf.random_uniform([dim, labels_shape[1]], -init_range, init_range), name='FC1_weights')
    self.variable_summaries(self._weights3)
    self._biases3 = tf.Variable(tf.constant(np.ones(labels_shape[1], np.float32) * self._bias_init), name='FC1_bias')
    self.variable_summaries(self._biases3)
    self._local3 = tf.matmul(self._dropout, self._weights3) + self._biases3
    tf.summary.histogram('local3', self._local3)
    # softmax
    self._prediction = tf.nn.softmax(self._local3)
    tf.summary.histogram('prediction', self._prediction)
    # Cross entropy
    self._cross_entropy = tf.reduce_mean(
                            -tf.reduce_sum(
                              self._labels * tf.log(tf.clip_by_value(self._prediction, 1e-10, 1.0)),
                              reduction_indices=[1]))
    tf.summary.scalar('xentropy', self._cross_entropy)
    # training loss
    self._loss = tf.reduce_mean(self._cross_entropy)
    tf.summary.scalar('loss', self._loss)
    # accuracy
    self._true_class = tf.argmax(self._labels, 1)
    self._predicted_class = tf.argmax(self._prediction, 1)
    self._accuracy = tf.reduce_mean(tf.cast(tf.equal(self._predicted_class, self._true_class), tf.float32))
    tf.summary.histogram('accuracy', self._accuracy)

  def set_save_files(self, model_param_file, summary_dir=None):
    """ adds ops to save all variables. and merge_all op for summaries. """
    self._model_param_file = model_param_file
    self._saver = tf.train.Saver() # by default save all variables
    if summary_dir is not None:
      self._summary_dir = summary_dir
      self._summaries = tf.summary.merge_all()

  def create_session(self, learning_rate=0.001):
    """
    Prepares session

    Creates TF session,
    Creates optimizer op,
    Creates init op,
    Configures GPU usage
    Runs init op
    Configures summary writer

    Prev actions:
        define_model
        set_save_files
    Next actions:
        restore_variables
        train
        predict
        close_session
    """
    self._optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self._loss, global_step=self._global_step)
    # GPU config
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # session
    self._session = tf.Session(config=config)
    self._session.run(tf.global_variables_initializer())
    # summary writer
    if self._summary_dir is not None:
      self._summary_writer = tf.summary.FileWriter(self._summary_dir, self._session.graph)

  def restore_variables(self):
    """
    Restore model variables from file
    """
    if self._session is not None:
      ckpt = tf.train.get_checkpoint_state(os.path.dirname(self._model_param_file))
      # if that checkpoint exists, restore from checkpoint
      if ckpt and ckpt.model_checkpoint_path:
        self._saver.restore(self._session, ckpt.model_checkpoint_path)

  def close_session(self):
    if self._session is not None:
      self._session.close()
      self._session = None

  def train(self,
            train_images,
            train_labels,
            validation_images=None,
            validation_labels=None,
            dropout_keep_probability=0.5,
            batch_size=150,
            epochs=50,
            max_iterations_without_improvement=5):

    initial_step = 0

    # Measurements use for graphing loss and accuracy
    best_validation_accuracy = 0.0
    last_improvment_epoch = 0
    loss_epoch = []
    train_acc_epoch = []
    valid_acc_epoch = []

    st = datetime.now()

    a_ = 0

    initial_step = self._global_step.eval(session=self._session)
    for epoch_i in range(initial_step, initial_step+epochs):
      # train for one epoch
      # random training set permutation for each epoch
      perm_index = np.random.permutation(len(train_images))
      train_images = train_images[perm_index, :, :, :]
      train_labels = train_labels[perm_index]
      # running optimization in batches of training set
      batch_count = int(math.ceil(len(train_images) / batch_size))
      batches_pbar = tqdm(range(batch_count), desc='Train Epoch {:>2}/{}'.format(epoch_i + 1, epochs), unit='batches')
      for batch_i in batches_pbar:
        batch_start = batch_i * batch_size
        batch_images = train_images[batch_start:batch_start + batch_size]
        batch_labels = train_labels[batch_start:batch_start + batch_size]
        # Run optimizer and get loss
        _, l, _summ = self._session.run(
                          [self._optimizer, self._loss, self._summaries],
                          feed_dict={self._images: batch_images,
                                     self._labels: batch_labels,
                                     self._keep_prob: dropout_keep_probability})
      # write summaries once per epoch
#      _summ = self._session.run(
#          [self._summaries],
#          feed_dict={self._images: batch_images,
#                     self._labels: batch_labels,
#                     self._keep_prob: dropout_keep_probability})
      self._summary_writer.add_summary(_summ, global_step=epoch_i)

      # Log accuracy every epoch
      # training accuracy
      batch_count = int(math.ceil(len(train_images) / batch_size))
      batches_pbar = tqdm(range(batch_count), desc='Train Accuracy Epoch {:>2}/{}'.format(epoch_i + 1, epochs),
                          unit='batches')
      l = 0.
      a = 0.
      for batch_i in batches_pbar:
        batch_start = batch_i * batch_size
        batch_images = train_images[batch_start:batch_start + batch_size]
        batch_labels = train_labels[batch_start:batch_start + batch_size]
        l_, a_ = self._session.run(
                            [self._loss, self._accuracy],
                            feed_dict={self._images: batch_images,
                                       self._labels: batch_labels,
                                       self._keep_prob: 1.0})
        l += l_ * len(batch_images)
        a += a_ * len(batch_images)
      loss_epoch.append(l / len(train_images))
      train_acc_epoch.append(a / len(train_images))

      if validation_images is None:
        validation_images = train_images
        validation_labels = train_labels

      # validation accuracy
      batch_count = int(math.ceil(len(validation_images) / batch_size))
      batches_pbar = tqdm(range(batch_count), desc='Val Accuracy Epoch {:>2}/{}'.format(epoch_i + 1, epochs),
                          unit='batches')
      a = 0.
      for batch_i in batches_pbar:
        batch_start = batch_i * batch_size
        batch_images = validation_images[batch_start:batch_start + batch_size]
        batch_labels = validation_labels[batch_start:batch_start + batch_size]
        # Run optimizer and get loss
        a_ = self._session.run(
          [self._accuracy],
          feed_dict={self._images: batch_images,
                     self._labels: batch_labels,
                     self._keep_prob: 1.0})
        a += a_[0] * len(batch_images)
      validation_accuracy = a / len(validation_images)
      valid_acc_epoch.append(validation_accuracy)
      print('epoch {}, val accuracy: {}'.format(epoch_i, validation_accuracy))
      if (validation_accuracy > best_validation_accuracy):
        best_validation_accuracy = validation_accuracy
        last_improvment_epoch = epoch_i
        # save checkpoint every time accuracy improved during the epoch
        save_path = self._saver.save(self._session, self._model_param_file, global_step=self._global_step)
        print('validation accuracy improved')
        print("checkpoint saved to {}".format(save_path))
      else:
        if (epoch_i - last_improvment_epoch >= max_iterations_without_improvement):
          print('no validation accuracy improvement over {} epochs. stop'.format(max_iterations_without_improvement))
          break  # stop learning
    print('runtime: {}'.format(datetime.now() - st))
    print('best val accuracy: {}'.format(best_validation_accuracy))
    print('epochs: {}'.format(epoch_i))
    return loss_epoch, train_acc_epoch, valid_acc_epoch, best_validation_accuracy

  def predict(self,
              images,
              true_labels=None,
              batch_size=150):

    # Measurements use for graphing loss and accuracy
    predicted_probabilities = []
    predicted_classes = []
    accuracy = 0
    batch_count = int(math.ceil(len(images) / batch_size))
    for batch_i in range(batch_count):
      batch_start = batch_i * batch_size
      ops = [self._prediction, self._predicted_class]
      feed_dict = {self._images: images[batch_start:batch_start + batch_size],
                   self._keep_prob: 1.0}
      if true_labels is not None:
        ops.append(self._accuracy)
        feed_dict[self._labels] = true_labels[batch_start:batch_start + batch_size]
      # Run optimizer and get loss
      output = self._session.run(ops, feed_dict=feed_dict)
      predicted_probabilities.append(output[0])
      predicted_classes.append(output[1])
      accuracy += output[2] * len(feed_dict[self._images])
    accuracy /= len(images)
    return np.vstack(predicted_probabilities), np.hstack(predicted_classes), accuracy
