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

from tf_helpers import *

class TLLabelConverter:
  def __init__(self):
    self._relevant_labels = ['off','green','yellow','red']
    self._n_classes = len(self._relevant_labels)
    self._label_to_i_dict = {}
    self._i_to_label_dict = {}
    self._label_to_oh_dict = {}
    for i, l in enumerate(self._relevant_labels):
      self._label_to_i_dict[l] = i
      self._i_to_label_dict[i] = l
      self._label_to_oh_dict[l] = [1 if x==i else 0 for x in range(self._n_classes)]

  def labels(self):
    return  self._relevant_labels

  def get_i(self, l):
    return self._label_to_i_dict[l]

  def get_l(self, i):
    return self._i_to_label_dict[i]

  def get_oh(self, l):
    return self._label_to_oh_dict[l]

  def filter(self, images, labels):
    x = images[np.isin(labels, self._relevant_labels)]
    y = labels[np.isin(labels, self._relevant_labels)]
    return x, y

  def convert_to_oh(self, labels):
    return np.array([self._label_to_oh_dict[l] for l in labels])

  def convert_to_labels(self, classes):
    return np.array([self._i_to_label_dict[i] for i in classes])

  def get_shape(self):
    return (self._n_classes,)

class TLClassifierCNN:

  #@define_scope(scope='data')
  def _create_inputs(self):
    """ defines input placeholders """
    with tf.name_scope("data"):
      self._images = tf.placeholder(tf.uint8, name='images', shape=self._features_shape)
      tf.summary.image('input_images', self._images, 3)
      self._labels = tf.placeholder(tf.uint8, name='labels', shape=self._labels_shape)

  #@define_scope(scope='pre_processing')
  def _create_input_transforms(self):
    """ ops to pre-process inputs. convert type and standardise image to [0,1] values """
    with tf.name_scope("pre_processing"):
      self._images_float = tf.image.convert_image_dtype(self._images, tf.float32)
      self._images_std = tf.map_fn(lambda img: tf.image.per_image_standardization(img), self._images_float)
      self._labels_float = tf.cast(self._labels, tf.float32)

  def _conv2d(self, input_op, input_channels, output_channels, kernel_size, stride_size):
    """ define conv+bias ops. plus all summaries"""
    trunc_normal_stddev = 0.05
    bias_init = 0.1
    strides = [1, stride_size, stride_size, 1]
    params = [kernel_size, kernel_size, input_channels, output_channels]
    weights = tf.Variable(tf.truncated_normal(params, stddev=trunc_normal_stddev), name='weights')
    #self.variable_summaries(weights)
    tf.summary.histogram("weights", weights)
    conv = tf.nn.conv2d(input_op, weights, strides=strides, padding='SAME')
    #tf.summary.histogram('conv', conv)
    biases = tf.Variable(tf.constant(np.ones(output_channels, np.float32) * bias_init), name='biases')
    #self.variable_summaries(biases)
    tf.summary.histogram("biases", biases)
    result = tf.nn.bias_add(conv, biases)
    return result

  #@define_scope(scope='conv1')
  def _create_layer_1(self):
    """ define first convolutional layer """
    with tf.name_scope("conv1"):
      kernel_size = 5
      input_channels = self._features_shape[3]
      output_channels = 64
      stride_size = 1
      # conv
      conv = self._conv2d(self._images_std, input_channels, output_channels, kernel_size, stride_size)
      # relu
      activations = tf.nn.relu(conv)
      tf.summary.histogram('activations', activations)
      # pool
      pool_size = 2
      params = [1, pool_size, pool_size, 1]
      pooling = tf.nn.max_pool(activations, ksize=params, strides=params, padding='SAME')
      #tf.summary.histogram('pooling', pooling)
    return pooling

  #@define_scope(scope='conv2')
  def _create_layer_2(self):
    """ define second convolutional layer """
    with tf.name_scope("conv2"):
      kernel_size = 5
      input_channels = 64
      output_channels = 32
      stride_size = 1
      # conv
      conv = self._conv2d(self._conv1, input_channels, output_channels, kernel_size, stride_size)
      # relu
      activations = tf.nn.relu(conv)
      tf.summary.histogram('activations', activations)
      # pool
      pool_size = 2
      params = [1, pool_size, pool_size, 1]
      pooling = tf.nn.max_pool(activations, ksize=params, strides=params, padding='SAME')
      #tf.summary.histogram('pooling', pooling)
    return pooling

  #@define_scope(scope='fc')
  def _create_layer_3(self):
    """ define final fully connected layer """
    with tf.name_scope("fc"):
      # reshape
      input_op = self._conv2
      shape = input_op.get_shape().as_list()
      dim = np.prod(shape[1:])
      reshape = tf.reshape(input_op, [-1, dim])
      # dropout
      dropout = tf.nn.dropout(reshape, self._keep_prob)
      tf.summary.histogram('dropout', dropout)
      # fc
      output_dim = self._labels_shape[1]
      init_range = math.sqrt(float(6.0) / (dim + output_dim)) # Xavier init
      weights = tf.Variable(tf.random_uniform([dim, output_dim], -init_range, init_range), name='weights')
      tf.summary.histogram('weights', weights)
      #self.variable_summaries(weights)
      bias_init = 0.1
      biases = tf.Variable(tf.ones(output_dim, np.float32) * bias_init, name='biases')
      tf.summary.histogram('biases', biases)
      #self.variable_summaries(biases)
      activations = tf.matmul(dropout, weights) + biases
      tf.summary.histogram('activations', activations)
    return activations

  #@define_scope(scope='predictions')
  def _create_predictions(self):
    with tf.name_scope("predictions"):
      self._prediction_softmax = tf.nn.softmax(self._logits, name="prediction_softmax")
      tf.summary.histogram('prediction_softmax', self._prediction_softmax)
      self._prediction_class = tf.argmax(self._prediction_softmax, 1, name="prediction_class")
      tf.summary.histogram('prediction_class', self._prediction_class)

  #@define_scope(scope='loss')
  def _create_loss(self):
    with tf.name_scope("loss"):
      cross_entropy = tf.negative(tf.reduce_sum(self._labels_float * tf.log(self._prediction_softmax),
                                                reduction_indices=[1]),
                                  name="cross_entropy")
      self._loss = tf.reduce_mean(cross_entropy, name='loss')
      tf.summary.scalar('loss', self._loss)

  #@define_scope(scope='accuracy')
  def _create_accuracy(self):
    with tf.name_scope("accuracy"):
      true_class = tf.argmax(self._labels_float, 1)
      correct_prediction = tf.equal(true_class, self._prediction_class)
      self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      tf.summary.scalar('accuracy', self._accuracy)

  #@define_scope(scope='optimizer')
  def _create_optimizer(self):
    with tf.name_scope("optimizer"):
      self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
      if self._learning_rate is not None:
        self._optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self._loss, global_step=self._global_step)
      else:
        self._optimizer = tf.train.AdamOptimizer().minimize(self._loss, global_step=self._global_step)

  def _create_session(self, gpu_mem_fraction=0.9):
    # GPU config
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = gpu_mem_fraction
    # session
    self._session = tf.Session(config=config)

  def __init__(self, image_shape=None, learning_rate=None):
    """
    Create calculation graph

    Takes batch of OpenCV images of type uint8.
    Does per-image normalization internally.
    Defines CNN.
    Defines summaries.

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
    # label converter
    self._label_converter = TLLabelConverter()
    # reset graph
    tf.reset_default_graph()
    self._tag = 'tl_classifier'
    # inputs
    self._features_shape = (None,)+image_shape
    self._labels_shape = (None, len(self._label_converter.labels()))
    # model
    self._create_inputs()
    self._create_input_transforms()
    self._conv1 = self._create_layer_1()
    self._conv2 = self._create_layer_2()
    self._keep_prob = tf.placeholder(tf.float32, name='dropout_keep_probability')
    self._logits = self._create_layer_3()
    self._create_predictions()
    self._create_loss()
    self._create_accuracy()
    self._learning_rate = learning_rate
    self._create_optimizer()
    # auxiliary objects
    self._summaries = tf.summary.merge_all()
    # session
    self._create_session(0.9) # gpu mem fraction to use
    self._session.run(tf.global_variables_initializer())

  def save_model(self, model_dir):
    if self._session is not None:
      print('saving SavedModel into {}'.format(model_dir))
      builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
      builder.add_meta_graph_and_variables(self._session, [self._tag])
      builder.save()

  def load_model(self, model_dir):
    if self._session is not None:
      self._session.close()
      self._session = None
    tf.reset_default_graph()
    self._create_session(0.9)
    tf.saved_model.loader.load(self._session, [self._tag], model_dir)
    # the ops we need to re-assign to instance variables for prediction
    graph = tf.get_default_graph()
    self._images = graph.get_tensor_by_name("data/images:0")
    self._keep_prob = graph.get_tensor_by_name("dropout_keep_probability:0")
    self._prediction_softmax = graph.get_tensor_by_name("predictions/prediction_softmax:0")
    self._prediction_class = graph.get_tensor_by_name("predictions/prediction_class:0")

  def restore_checkpoint(self, checkpoint_dir):
    if self._session is not None:
      ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_dir))
      # if that checkpoint exists, restore from checkpoint
      if ckpt and ckpt.model_checkpoint_path:
        saver = tf.train.Saver()
        saver.restore(self._session, ckpt.model_checkpoint_path)

  def save_checkpoint(self, checkpoint_dir):
    if self._session is not None:
      saver = tf.train.Saver() # by default saves all variables
      if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
      save_path = saver.save(self._session, checkpoint_dir, global_step=self._global_step)
      return save_path

  def close_session(self):
    if self._session is not None:
      self._session.close()
      self._session = None

  def train(self,
            train_images,
            train_labels_str,
            validation_images=None,
            validation_labels_str=None,
            dropout_keep_probability=0.5,
            batch_size=150,
            epochs=50,
            max_iterations_without_improvement=5,
            checkpoint_dir=None,
            summary_dir=None):
    """
    Run training for specified number of epochs in batches of specified size
    Every time accuracy on validation set improves, save weights
    """
    best_validation_accuracy = 0.0
    last_improvement_epoch = 0
    start_time = datetime.now()

    # convert string labels to one-hot-encoded labels
    train_labels = self._label_converter.convert_to_oh(train_labels_str)
    if validation_labels_str is not None:
      validation_labels = self._label_converter.convert_to_oh(validation_labels_str)
    else:
      validation_labels = None

    if summary_dir is not None:
      summary_writer = tf.summary.FileWriter(summary_dir, graph=tf.get_default_graph())

    ####################
    # code to visualize the embeddings. uncomment the below to visualize embeddings
    # final_embed_matrix = sess.run(model.embed_matrix)

    # # it has to variable. constants don't work here. you can't reuse model.embed_matrix
    # embedding_var = tf.Variable(final_embed_matrix[:1000], name='embedding')
    # sess.run(embedding_var.initializer)

    # config = projector.ProjectorConfig()
    # summary_writer = tf.summary.FileWriter('processed')

    # # add embedding to the config file
    # embedding = config.embeddings.add()
    # embedding.tensor_name = embedding_var.name

    # # link this tensor to its metadata file, in this case the first 500 words of vocab
    # embedding.metadata_path = 'processed/vocab_1000.tsv'

    # # saves a configuration file that TensorBoard will read during startup.
    # projector.visualize_embeddings(summary_writer, config)
    # saver_embed = tf.train.Saver([embedding_var])
    # saver_embed.save(sess, 'processed/model3.ckpt', 1)

    step = self._global_step.eval(session=self._session)
    if step>0:
      print("continuing training after {} steps done previously".format(step))
    for epoch_i in range(epochs):
      # train for one epoch
      # random permutation of training set
      n_samples = len(train_images)
      perm_index = np.random.permutation(n_samples)
      train_images = train_images[perm_index, :, :, :]
      train_labels = train_labels[perm_index]
      # running optimization in batches of training set
      n_batches = int(math.ceil(float(n_samples) / batch_size))
      batches_pbar = tqdm(range(n_batches), desc='Train Epoch {:>2}/{}'.format(epoch_i + 1, epochs), unit='batches')
      for batch_i in batches_pbar:
        batch_start = batch_i * batch_size
        batch_images = train_images[batch_start:batch_start + batch_size]
        batch_labels = train_labels[batch_start:batch_start + batch_size]
        _, loss, summaries = self._session.run([self._optimizer, self._loss, self._summaries],
                                                feed_dict={self._images: batch_images,
                                                           self._labels: batch_labels,
                                                           self._keep_prob: dropout_keep_probability})
        # write training summaries every so often
        step = self._global_step.eval(session=self._session)
        if step % 5 == 0:
          summary_writer.add_summary(summaries, global_step=step)

      if validation_images is None:
        # use training data to measure validation accuracy
        validation_images = train_images
        validation_labels = train_labels

      # validation accuracy
      n_batches = int(math.ceil(float(len(validation_images)) / batch_size))
      batches_pbar = tqdm(range(n_batches), desc='Accuracy Epoch {:>2}/{}'.format(epoch_i + 1, epochs), unit='batches')
      a = 0.
      l = 0.
      for batch_i in batches_pbar:
        batch_start = batch_i * batch_size
        batch_images = validation_images[batch_start:batch_start + batch_size]
        batch_labels = validation_labels[batch_start:batch_start + batch_size]
        a_, l_ = self._session.run([self._accuracy, self._loss],
                               feed_dict={self._images: batch_images,
                                          self._labels: batch_labels,
                                          self._keep_prob: 1.0})
        a += float(a_) * len(batch_images)
        l += float(l_) * len(batch_images)
      validation_accuracy = float(a) / len(validation_images)
      validation_loss = float(l) / len(validation_images)
      print('epoch {}: val accuracy {}, val loss {}'.format(epoch_i, validation_accuracy, validation_loss))
      if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        last_improvement_epoch = epoch_i
        # save checkpoint every time accuracy improved during the epoch
        if checkpoint_dir is not None:
          save_path = self.save_checkpoint(checkpoint_dir)
          print('validation accuracy improved')
          print("checkpoint saved to {}".format(save_path))
      else:
        if epoch_i - last_improvement_epoch >= max_iterations_without_improvement:
          print('no validation accuracy improvement over {} epochs. stop'.format(max_iterations_without_improvement))
          break  # stop learning
    print('epochs trained: {}'.format(epoch_i))
    print('total runtime: {}'.format(datetime.now() - start_time))
    print('best val accuracy: {}'.format(best_validation_accuracy))
    return best_validation_accuracy

  def predict(self,
              images,
              batch_size=150):
    """
    Predict labels for batch of images

    :param images:
    :param true_labels:
    :param batch_size:
    :return:
    """
    predicted_probabilities = []
    predicted_classes = []
    n_batches = int(math.ceil(float(len(images)) / batch_size))
    for batch_i in range(n_batches):
      batch_start = batch_i * batch_size
      ops = [self._prediction_softmax, self._prediction_class]
      feed_dict = {self._images: images[batch_start:batch_start + batch_size],
                   self._keep_prob: 1.0}
      output = self._session.run(ops, feed_dict=feed_dict)
      predicted_probabilities.append(output[0])
      predicted_classes.append(output[1])
    predicted_probabilities = np.vstack(predicted_probabilities)
    predicted_classes = np.hstack(predicted_classes)
    predicted_labels = self._label_converter.convert_to_labels(predicted_classes)
    return predicted_labels, predicted_probabilities
