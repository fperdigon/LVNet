#!/usr/bin/env python

#########################################################################################
# Copyright (c) 2017 Liset Vazquez Romaguera, Francisco Perdigon Romero
# Authors: Francisco Perdigon Romero
#          Liset Vazquez Romaguera
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from scipy import misc
import numpy as np
import matplotlib.pylab as plt
import sys, os

import tensorflow as tf


FLAGS = None


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.02)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def deconv2d(x, W):
  return tf.nn.conv2d_transpose(x, W, output_shape=[1, 256, 256, 2],  strides=[1, 8, 8, 1], padding= "SAME")

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def getData(PathImages, PathLabels):
    img_files_list = []
    lab_files_list = []

    # get the files names and sort
    for file in os.listdir(PathImages):
        if file[-3:] == 'png' or file[-3:] == 'PNG':
            img_files_list.append(PathImages + file)

    for file in os.listdir(PathLabels):
        if file[-3:] == 'png' or file[-3:] == 'PNG':
            lab_files_list.append(PathLabels + file)

    img_files_list.sort()
    lab_files_list.sort()

    return [img_files_list, lab_files_list]

# Funtions to get a bilinear weights for deconvolution
def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - factor % 2

def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

def bilinear_upsample_weights(factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """

    filter_size = get_kernel_size(factor)

    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)

    upsample_kernel = upsample_filt(filter_size)

    for i in xrange(number_of_classes):
        weights[:, :, i, i] = upsample_kernel

    return weights

def main(_):
  # Import data
  LABELS_DIR = "./data/labels/"
  IMAGES_DIR = "./data/mri/"

  [ImagesList, LabelsList] = getData(IMAGES_DIR, LABELS_DIR)
  Images_train = ImagesList[0:200]
  Images_test = ImagesList[201:-1]
  Labels_train = LabelsList[0:200]
  Labels_test = LabelsList[201:-1]

  # Prepare Data input

  image_filename_placeholder = tf.placeholder(tf.string)
  annotation_filename_placeholder = tf.placeholder(tf.string)
  is_training_placeholder = tf.placeholder(tf.bool)

  image_tensor = tf.read_file(image_filename_placeholder)
  annotation_tensor = tf.read_file(annotation_filename_placeholder)

  image_tensor = tf.reshape(tf.to_float(tf.image.decode_png(image_tensor, channels=1)), [1, 256, 256, 1])
  annotation_tensor = tf.reshape(tf.image.decode_png(annotation_tensor, channels=1), [1, 256, 256, 1])

  # Get ones for each class instead of a number -- we need that
  # for cross-entropy loss later on. Sometimes the groundtruth
  # masks have values other than 1 and 0.
  class_labels_tensor = tf.equal(annotation_tensor, 1)
  background_labels_tensor = tf.not_equal(annotation_tensor, 1)

  # Convert the boolean values into floats -- so that
  # computations in cross-entropy loss is correct
  bit_mask_class = tf.to_float(class_labels_tensor)
  bit_mask_background = tf.to_float(background_labels_tensor)

  #combined_mask = tf.reshape(tf.concat(concat_dim=2, values=[bit_mask_class, bit_mask_background]), [1, 256, 256, 2])
  combined_mask = tf.reshape(tf.concat(axis=2, values=[bit_mask_class, bit_mask_background]), [1, 256, 256, 2])

  # Create the model

  # First Convolutional Layer
  W_conv1 = weight_variable([3, 3, 1, 64])
  b_conv1 = bias_variable([64])
  h_conv1 = tf.nn.relu(conv2d(image_tensor, W_conv1) + b_conv1)

  # Second Convolutional Layer
  W_conv2 = weight_variable([3, 3, 64, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

  h_pool1 = max_pool_2x2(h_conv2)

  # Third Convolutional Layer
  W_conv3 = weight_variable([3, 3, 64, 128])
  b_conv3 = bias_variable([128])
  h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)

  # Fourth Convolutional Layer
  W_conv4 = weight_variable([3, 3, 128, 128])
  b_conv4 = bias_variable([128])
  h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

  h_pool2 = max_pool_2x2(h_conv4)

  # Fifth Convolutional Layer
  W_conv5 = weight_variable([3, 3, 128, 256])
  b_conv5 = bias_variable([256])
  h_conv5 = tf.nn.relu(conv2d(h_pool2, W_conv5) + b_conv5)

  # Sixth Convolutional Layer
  W_conv6 = weight_variable([3, 3, 256, 256])
  b_conv6 = bias_variable([256])
  h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)

  h_pool3 = max_pool_2x2(h_conv6)

  # Seventh Convolutional Layer
  W_conv7 = weight_variable([7, 7, 256, 512])
  b_conv7 = bias_variable([512])
  h_conv7 = tf.nn.relu(conv2d(h_pool3, W_conv7) + b_conv7)

  # Dropout 1
  keep_prob = tf.placeholder(tf.float32)
  h_drop1 = tf.nn.dropout(h_conv7, keep_prob)

  # Eighth Convolutional Layer
  W_conv8 = weight_variable([1, 1, 512, 512])
  b_conv8 = bias_variable([512])
  h_conv8 = tf.nn.relu(conv2d(h_drop1, W_conv8) + b_conv8)

  # Dropout 2
  h_drop2 = tf.nn.dropout(h_conv8, keep_prob)

  # Ninth Convolutional Layer
  W_conv9 = weight_variable([1, 1, 512, 2])
  b_conv9 = bias_variable([2])
  h_conv9 = tf.nn.relu(conv2d(h_drop2, W_conv9) + b_conv9)

  # Deconvolution
  # Can be used the bilinear upsample (fixed, no learning)
  # or learning upsample
  W_deconv1 = bilinear_upsample_weights(8, 2)
  #W_deconv1 = weight_variable([16, 16, 2, 2])
  b_deconv1 = bias_variable([2])
  logist = deconv2d(h_conv9, W_deconv1) + b_deconv1

  # Tensor to get the final prediction for each pixel -- pay
  # attention that we don't need softmax in this case because
  # we only need the final decision. If we also need the respective
  # probabilities we will have to apply softmax.
  prediction = tf.argmax(logist, dimension=3)
  probabilities = tf.nn.softmax(logist, dim=-1)

  # Define loss and optimizer
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=combined_mask, logits=logist))

  # Learning rate (lr) variable as place holder is usefull
  # when lr need to be changed during the train
  lr = tf.placeholder(tf.float32)
  # Optimization algoritms
  # train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
  train_step = tf.train.AdamOptimizer(lr).minimize(loss)

  # Define TF Session
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  # Restore a training
  # Add ops to save and restore all the variables.
  saver = tf.train.Saver()
  # Restore a training
  #saver.restore(sess, "./LVNet.ckpt-40000")

  # Train
  c = 0
  lr_t = 1e-4

  for i in range(100001):

      # Monitoring training loss
      if c % 100 == 0:
          train_loss = loss.eval(feed_dict={image_filename_placeholder: Images_train[c],
                                            annotation_filename_placeholder: Labels_train[c],
                                            keep_prob: 1.0, lr: lr_t})
          print("step %d, training loss %g" % (i, train_loss))

      # Save Training
      if i % 5000 == 0:
          # Save the variables to disk.
          save_path = saver.save(sess, "./LVNet.ckpt", global_step=i)
          print("Model saved in file: %s" % save_path)

      train_step.run(feed_dict={image_filename_placeholder: Images_train[c], annotation_filename_placeholder: Labels_train[c], keep_prob: 0.5, lr: lr_t})

      c = c + 1
      if c >= len(Images_train) :
          c = 0


  # Test trained model

  for i in range(len(Images_test)):
      # Show Input image
      image_t = image_tensor.eval(feed_dict={image_filename_placeholder: Images_test[i],
                                        annotation_filename_placeholder: Labels_test[i],
                                        keep_prob: 1.0, lr: lr_t})
      image_t = np.asarray(image_t[0, :, :, 0])
      plt.imshow(image_t)
      plt.show()

      # Show Label image
      label_t = annotation_tensor.eval(feed_dict={image_filename_placeholder: Images_test[i],
                                                  annotation_filename_placeholder: Labels_test[i],
                                                  keep_prob: 1.0, lr: lr_t})
      label_t = np.asarray(label_t[0, :, :, 0])
      plt.imshow(label_t)
      plt.show()

      # Show Predicted segmentation
      prediction_t = prediction.eval(feed_dict={image_filename_placeholder: Images_test[i],
                                                annotation_filename_placeholder: Labels_test[i],
                                                keep_prob: 1.0, lr: lr_t})

      prediction_t = np.asarray(prediction_t[0, :, :])
      plt.imshow(prediction_t)
      plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='./data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)