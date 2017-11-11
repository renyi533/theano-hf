# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
from hessian_free import HessianFreeOptimizer
from dc_asgd_optimizer import DCAsgdOptimizer

FLAGS = None


def main(_):
  # Import data

  # Create the model
  x = tf.placeholder(tf.float32, [None, 2])
  #W = tf.Variable(tf.random_normal([2, 2], stddev=0.1))
  #b = tf.Variable(tf.random_normal([2], stddev=0.1))
  W = tf.Variable(tf.constant([ [0.05, 0.06], [0.05, 0.06] ]))
  b = tf.Variable(tf.constant([0.1, 0.11]))
  y = tf.matmul(x, W) + b
  for k in range(FLAGS.layer):
    y_a = tf.nn.tanh(y)
    W2 = tf.Variable(tf.constant([ [0.05, 0.06], [0.06, 0.05] ]))
    b2 = tf.Variable(tf.constant([0.1, 0.11]))
    y = tf.matmul(y_a, W2) + b2

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 2])

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  tf.summary.scalar('cross_entropy', cross_entropy)
  if FLAGS.method == 'GradientDescent':
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    print('GradientDescent')
  elif FLAGS.method == 'Adagrad':
    optimizer = tf.train.AdagradOptimizer(FLAGS.learning_rate)
    print('Adagrad')
  else:
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    print('Adam')

  sess = tf.InteractiveSession()
  global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
  #tf.train.global_step(sess, global_step_tensor)
  optimizer = DCAsgdOptimizer(optimizer, lambda_val=0.01, ps_comp=True, rescale_variance = False, momentum = 0.0, global_step=global_step_tensor)
  train_step = optimizer.minimize(cross_entropy, global_step=global_step_tensor)

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter('./train', sess.graph)
  tf.global_variables_initializer().run()
  tf.local_variables_initializer().run()
  batch_xs = [ [0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
  batch_ys = [ [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
  # Train
  for i in range(10000):
    _, loss, accu, summary = sess.run([train_step, cross_entropy, accuracy, merged], feed_dict={x: batch_xs, y_: batch_ys})
    train_writer.add_summary(summary, i)
    if i % 50 == 0:
      print('batch: %d. loss: %f, accuracy: %f' % (i, loss, accu))

    if loss < 1e-3:
      print('converge in step: %d' % (i))
      break

  # Test trained model
  print(sess.run([accuracy, cross_entropy], feed_dict={x: batch_xs,
                                      y_: batch_ys}))
  train_writer.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--method', type=str, default='HessianFree',
                      help='Directory for storing input data')
  parser.add_argument('--init_decay', type=float, default=0.0,
                      help='Directory for storing input data')
  parser.add_argument('--hv_method', type=int, default=1,
                      help='Directory for storing input data')
  parser.add_argument('--layer', type=int, default=1,
                      help='Directory for storing input data')
  parser.add_argument('--cg_iter', type=int, default=2,
                      help='Directory for storing input data')
  parser.add_argument('--damping', type=float, default=0.012,
                      help='Directory for storing input data')
  parser.add_argument('--learning_rate', type=float, default=1.0,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

