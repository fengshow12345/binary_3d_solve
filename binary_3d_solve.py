# Copyright (c) 2018 Matthew J. Hergott. All Rights Reserved.
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


import os
os.environ['PYTHONHASHSEED'] = '0'

from tensorflow import set_random_seed as tf_set_random_seed
tf_set_random_seed(123)

import numpy as np
np.random.seed(42)

import random
random.seed(7)

# Can turn off GPU on CPU-only machines; maybe results in faster startup.
use_GPU = True
if use_GPU is False:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import math

from pathlib import Path
from datetime import datetime
from functools import partial

from enum import Enum


# Enumeration to select neural network activations.
#
class ActivationType(Enum):
    TANH = 1
    RELU = 2
    ELU = 3


# Makes it easier to call directory for TensorBoard saves.
#
class LogDirMetaClass(type):
    @property
    def logdir(self):
        return self._logdir


class LogDir(metaclass=LogDirMetaClass):
    _logdir = ''

    @classmethod
    def reset_logdir(cls):
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        root_logdir = "../tensorflow_logs/"
        cls._logdir = f"{root_logdir}/run-{now}/"


# This is a class structure that can be used to move TensorFlow references
# between functions. Doing this avoids global variables and the so-called
# "self. hell" of class instance variables.
#
class TFGraphRefs:
    def __init__(self, x_in, y_in, training_op, training_flag, loss_op,
                 base_loss_op, prediction_op, accuracy_op, file_writer,
                 loss_summary, accuracy_summary):
        self.x_in = x_in
        self.y_in = y_in
        self.training_op = training_op
        self.training_flag = training_flag
        self.loss_op = loss_op
        self.base_loss_op = base_loss_op
        self.prediction_op = prediction_op
        self.accuracy_op = accuracy_op
        self.file_writer = file_writer
        self.loss_summary = loss_summary
        self.accuracy_summary = accuracy_summary


# Simple way to make batches; randomly shuffled every epoch.
def make_batch_list(n_obs, batch_size):
    index_range = np.arange(0, n_obs)
    np.random.shuffle(index_range)

    index_list = list()
    if n_obs <= batch_size:
        index_list.append(index_range)
    else:
        for i in range(math.ceil(n_obs / batch_size)):
            start = i * batch_size
            stop = (i + 1) * batch_size
            if stop > n_obs:
                index_list.append(index_range[start:])
            else:
                index_list.append(index_range[start:stop])

    return index_list


# Trains the neural network and outputs to TensorBoard log directory.
# Optional 'save' flag will save trained model.
#
def train_nn(refs, x_train, y_train, x_val, y_val, model_name, n_epochs=100,
             save=True, batch_size=256):
    init = tf.global_variables_initializer()

    if save:
        saver = tf.train.Saver()

    with tf.Session() as sess:
        init.run()

        for epoch in range(n_epochs):

            # Gets list of index arrays, one per batch.
            batch_index_list = make_batch_list(x_train.shape[0], batch_size)

            for idx in batch_index_list:
                sess.run(refs.training_op, feed_dict={
                    refs.x_in: x_train[idx], refs.y_in: y_train[idx],
                    refs.training_flag: True
                    })

            last_epoch = True if epoch == (n_epochs - 1) else False

            if epoch % 100 == 0 or last_epoch:
                acc_train, base_loss, total_loss = sess.run(
                        [refs.accuracy_op, refs.base_loss_op, refs.loss_op],
                        feed_dict={
                            refs.x_in: x_train, refs.y_in: y_train,
                            refs.training_flag: False
                            })

                acc_val = sess.run([refs.accuracy_op], feed_dict={
                    refs.x_in: x_val, refs.y_in: y_val,
                    refs.training_flag: False
                    })
                acc_val = acc_val[0]

                print(f'epoch {epoch}  '
                      f'base loss: {"{:.6f}".format(base_loss)}  '
                      f'total loss: {"{:.6f}".format(total_loss)}  '
                      f'train acc: {"{:.4f}".format(acc_train)}   '
                      f'validation acc: {"{:.4f}".format(acc_val)}')

                merge = tf.summary.merge_all()

                summary = sess.run(merge, feed_dict={
                    refs.x_in: x_train, refs.y_in: y_train
                    })

                refs.file_writer.add_summary(summary, epoch)

        refs.file_writer.close()

        if save:
            saver.save(sess, f'../model_saves/{model_name}')

    return acc_train, acc_val


# Make TensorFlow graph and return references to key operations in
# a class structure.
#
def create_nn_graph(T, nodes_per_layer, layers, dropout_rate=0.5,
                    l2_reg_scale=0.01, batch_norm_momentum=0.9,
                    activation_type=ActivationType.RELU):
    # Clear TensorFlow graph.
    tf.reset_default_graph()

    # Code for learning rate scheduler is included, but not necessary with
    # all optimizers.
    include_learning_rate_scheduler = False

    # Set activation function and appropriate weight initializer.
    if activation_type == ActivationType.TANH:
        activation_layer = tf.nn.tanh
        nn_init = tf.random_normal_initializer()
    elif activation_type == ActivationType.RELU:
        activation_layer = tf.nn.relu
        nn_init = tf.variance_scaling_initializer(scale=2.0, mode='fan_in',
                                                  distribution='normal')
    elif activation_type == ActivationType.ELU:
        activation_layer = tf.nn.elu
        nn_init = tf.variance_scaling_initializer(scale=2.0, mode='fan_in',
                                                  distribution='normal')

    # Add optional L2 regularization.
    if l2_reg_scale > 0.:
        regularization = True
        l2_reg = tf.contrib.layers.l2_regularizer(l2_reg_scale)
    else:
        regularization = False
        l2_reg = None

    with tf.name_scope("inputs"):
        x_in = tf.placeholder(tf.float32, shape=(None, T), name='X')
        y_in = tf.placeholder(tf.float32, shape=(None, 1), name='Y')

    # Training flag: default is false. Turn on at training time.
    with tf.name_scope("constants"):
        training_flag = tf.placeholder_with_default(False, shape=(),
                                                    name='training_flag')

    # Make neural network layers.
    with tf.name_scope("dnn"):
        dense_layer = partial(tf.layers.dense, kernel_initializer=nn_init,
                              kernel_regularizer=l2_reg)
        batch_norm_layer = partial(tf.layers.batch_normalization,
                                   training=training_flag,
                                   momentum=batch_norm_momentum)
        dropout_layer = partial(tf.layers.dropout, rate=dropout_rate,
                                training=training_flag)

        dense = list()
        activation = list()
        batch_norm = list()
        batch_norm_dropout = list()

        for lyr in range(layers):
            dense_in = dropout_layer(x_in) if lyr == 0 else batch_norm_dropout[-1]

            d = dense_layer(dense_in, nodes_per_layer, name=f'dense_{lyr}')
            dense.append(d)

            activation.append(activation_layer(dense[-1]))

            batch_norm.append(batch_norm_layer(activation[-1]))

            batch_norm_dropout.append(dropout_layer(batch_norm[-1]))

        # Output layer.
        prediction_logit_op = tf.layers.dense(batch_norm_dropout[-1], units=1,
                                              activation=None,
                                              kernel_initializer=nn_init,
                                              kernel_regularizer=l2_reg,
                                              name='logit_out')
        prediction_sigmoid_op = tf.nn.sigmoid(prediction_logit_op,
                                              name='sigmoid_out')
        prediction_op = tf.cast(tf.greater(prediction_sigmoid_op, 0.5), 'float',
                                name='prediction')

    # Loss calculations.
    with tf.name_scope("loss"):
        base_loss_all = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=prediction_logit_op, labels=y_in)
        base_loss_op = tf.reduce_mean(base_loss_all, name='loss')

        if regularization:
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss_op = tf.add_n([base_loss_op] + reg_losses, name='reg_loss')
        else:
            loss_op = base_loss_op

    # Select optimizer.
    with tf.name_scope("train"):
        # Learning rate scheduling might not be necessary for RMSProp, Adam,
        # and AdaGrad optimizers.
        if include_learning_rate_scheduler:
            initial_learning_rate = 0.1
            decay_steps = 10000
            decay_rate = 1 / 10
            global_step = tf.Variable(0, trainable=False, name="global_step")
            learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                                       global_step, decay_steps,
                                                       decay_rate)

        # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate momentum=0.9,
        #                                       decay=0.9, epsilon=1e-10)
        optimizer = tf.train.AdamOptimizer()

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            if include_learning_rate_scheduler:
                training_op = optimizer.minimize(loss_op,
                                                 global_step=global_step)
            else:
                training_op = optimizer.minimize(loss_op)

    # Accuracy calculations.
    with tf.name_scope("eval"):
        correct_op = tf.cast(tf.equal(prediction_op, y_in), 'float')
        accuracy_op = tf.reduce_mean(correct_op)

    # This is for TensorBoard.
    loss_summary = tf.summary.scalar('TotalLoss', loss_op)
    accuracy_summary = tf.summary.scalar('accuracy', accuracy_op)

    LogDir.reset_logdir()
    file_writer = tf.summary.FileWriter(LogDir.logdir, tf.get_default_graph())

    # Count trainable parameters in graph.
    n_trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in
                                 tf.trainable_variables()])

    # Use this class structure to transport key operations and
    # variables--avoids global variables.
    out = TFGraphRefs(x_in=x_in, y_in=y_in, training_op=training_op,
                      training_flag=training_flag, loss_op=loss_op,
                      base_loss_op=base_loss_op, prediction_op=prediction_op,
                      accuracy_op=accuracy_op, file_writer=file_writer,
                      loss_summary=loss_summary,
                      accuracy_summary=accuracy_summary)

    return out, n_trainable_params


# Make data for the riddle to be solved here.
#
# There are 3 feature indices. All values are between 0 and 1.
#
# The label is TRUE if 1 or 3 of the features is greater than 0.5.
# The label is FALSE if 0 or 2 of the features is greater than 0.5.
#
# This can be thought of as a hypothetical 3D binary Sudoko.
#
def create_data(n_obs):
    x = np.random.uniform(size=(n_obs, 3))

    up_val = np.zeros((n_obs, 3))
    up_val[np.where(x >= 0.5)] = 1.
    up_val_sum = np.sum(up_val, axis=1)

    y = np.zeros((n_obs, 1))
    y[np.where((up_val_sum == 1) | (up_val_sum == 3))] = 1.

    return x, y


# Iterate over various neural network architectures as needed.
def main():
    n_train_obs = n_val_obs = 4096
    batch_size = 1024
    x_train, y_train = create_data(n_train_obs)
    x_val, y_val = create_data(n_val_obs)

    num_layers = range(1, 11)
    nodes_per_layer = [3, 4, 5, 6, 9, 12, 24, 48, 96, 192, 384, 768, 1536]

    n_epochs = 2000

    activations = [(ActivationType.TANH, 'tanh'), 
                   (ActivationType.RELU, 'relu'),
                   (ActivationType.ELU, 'elu')]

    for activation in activations:
        activation_type = activation[0]
        activation_str = activation[1]
        outstr = ''

        for n_lyrs in reversed(num_layers):
            for n_nodes in reversed(nodes_per_layer):
                total_nodes = n_lyrs * n_nodes

                if total_nodes > 4800:
                    continue

                tf_graph_refs, n_trainable_params = create_nn_graph(T=3,
                                                                    nodes_per_layer=n_nodes,
                                                                    layers=n_lyrs,
                                                                    dropout_rate=0.,
                                                                    l2_reg_scale=0.,
                                                                    batch_norm_momentum=0.9,
                                                                    activation_type=activation_type)

                acc_train, acc_val = train_nn(tf_graph_refs, x_train, y_train,
                                              x_val, y_val, model_name='model',
                                              n_epochs=n_epochs, save=False,
                                              batch_size=batch_size)

                outstr += f'layers: {n_lyrs}  nodes_per_layer: {n_nodes}  ' \
                          f'total_nodes: {total_nodes} ' \
                          f'   acc_train: {acc_train}    acc_val: {acc_val}    ' \
                          f'trainable parameters: {n_trainable_params}\n'

        results_path = Path(
                f"../results/results_trainsize_{n_train_obs}_epochs_{n_epochs}_"
                f"activation_{activation_str}.txt")

        with open(results_path, mode='w+', encoding="utf8") as f:
            f.write(outstr)


if __name__ == '__main__':
    main()
