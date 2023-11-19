# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# Lint as: python3
"""Generic helper functions used across codebase."""

import os
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

import keras.backend as K

# Generic.
def get_single_col_by_input_type(input_type, column_definition):
    """Returns name of single column.

    Args:
      input_type: Input type of column to extract
      column_definition: Column definition list for experiment
    """

    l = [tup[0] for tup in column_definition if tup[2] == input_type]

    if len(l) != 1:
        raise ValueError("Invalid number of columns for {}".format(input_type))

    return l[0]


def extract_cols_from_data_type(data_type, column_definition, excluded_input_types):
    """Extracts the names of columns that correspond to a define data_type.

    Args:
      data_type: DataType of columns to extract.
      column_definition: Column definition to use.
      excluded_input_types: Set of input types to exclude

    Returns:
      List of names for columns with data type specified.
    """
    return [tup[0] for tup in column_definition if tup[1] == data_type and tup[2] not in excluded_input_types]


# Loss functions.
def tensorflow_quantile_loss(y, y_pred, quantile):
    """Computes quantile loss for tensorflow.

    Standard quantile loss as defined in the "Training Procedure" section of
    the main TFT paper

    Args:
      y: Targets
      y_pred: Predictions
      quantile: Quantile to use for loss calculations (between 0 & 1)

    Returns:
      Tensor for quantile loss.
    """

    # Checks quantile
    if quantile < 0 or quantile > 1:
        raise ValueError("Illegal quantile value={}! Values should be between 0 and 1.".format(quantile))

    prediction_underflow = y - y_pred
    q_loss = quantile * tf.maximum(prediction_underflow, 0.0) + (1.0 - quantile) * tf.maximum(-prediction_underflow, 0.0)

    return tf.reduce_sum(input_tensor=q_loss, axis=-1)


def numpy_normalised_quantile_loss(y, y_pred, quantile):
    """Computes normalised quantile loss for numpy arrays.

    Uses the q-Risk metric as defined in the "Training Procedure" section of the
    main TFT paper.

    Args:
      y: Targets
      y_pred: Predictions
      quantile: Quantile to use for loss calculations (between 0 & 1)

    Returns:
      Float for normalised quantile loss.
    """
    prediction_underflow = y - y_pred
    weighted_errors = quantile * np.maximum(prediction_underflow, 0.0) + (1.0 - quantile) * np.maximum(-prediction_underflow, 0.0)

    quantile_loss = weighted_errors.mean()
    normaliser = y.abs().mean()

    return 2 * quantile_loss / normaliser


def wmae_loss(y_true, y_pred):
    
    print('y_true:',y_true.shape)
    print('y_pred:',y_pred.shape)
    
    def step_function(data):
        return tf.where(data >= 0., 1., 0.)
    
    def calculate_gain(data):
        print('data:', data.shape)
        # Set the first gain value is 1 since there is no past data for the first data.
        gain_data = tf.concat(
            [
                tf.constant([[1.,]], dtype=tf.float32), 
                tf.subtract(
                    tf.divide(
                        data[1:], 
                        data[:-1]
                    ), 
                    tf.constant(1., dtype=tf.float32))
            ],
            0 # tensor rank to concat axis
        )
        return gain_data
    
    l = tf.constant(1.5, dtype=tf.float32)
    print('l:',l.shape)
    
    diff = tf.subtract(y_true[:,:,1], y_pred[:,:,1])
    print('diff:',diff.shape)
    
    w_true = step_function(y_true[:,:,1])
    print('w_true:',w_true.shape)
    w_pred = step_function(calculate_gain(y_pred[:,:,1]))
    print('w_pred:',w_pred.shape)
    
    # Weighted mean absolute error
    threshold = tf.multiply(w_true, diff)
    print('threshold:',threshold.shape)
    print('step_function(threshold):', step_function(threshold).shape)
    
    wae = tf.multiply(
                step_function(threshold),
                (tf.multiply(
                    tf.add(
                        l,
                        tf.math.abs(tf.subtract(w_true,w_pred))
                    ),
                    tf.math.abs(diff)
                    )
                )
            )  \
        + tf.multiply(
                tf.subtract(
                    tf.constant(1., dtype=tf.float32),
                    step_function(threshold)),
                tf.multiply(
                    tf.divide(
                        tf.constant(1., dtype=tf.float32),
                        l),
                    tf.math.abs(diff)
                    )
        )
    
    print('wae:', wae.shape)
    
    wmae = tf.reduce_mean(tf.math.reduce_sum(wae))
    
    return wmae

def wmse_loss(y_true, y_pred):
    
    def step_function(data):
        return tf.where(data >= 0., 1., 0.)
    
    def calculate_gain(data):
        # Set the first gain value is 1 since there is no past data for the first data.
        gain_data = tf.concat([tf.constant([[1.,]], dtype=tf.float32), tf.subtract(tf.divide(data[1:], data[:-1]), tf.constant(1., dtype=tf.float32))], 1)
        return gain_data
    
    l = 1.5

    # -1.0 <= absolute_diff <= 1.0
    absolute_diff = y_true[:,0] - y_pred[:,0]
    
    # 1.0 <= squared_diff
    squared_diff = step_function(absolute_diff) * (absolute_diff+2)**2 \
            + (1 - step_function(absolute_diff)) * (absolute_diff-2)**2
    
    w_true = step_function(y_true[:, 1])
    w_pred = step_function(calculate_gain(y_pred[:, 1]))
    
    # Weighted mean squared error
    threshold = w_true * absolute_diff
    wse = step_function(threshold) * ((l + abs(w_true - w_pred)) * abs(squared_diff)) \
            + (1 - step_function(threshold)) * ((1/l) * abs(squared_diff))
    wmse = sum(wse) / wse.shape[0]
    
    return wmse

def f1_metric(y_true, y_pred):
    
    print('y_true(f1_metric):',y_true.shape)
    print('y_pred(f1_metric):',y_pred.shape)

    def step_function(data):
        return tf.where(data >= 0., 1., 0.)
    
    def calculate_gain(data):
        print('data:', data.shape)
        # Set the first gain value is 1 since there is no past data for the first data.
        gain_data = tf.concat(
            [
                tf.constant([[1.,]], dtype=tf.float32), 
                tf.subtract(
                    tf.divide(
                        data[1:], 
                        data[:-1]
                    ), 
                    tf.constant(1., dtype=tf.float32))
            ],
            0 # tensor rank to concat axis
        )
        return gain_data
        
    w_true = step_function(y_true[:,:,1])
    print('w_true(f1_metric):',w_true.shape)
    w_pred = step_function(calculate_gain(y_pred[:,:,1]))
    print('w_pred(f1_metric):',w_pred.shape)

    ground_positives = K.cast(K.sum(w_true, axis=0), "float") + K.epsilon()         # = TP + FN
    pred_positives = K.cast(K.sum(w_pred, axis=0), "float") + K.epsilon()         # = TP + FP
    true_positives = K.cast(K.sum(w_true * w_pred, axis=0), "float") + K.epsilon()  # = TP
    
    precision = true_positives / pred_positives 
    recall = true_positives / ground_positives
        #both = 1 if ground_positives == 0 or pred_positives == 0
        #shape (4,)

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        #still with shape (4,)

    return f1


# OS related functions.
def create_folder_if_not_exist(directory):
    """Creates folder if it doesn't exist.

    Args:
      directory: Folder path to create.
    """
    # Also creates directories recursively
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


# Tensorflow related functions.
def get_default_tensorflow_config(tf_device="gpu", gpu_id=0):
    """Creates tensorflow config for graphs to run on CPU or GPU.

    Specifies whether to run graph on gpu or cpu and which GPU ID to use for multi
    GPU machines.

    Args:
      tf_device: 'cpu' or 'gpu'
      gpu_id: GPU ID to use if relevant

    Returns:
      Tensorflow config.
    """

    if tf_device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # for training on cpu
        tf_config = tf.compat.v1.ConfigProto(log_device_placement=False, device_count={"GPU": 0})

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print("Selecting GPU ID={}".format(gpu_id))

        tf_config = tf.compat.v1.ConfigProto(log_device_placement=False)
        tf_config.gpu_options.allow_growth = True

    return tf_config


def save(tf_session, model_folder, cp_name, scope=None):
    """Saves Tensorflow graph to checkpoint.

    Saves all trainiable variables under a given variable scope to checkpoint.

    Args:
      tf_session: Session containing graph
      model_folder: Folder to save models
      cp_name: Name of Tensorflow checkpoint
      scope: Variable scope containing variables to save
    """
    # Save model
    if scope is None:
        saver = tf.compat.v1.train.Saver()
    else:
        var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        saver = tf.compat.v1.train.Saver(var_list=var_list, max_to_keep=100000)

    save_path = saver.save(tf_session, os.path.join(model_folder, "{0}.ckpt".format(cp_name)))
    print("Model saved to: {0}".format(save_path))


def load(tf_session, model_folder, cp_name, scope=None, verbose=False):
    """Loads Tensorflow graph from checkpoint.

    Args:
      tf_session: Session to load graph into
      model_folder: Folder containing serialised model
      cp_name: Name of Tensorflow checkpoint
      scope: Variable scope to use.
      verbose: Whether to print additional debugging information.
    """
    # Load model proper
    load_path = os.path.join(model_folder, "{0}.ckpt".format(cp_name))

    print("Loading model from {0}".format(load_path))

    print_weights_in_checkpoint(model_folder, cp_name)

    initial_vars = set([v.name for v in tf.compat.v1.get_default_graph().as_graph_def().node])

    # Saver
    if scope is None:
        saver = tf.compat.v1.train.Saver()
    else:
        var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        saver = tf.compat.v1.train.Saver(var_list=var_list, max_to_keep=100000)
    # Load
    saver.restore(tf_session, load_path)
    all_vars = set([v.name for v in tf.compat.v1.get_default_graph().as_graph_def().node])

    if verbose:
        print("Restored {0}".format(",".join(initial_vars.difference(all_vars))))
        print("Existing {0}".format(",".join(all_vars.difference(initial_vars))))
        print("All {0}".format(",".join(all_vars)))

    print("Done.")


def print_weights_in_checkpoint(model_folder, cp_name):
    """Prints all weights in Tensorflow checkpoint.

    Args:
      model_folder: Folder containing checkpoint
      cp_name: Name of checkpoint

    Returns:

    """
    load_path = os.path.join(model_folder, "{0}.ckpt".format(cp_name))

    print_tensors_in_checkpoint_file(file_name=load_path, tensor_name="", all_tensors=True, all_tensor_names=True)
