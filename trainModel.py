# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 09:39:48 2020

@author: sxm6202xx
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import os
import tensorflow as tf

# Number of classes here 16
num_classes = 16

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# directory where output files reside
output_directory = os.path.join(APP_ROOT,'Records','Records')

# directory where train tf record reside
training_data= os.path.join(APP_ROOT,'Records',"train-00000-of-00001")

# directory where validation tf record reside

eval_data= os.path.join(APP_ROOT,'Records',"validation-00000-of-00001")


# parametes used to train the network
num_layers=3
num_nodes=128
dropout=0.3
steps=100000
batch_size=8

def _get_input_tensors(features, labels):

    shapes = features["shape"]
    # lengths will be [batch_size]
    lengths = tf.squeeze(
        tf.slice(shapes, begin=[0, 0], size=[batch_size, 1]))
    inks = tf.reshape(features["ink"], [batch_size, -1, 3])
    if labels is not None:
      labels = tf.squeeze(labels)
    return inks, lengths, labels


def _add_regular_rnn_layers(convolved, lengths):
    """Adds RNN layers."""

    cell = tf.nn.rnn_cell.BasicLSTMCell

    cells_fw = [cell(num_nodes) for _ in range(num_layers)]
    cells_bw = [cell(num_nodes) for _ in range(num_layers)]
    if dropout > 0.0:
      cells_fw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_fw]
      cells_bw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_bw]
    outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cells_fw=cells_fw,
        cells_bw=cells_bw,
        inputs=convolved,
        sequence_length=lengths,
        dtype=tf.float32,
        scope="rnn_classification")
    return outputs



def _add_rnn_layers(convolved, lengths):

    outputs = _add_regular_rnn_layers(convolved, lengths)

    # outputs is [batch_size, L, N] where L is the maximal sequence length and N
    # the number of nodes in the last layer.
    mask = tf.tile(
        tf.expand_dims(tf.sequence_mask(lengths, tf.shape(outputs)[1]), 2),
        [1, 1, tf.shape(outputs)[2]])
    zero_outside = tf.where(mask, outputs, tf.zeros_like(outputs))
    outputs = tf.reduce_sum(zero_outside, axis=1)
    return outputs

def model_fn(features, labels, mode):

 
  # Build the model.
  inks, lengths, _labels = _get_input_tensors(features, labels)
  print(lengths)
#  print(labels.shape)  
#  print(_labels.shape) 
  
  """Adds convolution layers."""  
  
  with tf.name_scope('conv1d_1'):  
      convolved1 = tf.layers.conv1d(
          inks,
          filters=48,
          kernel_size=5,
          activation=None,
          strides=1,
          padding="same",
          name="conv1d_1")
    
  with tf.name_scope('conv1d_2'):  
      convolved2 = tf.layers.conv1d(
          convolved1,
          filters=64,
          kernel_size=5,
          activation=None,
          strides=1,
          padding="same",
          name="conv1d_2")

  with tf.name_scope('conv1d_3'):  
      convolved3 = tf.layers.conv1d(
          convolved2,
          filters=96,
          kernel_size=3,
          activation=None,
          strides=1,
          padding="same",
          name="conv1d_3")

  final_state = _add_rnn_layers(convolved3, lengths)
#  logits= tf.layers.dense(final_state, num_classes)
  with tf.name_scope('Final_Layer'):   
        logits = tf.layers.dense(inputs=final_state, units=num_classes, name= 'Final_Layer')
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
      }
  if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


  loss = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=_labels, logits=logits))
  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    

    
def get_file_lists(data_dir):
    import glob

    train_list = glob.glob(data_dir + '/' + 'train-*')
    valid_list = glob.glob(data_dir + '/' + 'validation-*')
#    test_list = glob.glob(data_dir + '/' + 'test-*')
    if len(train_list) == 0 and \
                    len(valid_list) == 0:
        raise IOError('No files found at specified path!')
    return train_list, valid_list


def get_input_fn(mode, tfrecord_pattern, batch_size):

  def _parse_tfexample_fn(example_proto, mode):

    feature_to_type = {
        "ink": tf.VarLenFeature(dtype=tf.float32),
        "shape": tf.FixedLenFeature([2], dtype=tf.int64)
    }
    if mode != tf.estimator.ModeKeys.PREDICT:

      feature_to_type["class_index"] = tf.FixedLenFeature([1], dtype=tf.int64)

    parsed_features = tf.parse_single_example(example_proto, feature_to_type)
    labels = None
    if mode != tf.estimator.ModeKeys.PREDICT:
      labels = parsed_features["class_index"]
    parsed_features["ink"] = tf.sparse_tensor_to_dense(parsed_features["ink"])
    return parsed_features, labels

  def _input_fn():

    dataset = tf.data.TFRecordDataset.list_files(tfrecord_pattern)


    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=10,
        block_length=1)
    dataset = dataset.map(
        functools.partial(_parse_tfexample_fn, mode=mode),
        num_parallel_calls=10)
    dataset = dataset.prefetch(10000)
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.repeat()
#      dataset = dataset.shuffle(buffer_size=1000000)
#        10000
    dataset = dataset.padded_batch(
        batch_size, padded_shapes=dataset.output_shapes,drop_remainder=True)
    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels

  return _input_fn



def main(unused_argv):


    classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir=os.path.join(output_directory, "tb"))

    tensors_to_log = {"probabilities": "softmax_tensor"}
    tensors_to_log ={}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50 )


    classifier.train(input_fn=get_input_fn(mode=tf.estimator.ModeKeys.EVAL, tfrecord_pattern=training_data, batch_size=batch_size), steps=1000000, hooks=[logging_hook])

    evalution = classifier.evaluate(input_fn=get_input_fn(mode=tf.estimator.ModeKeys.EVAL, tfrecord_pattern=eval_data, batch_size=batch_size))
    print(evalution)



if __name__ == "__main__":

  tf.app.run()