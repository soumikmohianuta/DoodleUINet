# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 18:04:39 2020

@author: soumik
"""

from datetime import datetime
import os
import random
import sys
import threading
import json
import numpy as np
import tensorflow as tf


# directory where test data resides
train_directory = r"C:\Users\sxm6202xx\Desktop\Data\DataShowCase\tfData\train"

# directory where train data resides
test_directory = r"C:\Users\sxm6202xx\Desktop\Data\DataShowCase\tfData\val"

# directory where generated tfrecords will reside
tfrecord_out_directory = r"C:\Users\sxm6202xx\Desktop\Data\DataShowCase\tfData"

# splits the train tfrecods into number of files 
num_shrads_in_train = 1

# splits the test tfrecods into number of files 
num_shrads_in_test = 1

# Label file containing the name of folders that will be considered to generate tfrecord
# either provide the full path or place it locally
label_file =  r"qlabel.txt"

# Number of threads to preprocess the images.
num_threads = 1

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _int64_shape(value):
  """Wrapper for inserting int64 features into Example proto."""

  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _convert_to_example(filename, image_buffer, label, text):

  label = label -1
  example = tf.train.Example(features=tf.train.Features(feature={
      'class_index': _int64_feature(label),
      'ink': _floats_feature(image_buffer.flatten()),
      'shape': _int64_shape(image_buffer.shape)}))
  return example



def parse_line(ndjson_line):
    
  """Parse a file and return ink (as np array) and classname."""
  inkarray = json.loads(ndjson_line)

  stroke_lengths = [len(stroke[0]) for stroke in inkarray]
  total_points = sum(stroke_lengths)
  np_ink = np.zeros((total_points, 3), dtype=np.float32)
  current_t = 0
  if not inkarray:
    print("Empty inkarray")
    return None
  for stroke in inkarray:
    if len(stroke[0]) != len(stroke[1]):
      print("Inconsistent number of x and y coordinates.")
      return None
    for i in [0, 1]:
      np_ink[current_t:(current_t + len(stroke[0])), i] = stroke[i]
    current_t += len(stroke[0])
    np_ink[current_t - 1, 2] = 1  # stroke_end
  # Preprocessing.
  # 1. Size normalization.
  lower = np.min(np_ink[:, 0:2], axis=0)
  upper = np.max(np_ink[:, 0:2], axis=0)
  scale = upper - lower
  scale[scale == 0] = 1
  np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale
  # 2. Compute deltas.
  np_ink[1:, 0:2] -= np_ink[0:-1, 0:2]
  np_ink = np_ink[1:, :]
  return np_ink



def _process_strokes(filename):

  # Read the raw file.
  with tf.gfile.FastGFile(filename, 'rb') as f:
    stroke_data = f.read()

  strokes = parse_line(stroke_data)

  return strokes


def _process_image_files_batch(thread_index, ranges, name, filenames,
                               texts, labels, num_shards):

  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(tfrecord_out_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]
      label = labels[i]
      text = texts[i]

      try:
        stroke_buffer = _process_strokes(filename)
      except Exception as e:
        print(e)
        print('SKIPPED: Unexpected eror while decoding %s.' % filename)
        continue
      if  stroke_buffer is not None:
          example = _convert_to_example(filename, stroke_buffer, label,
                                    text)
          writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()


def _process_image_files(name, filenames, texts, labels, num_shards):

  assert len(filenames) == len(texts)
  assert len(filenames) == len(labels)

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), num_threads + 1).astype(np.int)
  ranges = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
#  coder = ImageCoder()

  threads = []
  for thread_index in range(len(ranges)):
    args = (thread_index, ranges, name, filenames,
            texts, labels, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()


def _find_image_files(data_dir, labels_file):

  print('Determining list of input files and labels from %s.' % data_dir)
  unique_labels = [l.strip() for l in tf.gfile.FastGFile(
      labels_file, 'r').readlines()]

  labels = []
  filenames = []
  texts = []

  # Leave label index 0 empty as a background class.
  label_index = 1

  # Construct the list of JPEG files and labels.
  for text in unique_labels:
    jpeg_file_path = '%s/%s/*' % (data_dir, text)
    matching_files = tf.gfile.Glob(jpeg_file_path)

    labels.extend([label_index] * len(matching_files))
    texts.extend([text] * len(matching_files))
    filenames.extend(matching_files)

    if not label_index % 100:
      print('Finished finding files in %d of %d classes.' % (
          label_index, len(labels)))
    label_index += 1


  shuffled_index = list(range(len(filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  texts = [texts[i] for i in shuffled_index]
  labels = [labels[i] for i in shuffled_index]

  print('Found %d JPEG files across %d labels inside %s.' %
        (len(filenames), len(unique_labels), data_dir))
  return filenames, texts, labels


def _process_dataset(name, directory, num_shards, labels_file):

  filenames, texts, labels = _find_image_files(directory, labels_file)
  _process_image_files(name, filenames, texts, labels, num_shards)


def main(unused_argv):

  _process_dataset('validation', test_directory,
                   num_shrads_in_test, label_file)
  _process_dataset('train', train_directory,
                   num_shrads_in_train, label_file)


if __name__ == '__main__':

    tf.app.run()