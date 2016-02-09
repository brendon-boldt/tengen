import numpy as np
import tensorflow as tf

def normalize(array, lower, upper):
  maxval = np.amax(array)
  minval = np.amin(array)
  translated = list(map(lambda x: x-minval+lower, array))
  scaled = list(map(lambda x: np.uint8(x*(upper-lower)/(maxval-minval)), translated))
  return np.array(scaled)

def remove_padding(tensor, y_pad, x_pad):
  dims = len(tensor.get_shape().dims)
  shape = list(map(lambda x: x.value, tensor.get_shape().dims))
  start = np.array([y_pad,x_pad], dtype=np.int32)
  start = np.pad(start, (0,dims-2), mode='constant')
  size  = np.pad([shape[0]-2*y_pad, shape[1]-2*x_pad], (0,dims-2), mode='constant', constant_values=(-1))
  return tf.slice(tensor, start, size)

def at(tensor, y, x):
  dims = len(tensor.get_shape().dims)
  start = np.array([y,x], dtype=np.int32)
  start = np.pad(start, (0,dims-2), mode='constant')
  size  = np.pad([1,1], (0,dims-2), mode='constant', constant_values=(-1))
  return tf.slice(tensor, start, size)

def partial_add(larger, smaller, org):
  l_shape = list(map(lambda x: x.value, larger.get_shape().dims))
  s_shape = list(map(lambda x: x.value, smaller.get_shape().dims))
  pad_array = [[org[0], l_shape[0]-s_shape[0]-org[0]],[org[1], l_shape[1]-s_shape[1]-org[1]]]
  pad_array = np.pad(pad_array, ((0,len(smaller.get_shape().dims)-2), (0,0)), mode='constant', constant_values=([0,0]))
  s_pad = tf.pad(smaller, pad_array)
  return larger + s_pad

def conv2d_transpose_batch(value, filters, batch_size, padding=2):
  tensor_list = []
  shape = list(map(lambda x: x.value, value.get_shape().dims))
  for i in range(batch_size):
    start = np.array([i], dtype=np.int32)
    start = np.pad(start, (0,len(shape)-1), mode='constant')
    size  = np.pad([1], (0,len(shape)-1), mode='constant', constant_values=(-1))
    sliced = tf.squeeze(tf.slice(value, start, size))
    result = conv2d_transpose(sliced, filters, padding=2)
    tensor_list.append(tf.expand_dims(result, 0))
  return tf.concat(0, tensor_list)

def conv2d_transpose(value, filters, padding=2):
  v_shape = list(map(lambda x: x.value, value.get_shape().dims))
  f_shape = list(map(lambda x: x.value, filters.get_shape().dims))
  y_size = v_shape[0]
  x_size = v_shape[1]
  channels = f_shape[2]
  # I don't think this is right
  # channels = shape[2]
  image = tf.zeros([y_size + padding*2, x_size + padding*2, channels])
  for i in range(y_size):
    for j in range(x_size):
      weighted = filters*at(value, i,j)
      delta = tf.reduce_sum(weighted, 3)
      image = partial_add(image, delta, [i, j])
  if padding > 0:
    return remove_padding(image, padding, padding)
  else:
    return image
