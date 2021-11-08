from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import numbers

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

# This can be used with self.assertRaisesRegexp for assert_like_rnncell.
ASSERT_LIKE_RNNCELL_ERROR_REGEXP = "is not an RNNCell"


def assert_like_rnncell(cell_name, cell):

  conditions = [
      hasattr(cell, "output_size"),
      hasattr(cell, "state_size"),
      hasattr(cell, "get_initial_state") or hasattr(cell, "zero_state"),
      callable(cell),
  ]
  errors = [
      "'output_size' property is missing",
      "'state_size' property is missing",
      "either 'zero_state' or 'get_initial_state' method is required",
      "is not callable"
  ]

  if not all(conditions):

    errors = [error for error, cond in zip(errors, conditions) if not cond]
    raise TypeError("The argument {!r} ({}) is not an RNNCell: {}.".format(
        cell_name, cell, ", ".join(errors)))


def _concat(prefix, suffix, static=False):

  if isinstance(prefix, ops.Tensor):
    p = prefix
    p_static = tensor_util.constant_value(prefix)
    if p.shape.ndims == 0:
      p = array_ops.expand_dims(p, 0)
    elif p.shape.ndims != 1:
      raise ValueError("prefix tensor must be either a scalar or vector, "
                       "but saw tensor: %s" % p)
  else:
    p = tensor_shape.as_shape(prefix)
    p_static = p.as_list() if p.ndims is not None else None
    p = (constant_op.constant(p.as_list(), dtype=dtypes.int32)
         if p.is_fully_defined() else None)
  if isinstance(suffix, ops.Tensor):
    s = suffix
    s_static = tensor_util.constant_value(suffix)
    if s.shape.ndims == 0:
      s = array_ops.expand_dims(s, 0)
    elif s.shape.ndims != 1:
      raise ValueError("suffix tensor must be either a scalar or vector, "
                       "but saw tensor: %s" % s)
  else:
    s = tensor_shape.as_shape(suffix)
    s_static = s.as_list() if s.ndims is not None else None
    s = (constant_op.constant(s.as_list(), dtype=dtypes.int32)
         if s.is_fully_defined() else None)

  if static:
    shape = tensor_shape.as_shape(p_static).concatenate(s_static)
    shape = shape.as_list() if shape.ndims is not None else None
  else:
    if p is None or s is None:
      raise ValueError("Provided a prefix or suffix of None: %s and %s"
                       % (prefix, suffix))
    shape = array_ops.concat((p, s), 0)
  return shape


def _zero_state_tensors(state_size, batch_size, dtype):
  def get_state_shape(s):
    c = _concat(batch_size, s)
    size = array_ops.zeros(c, dtype=dtype)
    if not context.executing_eagerly():
      c_static = _concat(batch_size, s, static=True)
      size.set_shape(c_static)
    return size
  return nest.map_structure(get_state_shape, state_size)


@tf_export("nn.rnn_cell.RNNCell")
class RNNCell(base_layer.Layer):

  def __init__(self, trainable=True, name=None, dtype=None, **kwargs):
    super(RNNCell, self).__init__(
        trainable=trainable, name=name, dtype=dtype, **kwargs)
    # Attribute that indicates whether the cell is a TF RNN cell, due the slight
    # difference between TF and Keras RNN cell.
    self._is_tf_rnn_cell = True

  def __call__(self, inputs, state, scope=None):

    if scope is not None:
      with vs.variable_scope(scope,
                             custom_getter=self._rnn_get_variable) as scope:
        return super(RNNCell, self).__call__(inputs, state, scope=scope)
    else:
      scope_attrname = "rnncell_scope"
      scope = getattr(self, scope_attrname, None)
      if scope is None:
        scope = vs.variable_scope(vs.get_variable_scope(),
                                  custom_getter=self._rnn_get_variable)
        setattr(self, scope_attrname, scope)
      with scope:
        return super(RNNCell, self).__call__(inputs, state)

  def _rnn_get_variable(self, getter, *args, **kwargs):
    variable = getter(*args, **kwargs)
    if context.executing_eagerly():
      trainable = variable._trainable  # pylint: disable=protected-access
    else:
      trainable = (
          variable in tf_variables.trainable_variables() or
          (isinstance(variable, tf_variables.PartitionedVariable) and
           list(variable)[0] in tf_variables.trainable_variables()))
    if trainable and variable not in self._trainable_weights:
      self._trainable_weights.append(variable)
    elif not trainable and variable not in self._non_trainable_weights:
      self._non_trainable_weights.append(variable)
    return variable

  @property
  def state_size(self):

    raise NotImplementedError("Abstract method")

  @property
  def output_size(self):
    """Integer or TensorShape: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  def build(self, _):
    # This tells the parent Layer object that it's OK to call
    # self.add_variable() inside the call() method.
    pass

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    if inputs is not None:
      # Validate the given batch_size and dtype against inputs if provided.
      inputs = ops.convert_to_tensor(inputs, name="inputs")
      if batch_size is not None:
        if tensor_util.is_tensor(batch_size):
          static_batch_size = tensor_util.constant_value(
              batch_size, partial=True)
        else:
          static_batch_size = batch_size
        if inputs.shape[0].value != static_batch_size:
          raise ValueError(
              "batch size from input tensor is different from the "
              "input param. Input tensor batch: {}, batch_size: {}".format(
                  inputs.shape[0].value, batch_size))

      if dtype is not None and inputs.dtype != dtype:
        raise ValueError(
            "dtype from input tensor is different from the "
            "input param. Input tensor dtype: {}, dtype: {}".format(
                inputs.dtype, dtype))

      batch_size = inputs.shape[0].value or array_ops.shape(inputs)[0]
      dtype = inputs.dtype
    if None in [batch_size, dtype]:
      raise ValueError(
          "batch_size and dtype cannot be None while constructing initial "
          "state: batch_size={}, dtype={}".format(batch_size, dtype))
    return self.zero_state(batch_size, dtype)

  def zero_state(self, batch_size, dtype):

    state_size = self.state_size
    is_eager = context.executing_eagerly()
    if is_eager and hasattr(self, "_last_zero_state"):
      (last_state_size, last_batch_size, last_dtype,
       last_output) = getattr(self, "_last_zero_state")
      if (last_batch_size == batch_size and
          last_dtype == dtype and
          last_state_size == state_size):
        return last_output
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      output = _zero_state_tensors(state_size, batch_size, dtype)
    if is_eager:
      self._last_zero_state = (state_size, batch_size, dtype, output)
    return output
    
    

class LayerRNNCell(RNNCell):
  def __call__(self, inputs, inputs_minus, state, state_minus, scope=None, *args, **kwargs):
    return base_layer.Layer.__call__(self, inputs, inputs_minus, state, state_minus, scope=scope,
                                     *args, **kwargs)

_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))



@tf_export("nn.rnn_cell.LSTMStateTuple")
class LSTMStateTuple(_LSTMStateTuple):

  __slots__ = ()

  @property
  def dtype(self):
    (c, h) = self
    if c.dtype != h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    return c.dtype



@tf_export("nn.rnn_cell.BasicLSTMCell")
class BasicLSTMCell(LayerRNNCell):

  @deprecated(None, "This class is deprecated, please use "
                    "tf.nn.rnn_cell.LSTMCell, which supports all the feature "
                    "this cell currently has. Please replace the existing code "
                    "with tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell').")
  def __init__(self,
               num_units,
               forget_bias=1.0,
               state_is_tuple=True,
               activation=None,
               reuse=None,
               name=None,
               dtype=None,
               **kwargs):
    super(BasicLSTMCell, self).__init__(
        _reuse=reuse, name=name, dtype=dtype, **kwargs)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if context.executing_eagerly() and context.num_gpus() > 0:
      logging.warn("%s: Note that this cell is not optimized for performance. "
                   "Please use tf.contrib.cudnn_rnn.CudnnLSTM for better "
                   "performance on GPU.", self)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  @tf_utils.shape_type_conversion
  def build(self, inputs_shape):
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[-1]
    h_depth = self._num_units
    self._kernel1 = self.add_variable("kernel1", shape=[input_depth + 2 * h_depth, self._num_units])
    self._bias1 = self.add_variable("bias1", shape=[self._num_units], initializer=init_ops.zeros_initializer(dtype=self.dtype))
    self._kernel2 = self.add_variable("kernel2", shape=[input_depth + h_depth, self._num_units])
    self._bias2 = self.add_variable("bias2", shape=[self._num_units], initializer=init_ops.zeros_initializer(dtype=self.dtype))
    self._kernel3 = self.add_variable("kernel3", shape=[input_depth + 2 * h_depth, self._num_units])
    self._bias3 = self.add_variable("bias3", shape=[self._num_units], initializer=init_ops.zeros_initializer(dtype=self.dtype))
    self._kernel4 = self.add_variable("kernel4", shape=[input_depth + 2 * h_depth, self._num_units])
    self._bias4 = self.add_variable("bias4", shape=[self._num_units], initializer=init_ops.zeros_initializer(dtype=self.dtype))
    
    self._kernel21 = self.add_variable("kernel21", shape=[input_depth + h_depth, self._num_units])
    self._bias21 = self.add_variable("bias21", shape=[self._num_units], initializer=init_ops.zeros_initializer(dtype=self.dtype))
    self._kernel22 = self.add_variable("kernel22", shape=[input_depth + h_depth, self._num_units])
    self._bias22 = self.add_variable("bias22", shape=[self._num_units], initializer=init_ops.zeros_initializer(dtype=self.dtype))
    self._kernel23 = self.add_variable("kernel23", shape=[input_depth + h_depth, self._num_units])
    self._bias23 = self.add_variable("bias23", shape=[self._num_units], initializer=init_ops.zeros_initializer(dtype=self.dtype))
    self._kernel24 = self.add_variable("kernel24", shape=[input_depth + h_depth, self._num_units])
    self._bias24 = self.add_variable("bias24", shape=[self._num_units], initializer=init_ops.zeros_initializer(dtype=self.dtype))
    
    self.built = True

  def call(self, inputs, inputs_minus, state, state_minus):

    sigmoid = math_ops.sigmoid
    one = constant_op.constant(1, dtype=dtypes.int32)
    add = math_ops.add
    multiply = math_ops.multiply
    
    
    #difference unit
    if self._state_is_tuple:
      c_minus, h_minus = state_minus
    else:
      c_minus, h_minus = array_ops.split(value=state_minus, num_or_size_splits=2, axis=one)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i_minus = nn_ops.bias_add( math_ops.matmul(array_ops.concat([inputs_minus, h_minus], 1), self._kernel21), self._bias21 )
    j_minus = nn_ops.bias_add( math_ops.matmul(array_ops.concat([inputs_minus, h_minus], 1), self._kernel22), self._bias22 )
    f_minus = nn_ops.bias_add( math_ops.matmul(array_ops.concat([inputs_minus, h_minus], 1), self._kernel23), self._bias23 )
    o_minus = nn_ops.bias_add( math_ops.matmul(array_ops.concat([inputs_minus, h_minus], 1), self._kernel24), self._bias24 )

    
    forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f_minus.dtype)
    
    new_c_minus = add(multiply(c_minus, sigmoid(add(f_minus, forget_bias_tensor))),multiply(sigmoid(i_minus), self._activation(j_minus)))
    new_h_minus = multiply(self._activation(new_c_minus), sigmoid(o_minus))

    if self._state_is_tuple:
      new_state_minus = LSTMStateTuple(new_c_minus, new_h_minus)
    else:
      new_state_minus = array_ops.concat([new_c_minus, new_h_minus], 1)
    
    
    #original cell
    if self._state_is_tuple:
      c, h = state
    else:
      c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

    i = nn_ops.bias_add( math_ops.matmul(array_ops.concat([inputs, new_h_minus, h], 1), self._kernel1), self._bias1 )
    j = nn_ops.bias_add( math_ops.matmul(array_ops.concat([inputs, h], 1), self._kernel2), self._bias2 )
    f = nn_ops.bias_add( math_ops.matmul(array_ops.concat([inputs, new_h_minus, h], 1), self._kernel3), self._bias3 )
    o = nn_ops.bias_add( math_ops.matmul(array_ops.concat([inputs, new_h_minus, h], 1), self._kernel4), self._bias4 )

    
    new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),multiply(sigmoid(i), self._activation(j)))
    new_h = multiply(self._activation(new_c), sigmoid(o))

    if self._state_is_tuple:
      new_state = LSTMStateTuple(new_c, new_h)
    else:
      new_state = array_ops.concat([new_c, new_h], 1)

      
    return new_h, new_state, new_h_minus, new_state_minus
    
    
  def get_config(self):
    config = {
        "num_units": self._num_units,
        "forget_bias": self._forget_bias,
        "state_is_tuple": self._state_is_tuple,
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(BasicLSTMCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

