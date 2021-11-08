"""RNN helpers for TensorFlow models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


# pylint: disable=protected-access
_concat = rnn_cell_impl._concat
# pylint: enable=protected-access


def _transpose_batch_time(x):

  x_static_shape = x.get_shape()
  if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
    return x

  x_rank = array_ops.rank(x)
  x_t = array_ops.transpose(
      x, array_ops.concat(
          ([1, 0], math_ops.range(2, x_rank)), axis=0))
  x_t.set_shape(
      tensor_shape.TensorShape([
          x_static_shape[1].value, x_static_shape[0].value
      ]).concatenate(x_static_shape[2:]))
  return x_t


def _best_effort_input_batch_size(flat_input):

  for input_ in flat_input:
    shape = input_.shape
    if shape.ndims is None:
      continue
    if shape.ndims < 2:
      raise ValueError(
          "Expected input tensor %s to have rank at least 2" % input_)
    batch_size = shape[1].value
    if batch_size is not None:
      return batch_size
  # Fallback to the dynamic batch size of the first input.
  return array_ops.shape(flat_input[0])[1]


def _infer_state_dtype(explicit_dtype, state):

  if explicit_dtype is not None:
    return explicit_dtype
  elif nest.is_sequence(state):
    inferred_dtypes = [element.dtype for element in nest.flatten(state)]
    if not inferred_dtypes:
      raise ValueError("Unable to infer dtype from empty state.")
    all_same = all([x == inferred_dtypes[0] for x in inferred_dtypes])
    if not all_same:
      raise ValueError(
          "State has tensors of different inferred_dtypes. Unable to infer a "
          "single representative dtype.")
    return inferred_dtypes[0]
  else:
    return state.dtype


def _maybe_tensor_shape_from_tensor(shape):
  if isinstance(shape, ops.Tensor):
    return tensor_shape.as_shape(tensor_util.constant_value(shape))
  else:
    return shape


def _should_cache():
  """Returns True if a default caching device should be set, otherwise False."""
  if context.executing_eagerly():
    return False
  ctxt = ops.get_default_graph()._get_control_flow_context()  # pylint: disable=protected-access
  return control_flow_util.GetContainingWhileContext(ctxt) is None


def _is_keras_rnn_cell(rnn_cell):

  return (not isinstance(rnn_cell, rnn_cell_impl.RNNCell)
          and isinstance(rnn_cell, base_layer.Layer)
          and getattr(rnn_cell, "zero_state", None) is None)


# pylint: disable=unused-argument
def _rnn_step(
    time, sequence_length, min_sequence_length, max_sequence_length,
    zero_output, state, call_cell, state_size, skip_conditionals=False):

  # Convert state to a list for ease of use
  flat_state = nest.flatten(state)
  flat_zero_output = nest.flatten(zero_output)

  # Vector describing which batch entries are finished.
  copy_cond = time >= sequence_length

  def _copy_one_through(output, new_output):
    # TensorArray and scalar get passed through.
    if isinstance(output, tensor_array_ops.TensorArray):
      return new_output
    if output.shape.ndims == 0:
      return new_output
    # Otherwise propagate the old or the new value.
    with ops.colocate_with(new_output):
      return array_ops.where(copy_cond, output, new_output)

  def _copy_some_through(flat_new_output, flat_new_state):
    # Use broadcasting select to determine which values should get
    # the previous state & zero output, and which values should get
    # a calculated state & output.
    flat_new_output = [
        _copy_one_through(zero_output, new_output)
        for zero_output, new_output in zip(flat_zero_output, flat_new_output)]
    flat_new_state = [
        _copy_one_through(state, new_state)
        for state, new_state in zip(flat_state, flat_new_state)]
    return flat_new_output + flat_new_state

  def _maybe_copy_some_through():
    """Run RNN step.  Pass through either no or some past state."""
    new_output, new_state = call_cell()

    nest.assert_same_structure(state, new_state)

    flat_new_state = nest.flatten(new_state)
    flat_new_output = nest.flatten(new_output)
    return control_flow_ops.cond(
        # if t < min_seq_len: calculate and return everything
        time < min_sequence_length, lambda: flat_new_output + flat_new_state,
        # else copy some of it through
        lambda: _copy_some_through(flat_new_output, flat_new_state))

  # TODO(ebrevdo): skipping these conditionals may cause a slowdown,
  # but benefits from removing cond() and its gradient.  We should
  # profile with and without this switch here.
  if skip_conditionals:
    # Instead of using conditionals, perform the selective copy at all time
    # steps.  This is faster when max_seq_len is equal to the number of unrolls
    # (which is typical for dynamic_rnn).
    new_output, new_state = call_cell()
    nest.assert_same_structure(state, new_state)
    new_state = nest.flatten(new_state)
    new_output = nest.flatten(new_output)
    final_output_and_state = _copy_some_through(new_output, new_state)
  else:
    empty_update = lambda: flat_zero_output + flat_state
    final_output_and_state = control_flow_ops.cond(
        # if t >= max_seq_len: copy all state through, output zeros
        time >= max_sequence_length, empty_update,
        # otherwise calculation is required: copy some or all of it through
        _maybe_copy_some_through)

  if len(final_output_and_state) != len(flat_zero_output) + len(flat_state):
    raise ValueError("Internal error: state and output were not concatenated "
                     "correctly.")
  final_output = final_output_and_state[:len(flat_zero_output)]
  final_state = final_output_and_state[len(flat_zero_output):]

  for output, flat_output in zip(final_output, flat_zero_output):
    output.set_shape(flat_output.get_shape())
  for substate, flat_substate in zip(final_state, flat_state):
    if not isinstance(substate, tensor_array_ops.TensorArray):
      substate.set_shape(flat_substate.get_shape())

  final_output = nest.pack_sequence_as(
      structure=zero_output, flat_sequence=final_output)
  final_state = nest.pack_sequence_as(
      structure=state, flat_sequence=final_state)

  return final_output, final_state


def _reverse_seq(input_seq, lengths):

  if lengths is None:
    return list(reversed(input_seq))

  flat_input_seq = tuple(nest.flatten(input_) for input_ in input_seq)

  flat_results = [[] for _ in range(len(input_seq))]
  for sequence in zip(*flat_input_seq):
    input_shape = tensor_shape.unknown_shape(
        ndims=sequence[0].get_shape().ndims)
    for input_ in sequence:
      input_shape.merge_with(input_.get_shape())
      input_.set_shape(input_shape)

    # Join into (time, batch_size, depth)
    s_joined = array_ops.stack(sequence)

    # Reverse along dimension 0
    s_reversed = array_ops.reverse_sequence(s_joined, lengths, 0, 1)
    # Split again into list
    result = array_ops.unstack(s_reversed)
    for r, flat_result in zip(result, flat_results):
      r.set_shape(input_shape)
      flat_result.append(r)

  results = [nest.pack_sequence_as(structure=input_, flat_sequence=flat_result)
             for input_, flat_result in zip(input_seq, flat_results)]
  return results


@tf_export("nn.bidirectional_dynamic_rnn")
def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                              initial_state_fw=None, initial_state_bw=None,
                              dtype=None, parallel_iterations=None,
                              swap_memory=False, time_major=False, scope=None):

  rnn_cell_impl.assert_like_rnncell("cell_fw", cell_fw)
  rnn_cell_impl.assert_like_rnncell("cell_bw", cell_bw)

  with vs.variable_scope(scope or "bidirectional_rnn"):
    # Forward direction
    with vs.variable_scope("fw") as fw_scope:
      output_fw, output_state_fw = dynamic_rnn(
          cell=cell_fw, inputs=inputs, sequence_length=sequence_length,
          initial_state=initial_state_fw, dtype=dtype,
          parallel_iterations=parallel_iterations, swap_memory=swap_memory,
          time_major=time_major, scope=fw_scope)

    # Backward direction
    if not time_major:
      time_axis = 1
      batch_axis = 0
    else:
      time_axis = 0
      batch_axis = 1

    def _reverse(input_, seq_lengths, seq_axis, batch_axis):
      if seq_lengths is not None:
        return array_ops.reverse_sequence(
            input=input_, seq_lengths=seq_lengths,
            seq_axis=seq_axis, batch_axis=batch_axis)
      else:
        return array_ops.reverse(input_, axis=[seq_axis])

    with vs.variable_scope("bw") as bw_scope:

      def _map_reverse(inp):
        return _reverse(
            inp,
            seq_lengths=sequence_length,
            seq_axis=time_axis,
            batch_axis=batch_axis)

      inputs_reverse = nest.map_structure(_map_reverse, inputs)
      tmp, output_state_bw = dynamic_rnn(
          cell=cell_bw, inputs=inputs_reverse, sequence_length=sequence_length,
          initial_state=initial_state_bw, dtype=dtype,
          parallel_iterations=parallel_iterations, swap_memory=swap_memory,
          time_major=time_major, scope=bw_scope)

  output_bw = _reverse(
      tmp, seq_lengths=sequence_length,
      seq_axis=time_axis, batch_axis=batch_axis)

  outputs = (output_fw, output_bw)
  output_states = (output_state_fw, output_state_bw)

  return (outputs, output_states)


@tf_export("nn.dynamic_rnn")
def dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
                dtype=None, parallel_iterations=None, swap_memory=False,
                time_major=False, scope=None):

  rnn_cell_impl.assert_like_rnncell("cell", cell)

  with vs.variable_scope(scope or "rnn") as varscope:
    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    if _should_cache():
      if varscope.caching_device is None:
        varscope.set_caching_device(lambda op: op.device)

    # By default, time_major==False and inputs are batch-major: shaped
    #   [batch, time, depth]
    # For internal calculations, we transpose to [time, batch, depth]
    flat_input = nest.flatten(inputs)

    if not time_major:
      # (B,T,D) => (T,B,D)
      flat_input = [ops.convert_to_tensor(input_) for input_ in flat_input]
      flat_input = tuple(_transpose_batch_time(input_) for input_ in flat_input)

    parallel_iterations = parallel_iterations or 32
    if sequence_length is not None:
      sequence_length = math_ops.to_int32(sequence_length)
      if sequence_length.get_shape().ndims not in (None, 1):
        raise ValueError(
            "sequence_length must be a vector of length batch_size, "
            "but saw shape: %s" % sequence_length.get_shape())
      sequence_length = array_ops.identity(  # Just to find it in the graph.
          sequence_length, name="sequence_length")

    batch_size = _best_effort_input_batch_size(flat_input)

    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError("If there is no initial_state, you must give a dtype.")
      if getattr(cell, "get_initial_state", None) is not None:
        state = cell.get_initial_state(
            inputs=None, batch_size=batch_size, dtype=dtype)
      else:
        state = cell.zero_state(batch_size, dtype)

    def _assert_has_shape(x, shape):
      x_shape = array_ops.shape(x)
      packed_shape = array_ops.stack(shape)
      return control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(x_shape, packed_shape)),
          ["Expected shape for Tensor %s is " % x.name,
           packed_shape, " but saw shape: ", x_shape])

    if not context.executing_eagerly() and sequence_length is not None:
      # Perform some shape validation
      with ops.control_dependencies(
          [_assert_has_shape(sequence_length, [batch_size])]):
        sequence_length = array_ops.identity(
            sequence_length, name="CheckSeqLen")

    inputs = nest.pack_sequence_as(structure=inputs, flat_sequence=flat_input)

    (outputs, final_state) = _dynamic_rnn_loop(
        cell,
        inputs,
        state,
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory,
        sequence_length=sequence_length,
        dtype=dtype)

    # Outputs of _dynamic_rnn_loop are always shaped [time, batch, depth].
    # If we are performing batch-major calculations, transpose output back
    # to shape [batch, time, depth]
    if not time_major:
      # (T,B,D) => (B,T,D)
      outputs = nest.map_structure(_transpose_batch_time, outputs)

    return (outputs, final_state)


def _dynamic_rnn_loop(cell,
                      inputs,
                      initial_state,
                      parallel_iterations,
                      swap_memory,
                      sequence_length=None,
                      dtype=None):

  state = initial_state
  assert isinstance(parallel_iterations, int), "parallel_iterations must be int"

  state_size = cell.state_size

  flat_input = nest.flatten(inputs)
  flat_output_size = nest.flatten(cell.output_size)

  # Construct an initial output
  input_shape = array_ops.shape(flat_input[0])
  time_steps = input_shape[0]
  batch_size = _best_effort_input_batch_size(flat_input)

  inputs_got_shape = tuple(input_.get_shape().with_rank_at_least(3)
                           for input_ in flat_input)

  const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]

  for shape in inputs_got_shape:
    if not shape[2:].is_fully_defined():
      raise ValueError(
          "Input size (depth of inputs) must be accessible via shape inference,"
          " but saw value None.")
    got_time_steps = shape[0].value
    got_batch_size = shape[1].value
    if const_time_steps != got_time_steps:
      raise ValueError(
          "Time steps is not the same for all the elements in the input in a "
          "batch.")
    if const_batch_size != got_batch_size:
      raise ValueError(
          "Batch_size is not the same for all the elements in the input.")

  # Prepare dynamic conditional copying of state & output
  def _create_zero_arrays(size):
    size = _concat(batch_size, size)
    return array_ops.zeros(
        array_ops.stack(size), _infer_state_dtype(dtype, state))

  flat_zero_output = tuple(_create_zero_arrays(output)
                           for output in flat_output_size)
  zero_output = nest.pack_sequence_as(structure=cell.output_size,
                                      flat_sequence=flat_zero_output)

  if sequence_length is not None:
    min_sequence_length = math_ops.reduce_min(sequence_length)
    max_sequence_length = math_ops.reduce_max(sequence_length)
  else:
    max_sequence_length = time_steps

  time = array_ops.constant(0, dtype=dtypes.int32, name="time")

  with ops.name_scope("dynamic_rnn") as scope:
    base_name = scope

  def _create_ta(name, element_shape, dtype):
    return tensor_array_ops.TensorArray(dtype=dtype,
                                        size=time_steps,
                                        element_shape=element_shape,
                                        tensor_array_name=base_name + name)

  in_graph_mode = not context.executing_eagerly()
  if in_graph_mode:
    output_ta = tuple(
        _create_ta(
            "output_%d" % i,
            element_shape=(tensor_shape.TensorShape([const_batch_size])
                           .concatenate(
                               _maybe_tensor_shape_from_tensor(out_size))),
            dtype=_infer_state_dtype(dtype, state))
        for i, out_size in enumerate(flat_output_size))
    input_ta = tuple(
        _create_ta(
            "input_%d" % i,
            element_shape=flat_input_i.shape[1:],
            dtype=flat_input_i.dtype)
        for i, flat_input_i in enumerate(flat_input))
    input_ta = tuple(ta.unstack(input_)
                     for ta, input_ in zip(input_ta, flat_input))
  else:
    output_ta = tuple([0 for _ in range(time_steps.numpy())]
                      for i in range(len(flat_output_size)))
    input_ta = flat_input

  def _time_step(time, output_ta_t, state):

    if in_graph_mode:
      input_t = tuple(ta.read(time) for ta in input_ta)
      # Restore some shape information
      for input_, shape in zip(input_t, inputs_got_shape):
        input_.set_shape(shape[1:])
    else:
      input_t = tuple(ta[time.numpy()] for ta in input_ta)

    input_t = nest.pack_sequence_as(structure=inputs, flat_sequence=input_t)
    # Keras RNN cells only accept state as list, even if it's a single tensor.
    is_keras_rnn_cell = _is_keras_rnn_cell(cell)
    if is_keras_rnn_cell and not nest.is_sequence(state):
      state = [state]
    call_cell = lambda: cell(input_t, state)

    if sequence_length is not None:
      (output, new_state) = _rnn_step(
          time=time,
          sequence_length=sequence_length,
          min_sequence_length=min_sequence_length,
          max_sequence_length=max_sequence_length,
          zero_output=zero_output,
          state=state,
          call_cell=call_cell,
          state_size=state_size,
          skip_conditionals=True)
    else:
      (output, new_state) = call_cell()

    # Keras cells always wrap state as list, even if it's a single tensor.
    if is_keras_rnn_cell and len(new_state) == 1:
      new_state = new_state[0]
    # Pack state if using state tuples
    output = nest.flatten(output)

    if in_graph_mode:
      output_ta_t = tuple(
          ta.write(time, out) for ta, out in zip(output_ta_t, output))
    else:
      for ta, out in zip(output_ta_t, output):
        ta[time.numpy()] = out

    return (time + 1, output_ta_t, new_state)

  if in_graph_mode:
    # Make sure that we run at least 1 step, if necessary, to ensure
    # the TensorArrays pick up the dynamic shape.
    loop_bound = math_ops.minimum(
        time_steps, math_ops.maximum(1, max_sequence_length))
  else:
    # Using max_sequence_length isn't currently supported in the Eager branch.
    loop_bound = time_steps

  _, output_final_ta, final_state = control_flow_ops.while_loop(
      cond=lambda time, *_: time < loop_bound,
      body=_time_step,
      loop_vars=(time, output_ta, state),
      parallel_iterations=parallel_iterations,
      maximum_iterations=time_steps,
      swap_memory=swap_memory)

  # Unpack final output if not using output tuples.
  if in_graph_mode:
    final_outputs = tuple(ta.stack() for ta in output_final_ta)
    # Restore some shape information
    for output, output_size in zip(final_outputs, flat_output_size):
      shape = _concat(
          [const_time_steps, const_batch_size], output_size, static=True)
      output.set_shape(shape)
  else:
    final_outputs = output_final_ta

  final_outputs = nest.pack_sequence_as(
      structure=cell.output_size, flat_sequence=final_outputs)
  if not in_graph_mode:
    final_outputs = nest.map_structure_up_to(
        cell.output_size, lambda x: array_ops.stack(x, axis=0), final_outputs)

  return (final_outputs, final_state)


@tf_export("nn.raw_rnn")
def raw_rnn(cell, loop_fn,
            parallel_iterations=None, swap_memory=False, scope=None):

  rnn_cell_impl.assert_like_rnncell("cell", cell)

  if not callable(loop_fn):
    raise TypeError("loop_fn must be a callable")

  parallel_iterations = parallel_iterations or 32

  # Create a new scope in which the caching device is either
  # determined by the parent scope, or is set to place the cached
  # Variable using the same placement as for the rest of the RNN.
  with vs.variable_scope(scope or "rnn") as varscope:
    if _should_cache():
      if varscope.caching_device is None:
        varscope.set_caching_device(lambda op: op.device)

    time = constant_op.constant(0, dtype=dtypes.int32)
    (elements_finished, next_input, initial_state, emit_structure,
     init_loop_state) = loop_fn(
         time, None, None, None)  # time, cell_output, cell_state, loop_state
    flat_input = nest.flatten(next_input)

    # Need a surrogate loop state for the while_loop if none is available.
    loop_state = (init_loop_state if init_loop_state is not None
                  else constant_op.constant(0, dtype=dtypes.int32))

    input_shape = [input_.get_shape() for input_ in flat_input]
    static_batch_size = input_shape[0][0]

    for input_shape_i in input_shape:
      # Static verification that batch sizes all match
      static_batch_size.merge_with(input_shape_i[0])

    batch_size = static_batch_size.value
    const_batch_size = batch_size
    if batch_size is None:
      batch_size = array_ops.shape(flat_input[0])[0]

    nest.assert_same_structure(initial_state, cell.state_size)
    state = initial_state
    flat_state = nest.flatten(state)
    flat_state = [ops.convert_to_tensor(s) for s in flat_state]
    state = nest.pack_sequence_as(structure=state,
                                  flat_sequence=flat_state)

    if emit_structure is not None:
      flat_emit_structure = nest.flatten(emit_structure)
      flat_emit_size = [emit.shape if emit.shape.is_fully_defined() else
                        array_ops.shape(emit) for emit in flat_emit_structure]
      flat_emit_dtypes = [emit.dtype for emit in flat_emit_structure]
    else:
      emit_structure = cell.output_size
      flat_emit_size = nest.flatten(emit_structure)
      flat_emit_dtypes = [flat_state[0].dtype] * len(flat_emit_size)

    flat_emit_ta = [
        tensor_array_ops.TensorArray(
            dtype=dtype_i,
            dynamic_size=True,
            element_shape=(tensor_shape.TensorShape([const_batch_size])
                           .concatenate(
                               _maybe_tensor_shape_from_tensor(size_i))),
            size=0,
            name="rnn_output_%d" % i)
        for i, (dtype_i, size_i)
        in enumerate(zip(flat_emit_dtypes, flat_emit_size))]
    emit_ta = nest.pack_sequence_as(structure=emit_structure,
                                    flat_sequence=flat_emit_ta)
    flat_zero_emit = [
        array_ops.zeros(_concat(batch_size, size_i), dtype_i)
        for size_i, dtype_i in zip(flat_emit_size, flat_emit_dtypes)]
    zero_emit = nest.pack_sequence_as(structure=emit_structure,
                                      flat_sequence=flat_zero_emit)

    def condition(unused_time, elements_finished, *_):
      return math_ops.logical_not(math_ops.reduce_all(elements_finished))

    def body(time, elements_finished, current_input,
             emit_ta, state, loop_state):

      (next_output, cell_state) = cell(current_input, state)

      nest.assert_same_structure(state, cell_state)
      nest.assert_same_structure(cell.output_size, next_output)

      next_time = time + 1
      (next_finished, next_input, next_state, emit_output,
       next_loop_state) = loop_fn(
           next_time, next_output, cell_state, loop_state)

      nest.assert_same_structure(state, next_state)
      nest.assert_same_structure(current_input, next_input)
      nest.assert_same_structure(emit_ta, emit_output)

      # If loop_fn returns None for next_loop_state, just reuse the
      # previous one.
      loop_state = loop_state if next_loop_state is None else next_loop_state

      def _copy_some_through(current, candidate):
        """Copy some tensors through via array_ops.where."""
        def copy_fn(cur_i, cand_i):
          # TensorArray and scalar get passed through.
          if isinstance(cur_i, tensor_array_ops.TensorArray):
            return cand_i
          if cur_i.shape.ndims == 0:
            return cand_i
          # Otherwise propagate the old or the new value.
          with ops.colocate_with(cand_i):
            return array_ops.where(elements_finished, cur_i, cand_i)
        return nest.map_structure(copy_fn, current, candidate)

      emit_output = _copy_some_through(zero_emit, emit_output)
      next_state = _copy_some_through(state, next_state)

      emit_ta = nest.map_structure(
          lambda ta, emit: ta.write(time, emit), emit_ta, emit_output)

      elements_finished = math_ops.logical_or(elements_finished, next_finished)

      return (next_time, elements_finished, next_input,
              emit_ta, next_state, loop_state)

    returned = control_flow_ops.while_loop(
        condition, body, loop_vars=[
            time, elements_finished, next_input,
            emit_ta, state, loop_state],
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory)

    (emit_ta, final_state, final_loop_state) = returned[-3:]

    if init_loop_state is None:
      final_loop_state = None

    return (emit_ta, final_state, final_loop_state)


@tf_export("nn.static_rnn")
def static_rnn(cell,
               inputs,
               inputs_minus,
               initial_state=None,
               dtype=None,
               sequence_length=None,
               scope=None):
  rnn_cell_impl.assert_like_rnncell("cell", cell)
  if not nest.is_sequence(inputs):
    raise TypeError("inputs must be a sequence")
  if not inputs:
    raise ValueError("inputs must not be empty")

  outputs = []

  with vs.variable_scope(scope or "rnn") as varscope:
    if _should_cache():
      if varscope.caching_device is None:
        varscope.set_caching_device(lambda op: op.device)

    # Obtain the first sequence of the input
    first_input = inputs
    while nest.is_sequence(first_input):
      first_input = first_input[0]

    if first_input.get_shape().ndims != 1:

      input_shape = first_input.get_shape().with_rank_at_least(2)
      fixed_batch_size = input_shape[0]

      flat_inputs = nest.flatten(inputs)
      for flat_input in flat_inputs:
        input_shape = flat_input.get_shape().with_rank_at_least(2)
        batch_size, input_size = input_shape[0], input_shape[1:]
        fixed_batch_size.merge_with(batch_size)
        for i, size in enumerate(input_size):
          if size.value is None:
            raise ValueError(
                "Input size (dimension %d of inputs) must be accessible via "
                "shape inference, but saw value None." % i)
    else:
      fixed_batch_size = first_input.get_shape().with_rank_at_least(1)[0]

    if fixed_batch_size.value:
      batch_size = fixed_batch_size.value
    else:
      batch_size = array_ops.shape(first_input)[0]
    if initial_state is not None:
      state = initial_state
      state_minus = initial_state
    else:
      if not dtype:
        raise ValueError("If no initial_state is provided, "
                         "dtype must be specified")
      if getattr(cell, "get_initial_state", None) is not None:
        state = cell.get_initial_state(
            inputs=None, batch_size=batch_size, dtype=dtype)
        state_minus = cell.get_initial_state(
            inputs=None, batch_size=batch_size, dtype=dtype)
      else:
        state = cell.zero_state(batch_size, dtype)
        state_minus = cell.zero_state(batch_size, dtype)

    if sequence_length is not None:  # Prepare variables
      sequence_length = ops.convert_to_tensor(
          sequence_length, name="sequence_length")
      if sequence_length.get_shape().ndims not in (None, 1):
        raise ValueError(
            "sequence_length must be a vector of length batch_size")

      def _create_zero_output(output_size):
        # convert int to TensorShape if necessary
        size = _concat(batch_size, output_size)
        output = array_ops.zeros(
            array_ops.stack(size), _infer_state_dtype(dtype, state))
        shape = _concat(fixed_batch_size.value, output_size, static=True)
        output.set_shape(tensor_shape.TensorShape(shape))
        return output

      output_size = cell.output_size
      flat_output_size = nest.flatten(output_size)
      flat_zero_output = tuple(
          _create_zero_output(size) for size in flat_output_size)
      zero_output = nest.pack_sequence_as(
          structure=output_size, flat_sequence=flat_zero_output)

      sequence_length = math_ops.to_int32(sequence_length)
      min_sequence_length = math_ops.reduce_min(sequence_length)
      max_sequence_length = math_ops.reduce_max(sequence_length)

    # Keras RNN cells only accept state as list, even if it's a single tensor.
    is_keras_rnn_cell = _is_keras_rnn_cell(cell)
    if is_keras_rnn_cell and not nest.is_sequence(state):
      state = [state]
      state_minus = [state_minus]
    for time, input_ in enumerate(inputs):
      if time > 0:
        varscope.reuse_variables()
      input_minus = inputs_minus[time]
      # pylint: disable=cell-var-from-loop
      call_cell = lambda: cell(input_, input_minus, state ,state_minus)
      # pylint: enable=cell-var-from-loop
      if sequence_length is not None:
        (output, state) = _rnn_step(
            time=time,
            sequence_length=sequence_length,
            min_sequence_length=min_sequence_length,
            max_sequence_length=max_sequence_length,
            zero_output=zero_output,
            state=state,
            call_cell=call_cell,
            state_size=cell.state_size)
      else:
        (output, state, output_minus, state_minus) = call_cell()
      output = array_ops.concat([output, output_minus], -1)
      outputs.append(output)
    if is_keras_rnn_cell and len(state) == 1:
      state = state[0]
      state_minus = state_minus[0]

    return (outputs, state, state_minus)


@tf_export("nn.static_state_saving_rnn")
def static_state_saving_rnn(cell,
                            inputs,
                            state_saver,
                            state_name,
                            sequence_length=None,
                            scope=None):

  state_size = cell.state_size
  state_is_tuple = nest.is_sequence(state_size)
  state_name_tuple = nest.is_sequence(state_name)

  if state_is_tuple != state_name_tuple:
    raise ValueError("state_name should be the same type as cell.state_size.  "
                     "state_name: %s, cell.state_size: %s" % (str(state_name),
                                                              str(state_size)))

  if state_is_tuple:
    state_name_flat = nest.flatten(state_name)
    state_size_flat = nest.flatten(state_size)

    if len(state_name_flat) != len(state_size_flat):
      raise ValueError("#elems(state_name) != #elems(state_size): %d vs. %d" %
                       (len(state_name_flat), len(state_size_flat)))

    initial_state = nest.pack_sequence_as(
        structure=state_size,
        flat_sequence=[state_saver.state(s) for s in state_name_flat])
  else:
    initial_state = state_saver.state(state_name)

  (outputs, state) = static_rnn(
      cell,
      inputs,
      initial_state=initial_state,
      sequence_length=sequence_length,
      scope=scope)

  if state_is_tuple:
    flat_state = nest.flatten(state)
    state_name = nest.flatten(state_name)
    save_state = [
        state_saver.save_state(name, substate)
        for name, substate in zip(state_name, flat_state)
    ]
  else:
    save_state = [state_saver.save_state(state_name, state)]

  with ops.control_dependencies(save_state):
    last_output = outputs[-1]
    flat_last_output = nest.flatten(last_output)
    flat_last_output = [
        array_ops.identity(output) for output in flat_last_output
    ]
    outputs[-1] = nest.pack_sequence_as(
        structure=last_output, flat_sequence=flat_last_output)

    if state_is_tuple:
      state = nest.pack_sequence_as(
          structure=state,
          flat_sequence=[array_ops.identity(s) for s in flat_state])
    else:
      state = array_ops.identity(state)

  return (outputs, state)


@tf_export("nn.static_bidirectional_rnn")
def static_bidirectional_rnn(cell_fw,
                             cell_bw,
                             inputs,
                             initial_state_fw=None,
                             initial_state_bw=None,
                             dtype=None,
                             sequence_length=None,
                             scope=None):
  rnn_cell_impl.assert_like_rnncell("cell_fw", cell_fw)
  rnn_cell_impl.assert_like_rnncell("cell_bw", cell_bw)
  if not nest.is_sequence(inputs):
    raise TypeError("inputs must be a sequence")
  if not inputs:
    raise ValueError("inputs must not be empty")

  with vs.variable_scope(scope or "bidirectional_rnn"):
    # Forward direction
    with vs.variable_scope("fw") as fw_scope:
      output_fw, output_state_fw = static_rnn(
          cell_fw,
          inputs,
          initial_state_fw,
          dtype,
          sequence_length,
          scope=fw_scope)

    # Backward direction
    with vs.variable_scope("bw") as bw_scope:
      reversed_inputs = _reverse_seq(inputs, sequence_length)
      tmp, output_state_bw = static_rnn(
          cell_bw,
          reversed_inputs,
          initial_state_bw,
          dtype,
          sequence_length,
          scope=bw_scope)

  output_bw = _reverse_seq(tmp, sequence_length)
  # Concat each of the forward/backward outputs
  flat_output_fw = nest.flatten(output_fw)
  flat_output_bw = nest.flatten(output_bw)

  flat_outputs = tuple(
      array_ops.concat([fw, bw], 1)
      for fw, bw in zip(flat_output_fw, flat_output_bw))

  outputs = nest.pack_sequence_as(
      structure=output_fw, flat_sequence=flat_outputs)

  return (outputs, output_state_fw, output_state_bw)
