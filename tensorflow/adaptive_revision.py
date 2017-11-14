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

"""AdaptiveRevision for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import optimizer
from tensorflow.python.training import queue_runner
from tensorflow.python.training import session_manager
from tensorflow.python.training import session_run_hook
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.training import training_util
from tensorflow.python.summary import summary
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import gradients


class AdaptiveRevisionOptimizer(optimizer.Optimizer):
  """Optimizer that implements the AdaptiveRevision algorithm.

  See this [paper](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
  or this
  [intro](http://cs.stanford.edu/~ppasupat/a9online/uploads/proximal_notes.pdf).
  """

  def __init__(self, learning_rate, initial_accumulator_value=0.1, local_idx=-1,
               delay_tolerant = True,
               use_locking=False, name="AdaptiveRevision"):
    """Construct a new AdaptiveRevision optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      initial_accumulator_value: A floating point value.
        Starting value for the accumulators, must be positive.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "AdaptiveRevision".

    Raises:
      ValueError: If the `initial_accumulator_value` is invalid.
    """
    if initial_accumulator_value <= 0.0:
      raise ValueError("initial_accumulator_value must be positive: %s" %
                       initial_accumulator_value)
    super(AdaptiveRevisionOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._initial_accumulator_value = initial_accumulator_value
    # Created in Initialize.
    self._learning_rate_tensor = None
    self._local_vars = []
    self._var_local_var_maps = {}
    self._local_idx = local_idx
    self._delay_tolerant = delay_tolerant

  def compute_gradients(self, loss, var_list=None,
                        gate_gradients=optimizer.Optimizer.GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    """Compute gradients of "loss" for the variables in "var_list".

    This simply wraps the compute_gradients() from the real optimizer. The
    gradients will be aggregated in the apply_gradients() so that user can
    modify the gradients like clipping with per replica global norm if needed.
    The global norm with aggregated gradients can be bad as one replica's huge
    gradients can hurt the gradients from other replicas.

    Args:
      *args: Arguments for compute_gradients().
      **kwargs: Keyword arguments for compute_gradients().

    Returns:
      A list of (gradient, variable) pairs.
    """
    if var_list is None:
      var_list = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)

    local_vars_assign = self.allocate_local_vars(var_list)

    with ops.control_dependencies(local_vars_assign):
      loss = gen_array_ops.identity(loss)
      self._loss = loss
      grads_and_vars = super(AdaptiveRevisionOptimizer, self).compute_gradients(loss, var_list=var_list, gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss)

    vars_with_grad = [v for g, v in grads_and_vars if g is not None]
    if not vars_with_grad:
      raise ValueError(
          "No gradients provided for any variable, check your graph for ops"
          " that do not support gradients, between variables %s and loss %s." %
          ([str(v) for _, v in grads_and_vars], loss))

    valid_grads = [g for g, v in grads_and_vars if g is not None]

    return list(zip(valid_grads, vars_with_grad))

  def allocate_local_vars(self, var_list):
    if self._local_idx < 0:
      return self.get_var_local_copies(var_list)
    else:
      with ops.device("/job:worker/task:%d" % self._local_idx):
        return self.get_var_local_copies(var_list)

  def get_var_local_copies(self, var_list):
    local_vars_assign = []
    if var_list is None:
      var_list = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
    with ops.name_scope("local_var_copies", self._name) as name:
      for v in var_list:
        local_var = variables.Variable(
                        array_ops.zeros(v.get_shape(),dtype=v.dtype),
                        trainable=False,
                        collections=[ops.GraphKeys.LOCAL_VARIABLES],
                        #dtype=v.dtype,
                        name="local_var")
        curr_g = self._zeros_slot(v, 'g', self._name)
        assign_op = state_ops.assign(local_var, curr_g)
        self._local_vars.append(local_var)
        local_vars_assign.append(assign_op)
        self._var_local_var_maps[v] = local_var
    return local_vars_assign

  def _create_slots(self, var_list):
    for v in var_list:
      with ops.colocate_with(v):
        dtype = v.dtype.base_dtype
        if v.get_shape().is_fully_defined():
          init = init_ops.constant_initializer(self._initial_accumulator_value,
                                               dtype=dtype)
        else:
          # Use a Tensor instead of initializer if variable does not have static
          # shape.
          init_constant = gen_array_ops.fill(array_ops.shape(v),
                                             self._initial_accumulator_value)
          init = math_ops.cast(init_constant, dtype)
      self._get_or_make_slot_with_initializer(v, init, v.get_shape(), dtype,
                                            "z", self._name)
      self._get_or_make_slot_with_initializer(v, init, v.get_shape(), dtype,
                                            "z2", self._name)
      self._zeros_slot(v, "g", self._name)

  def _prepare(self):
    self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                       name="learning_rate")

  def _apply_dense(self, grad, var):
    with ops.colocate_with(var), ops.control_dependencies([grad]):
      g = self.get_slot(var, "g")
      z = self.get_slot(var, "z")
      z2 = self.get_slot(var, "z2")
      z2_val = gen_array_ops.identity(z2)
      z_val = gen_array_ops.identity(z)
      g_val = gen_array_ops.identity(g)
      old_g_val = gen_array_ops.identity(self._var_local_var_maps[var])

      if self._delay_tolerant:
        g_bck = g_val - old_g_val
      else:
        g_bck = array_ops.zeros_like(g_val)

      lr_old = self._learning_rate_tensor / math_ops.sqrt(z2_val)
      z_delta = math_ops.multiply(grad, grad) + 2 * math_ops.multiply(grad, g_bck)
      new_z = z_val + z_delta
      new_z2 = gen_math_ops.maximum(new_z, z2_val)
      lr_new = self._learning_rate_tensor / math_ops.sqrt(new_z2)
      delta = (-lr_new * grad) + (lr_old - lr_new) * g_bck
      v_update = state_ops.assign_add(var, delta, use_locking=self._use_locking)
      with ops.control_dependencies([v_update]):
        g_update = state_ops.assign_add(g, grad, use_locking=self._use_locking)
        z_update = state_ops.assign_add(z, z_delta, use_locking=self._use_locking)
        z2_update = state_ops.assign(z2, new_z2, use_locking=self._use_locking)

        return control_flow_ops.group(*[g_update, z_update, z2_update])

  def _resource_apply_dense(self, grad, var):
    self._apply_dense(grad, var)

