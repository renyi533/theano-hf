# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Synchronize replicas for training."""
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
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.training import training_util
from tensorflow.python.summary import summary
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import gradients


def _hessian_vector_product(ys, xs, v, grads=None):

  # Validate the input
  length = len(xs)
  if len(v) != length:
    raise ValueError("xs and v must have the same length.")

  # First backprop
  if grads is None:
    grads = gradients.gradients(ys, xs)

  assert len(grads) == length
  elemwise_products = [
      math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
      for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
  ]

  # Second backprop
  return gradients.gradients(elemwise_products, xs)

class DCAsgdOptimizer(optimizer.Optimizer):

  def __init__(self,
               opt,
               lambda_val,
               local_idx,
               rescale_variance = True,
               momentum = 0.95,
               epsilon = 1e-7,
               ps_comp = True,
               worker_cnt = -1,
               global_step=None,
               max_comp_ratio = 1e5,
               use_locking=False,
               name="DCAsgdOptimizer"):

    super(DCAsgdOptimizer, self).__init__(use_locking, name)
    logging.info("DCAsgdOptimizer init")
    self._opt = opt
    self._lambda = ops.convert_to_tensor(lambda_val)
    self._local_vars = []
    self._var_local_var_maps = {}
    self._local_idx = local_idx
    self._global_step=global_step
    self._ps_comp = ps_comp
    self._momentum = momentum
    self._epsilon = epsilon
    self._rescale_variance = rescale_variance
    self._worker_cnt = worker_cnt
    self._max_comp_ratio = max_comp_ratio
    assert local_idx >= 0


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
    self._create_slots(var_list)
    local_vars_assign = self.allocate_local_vars(var_list)

    with ops.colocate_with(loss):
      if self._global_step is None:
        curr_step = training_util.get_global_step()
        if isinstance(curr_step, ops.Tensor):
          self._local_step = curr_step + 0
          local_vars_assign.append(self._local_step)
        else:
          self._local_step = None
      else:
        self._local_step = self._global_step+0
        local_vars_assign.append(self._local_step)

    with ops.control_dependencies(local_vars_assign):
      loss = gen_array_ops.identity(loss)
      self._loss = loss
      grads_and_vars = self._opt.compute_gradients(loss, var_list=var_list, gate_gradients=gate_gradients,
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
    if self._worker_cnt > 0:
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
        if self._worker_cnt <= 0:
          local_var = variables.VariableV1(
                        array_ops.zeros(v.get_shape(),dtype=v.dtype),
                        trainable=False,
                        collections=[ops.GraphKeys.LOCAL_VARIABLES],
                        #dtype=v.dtype,
                        name="dc_asgd_local_var")
        else:
          local_var = self.get_slot(v, "var_copy_%d" % (self._local_idx))

        with ops.colocate_with(local_var):
          assign_op = state_ops.assign(local_var, v)

        self._local_vars.append(local_var)
        local_vars_assign.append(assign_op)
        self._var_local_var_maps[v] = local_var
    return local_vars_assign

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables.

    This contains most of the synchronization implementation and also wraps the
    apply_gradients() from the real optimizer.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      train_op: The op to dequeue a token so the replicas can exit this batch
      and start the next one. This is executed by each replica.

    Raises:
      ValueError: If the grads_and_vars is empty.
      ValueError: If global step is not provided, the staleness cannot be
        checked.
    """

    if not grads_and_vars:
      raise ValueError("Must supply at least one variable")
    grads = [g for g, v in grads_and_vars]
    var_list = [v for g, v in grads_and_vars]

    if self._global_step is not None:
      global_step = self._global_step

    def _comp_grad_worker():
      new_grads_worker = []
      var_diff_array = []
      with ops.name_scope("gradient_compensation_worker", self._name) as name:
        for g,v in grads_and_vars:
          assert v in self._var_local_var_maps
          with ops.device(g.device), ops.control_dependencies([g]):
            var_diff = math_ops.subtract(v.read_value(), self._var_local_var_maps[v].read_value())
            var_diff_array.append(var_diff)
        Hv = _hessian_vector_product(self._loss, var_list, var_diff_array, grads=grads)

        for g, delta in zip(grads, Hv):
          new_grads_worker.append(math_ops.add(g, delta * self._lambda))
      return new_grads_worker, var_diff_array

    def _comp_grad_ps():
      new_grads_ps = []
      var_diff_array = []

      with ops.name_scope("gradient_compensation_ps", self._name) as name:
        for g,v in grads_and_vars:
          assert v in self._var_local_var_maps
          with ops.device(v.device), ops.control_dependencies([g]):
            var_diff = math_ops.subtract(v.read_value(), self._var_local_var_maps[v].read_value())
            var_diff_array.append(var_diff)
            g_dot_g = math_ops.multiply(g, g)
            delta = math_ops.multiply(var_diff, g_dot_g)

            if self._rescale_variance:
              if self._momentum > 0.0:
                acc = self.get_slot(v, "dc_asgd_accumulator")
                updated_acc = self._momentum * acc + (1.0 - self._momentum) * g_dot_g
                if global_step is not None:
                  updated_acc = updated_acc / (1.0 - math_ops.pow(self._momentum, math_ops.cast(global_step, dtypes.float32)+1.0))
                variance_update_op = state_ops.assign(acc, updated_acc)
                with ops.control_dependencies([variance_update_op]):
                  delta = delta / math_ops.sqrt(updated_acc + self._epsilon)
              else:
                delta = math_ops.multiply(var_diff, math_ops.abs(g))

          g = math_ops.add(g, delta * self._lambda)
          new_grads_ps.append(g)

      return new_grads_ps, var_diff_array

    with ops.name_scope("gradient_compensation", self._name) as name:
      #new_grads, var_diff_array = control_flow_ops.cond(self._lambda3 > 0, _comp_grad_worker, _comp_grad_ps)
      if self._ps_comp:
        new_grads, var_diff_array = _comp_grad_ps()
      else:
        new_grads, var_diff_array = _comp_grad_worker()

      for v, diff in zip(var_list, var_diff_array):
        summary.histogram(v.name+'_delta', diff)
        summary.histogram(v.name, v)

      final_grads = []
      grads_zip = list(zip(new_grads, grads, var_list))

      for new_g, g, v in grads_zip:
        with ops.colocate_with(new_g):
          comp_delta = new_g - g
          delta_dot = math_ops.sqrt(math_ops.reduce_sum(comp_delta * comp_delta))
          g_dot = math_ops.sqrt(math_ops.reduce_sum(g * g))
          delta_ratio = delta_dot/g_dot
          summary.scalar(v.name+'_g_dot', g_dot)
          summary.scalar(v.name+'_delta_dot', delta_dot)
          summary.scalar(v.name+'_delta_ratio', delta_ratio)
          final_g = control_flow_ops.cond(delta_ratio > self._max_comp_ratio,
                        lambda: g + comp_delta * self._max_comp_ratio / delta_ratio,
                        lambda: new_g)
          final_grads.append(final_g)

      comp_grads_and_vars = list(zip(final_grads, var_list))

      train_op = self._opt.apply_gradients(comp_grads_and_vars, global_step=global_step, name=name)

    if global_step is not None and self._local_step is not None:
      with ops.control_dependencies(grads), ops.colocate_with(global_step):
        staleness = gen_array_ops.reshape(gen_array_ops.identity(global_step) - self._local_step, shape=())
        summary.scalar("DC-ASGD Gradient staleness", staleness)

      with ops.control_dependencies(new_grads), ops.colocate_with(global_step):
        staleness_final = gen_array_ops.reshape(gen_array_ops.identity(global_step) - self._local_step, shape=())
        summary.scalar("DC-ASGD final staleness", staleness_final)
      return control_flow_ops.group(*[train_op, staleness, staleness_final])
    else:
      return train_op

  def _create_slots(self, var_list):
    for v in var_list:
      with ops.colocate_with(v):
        for i in range(self._worker_cnt):
          self._zeros_slot(v, "var_copy_%d" % (i), self._name)

        if self._rescale_variance and (self._momentum > 0.0):
          dtype = v.dtype.base_dtype
          if v.get_shape().is_fully_defined():
            init = init_ops.constant_initializer(0.0,
                                               dtype=dtype)
          else:
            # Use a Tensor instead of initializer if variable does not have static
            # shape.
            init_constant = gen_array_ops.fill(array_ops.shape(v),
                                             0.0)
            init = math_ops.cast(init_constant, dtype)
          self._get_or_make_slot_with_initializer(v, init, v.get_shape(), dtype,
                                              "dc_asgd_accumulator", self._name)


  def get_slot(self, *args, **kwargs):
    """Return a slot named "name" created for "var" by the Optimizer.

    This simply wraps the get_slot() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      The `Variable` for the slot if it was created, `None` otherwise.
    """
    result  = super(DCAsgdOptimizer, self).get_slot(*args, **kwargs)
    if result is not None:
      return result

    return self._opt.get_slot(*args, **kwargs)

  def get_slot_names(self, *args, **kwargs):
    """Return a list of the names of slots created by the `Optimizer`.

    This simply wraps the get_slot_names() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      A list of strings.
    """
    result  = super(DCAsgdOptimizer, self).get_slot_names(*args, **kwargs)
    result2 = self._opt.get_slot_names(*args, **kwargs)
    result.extend(result2)
    return sorted(result)
