from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import slot_creator
from tensorflow.python.ops import tensor_array_ops
#from tensorflow_forward_ad.second_order import hessian_vec_fw
#from tensorflow_forward_ad.second_order import gauss_newton_vec
import tensorflow as tf

def _Lop(f, x, v):
  assert not isinstance(f, list) or isinstance(v, list), "f and v should be of the same type"
  if isinstance(x, list):
    return gradients.gradients(f, x, grad_ys=v)
  else:
    return gradients.gradients(f, x, grad_ys=v)

def _Rop(f, x, v):
  assert not isinstance(x, list) or isinstance(v, list), "x and v should be of the same type"
  if isinstance(f, list):
    w = [array_ops.ones_like(_) for _ in f]
    return gradients.gradients(_Lop(f, x, w), w, grad_ys=v)
  else:
    w = array_ops.ones_like(f)
    return gradients.gradients(_Lop(f, x, w), w, grad_ys=v)

def _gauss_newton_vec(ys, zs, xs, vs):
  """Implements Gauss-Newton vector product.
  Args:
    ys: Loss function.
    zs: Before output layer (input to softmax).
    xs: Weights, list of tensors.
    vs: List of perturbation vector for each weight tensor.
  Returns:
    J'HJv: Guass-Newton vector product.
  """
  # Validate the input
  if type(xs) == list:
    if len(vs) != len(xs):
      raise ValueError("xs and vs must have the same length.")

  grads_z = gradients.gradients(ys, zs, gate_gradients=True)

  #logic with Rop
  #hjv = forward_gradients(grads_z, xs, vs, gate_gradients=True)
  hjv = _Rop(grads_z, xs, vs)

  jhjv = gradients.gradients(zs, xs, hjv, gate_gradients=True)
  return jhjv, hjv


def _gauss_newton_vec_2(ys, zs, xs, vs):
  """Implements Gauss-Newton vector product.
  Args:
    ys: Loss function.
    zs: Before output layer (input to softmax).
    xs: Weights, list of tensors.
    vs: List of perturbation vector for each weight tensor.
  Returns:
    J'HJv: Guass-Newton vector product.
  """
  # Validate the input
  if type(xs) == list:
    if len(vs) != len(xs):
      raise ValueError("xs and vs must have the same length.")

  grads_z = gradients.gradients(ys, zs, gate_gradients=True)

  Jv = _Rop(zs, xs, vs)

  HJv = _Rop(grads_z, zs, Jv)

  Gv = gradients.gradients(zs, xs, HJv, gate_gradients=True)

  return Gv, HJv

def _hessian_vector_product(ys, xs, v):
  """Multiply the Hessian of `ys` wrt `xs` by `v`.
  This is an efficient construction that uses a backprop-like approach
  to compute the product between the Hessian and another vector. The
  Hessian is usually too large to be explicitly computed or even
  represented, but this method allows us to at least multiply by it
  for the same big-O cost as backprop.
  Implicit Hessian-vector products are the main practical, scalable way
  of using second derivatives with neural networks. They allow us to
  do things like construct Krylov subspaces and approximate conjugate
  gradient descent.
  Example: if `y` = 1/2 `x`^T A `x`, then `hessian_vector_product(y,
  x, v)` will return an expression that evaluates to the same values
  as (A + A.T) `v`.
  Args:
    ys: A scalar value, or a tensor or list of tensors to be summed to
        yield a scalar.
    xs: A list of tensors that we should construct the Hessian over.
    v: A list of tensors, with the same shapes as xs, that we want to
       multiply by the Hessian.
  Returns:
    A list of tensors (or if the list would be length 1, a single tensor)
    containing the product between the Hessian and `v`.
  Raises:
    ValueError: `xs` and `v` have different length.
  """

  # Validate the input
  length = len(xs)
  if len(v) != length:
    raise ValueError("xs and v must have the same length.")

  # First backprop
  grads = gradients.gradients(ys, xs)

  assert len(grads) == length
  elemwise_products = [
      math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
      for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
  ]

  # Second backprop
  return gradients.gradients(elemwise_products, xs)

def _dot(t0, t1):
  return math_ops.reduce_sum(math_ops.multiply(t0, t1))

def Hv( loss, z, variables, v, damping):
  Hvs = _hessian_vector_product(loss, variables, v)
  for k in range(len(Hvs)):
    Hvs[k] = Hvs[k] + damping * v[k]
  return Hvs

def Gv( loss, z, variables, v, damping):
  Gvs = _gauss_newton_vec(loss, z, variables, v)[0]
  for k in range(len(Gvs)):
    Gvs[k] = Gvs[k] + damping * v[k]
  return Gvs

def Gv2( loss, z, variables, v, damping):
  Gvs = _gauss_newton_vec_2(loss, z, variables, v)[0]
  for k in range(len(Gvs)):
    Gvs[k] = Gvs[k] + damping * v[k]
  return Gvs

def Kv( loss, z, variables, v, damping):
  Gvs = [array_ops.zeros_like(g) for g in v]
  for k in range(len(Gvs)):
    Gvs[k] = Gvs[k] + damping * v[k]
  return Gvs

class HessianFreeOptimizer(optimizer.Optimizer):
  """Optimizer that implements the gradient descent algorithm.
  """
  GATE_NONE = 0
  GATE_OP = 1
  GATE_GRAPH = 2
  def __init__(self, cg_iter, learning_rate=1.0, damping=1.0, fix_first_step=False,
          hv_method = 0, init_decay=0.0, cg_init_ratio=1.0, loss_decay=0.0,
          use_locking=False, name="HessianFree"):
    """Construct a new gradient descent optimizer.
    Args:
      learning_rate: A Tensor or a floating point value.  The learning
        rate to use.
      use_locking: If True use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "HessianFree".
    """
    super(HessianFreeOptimizer, self).__init__(use_locking, name)
    self._cg_iter = cg_iter
    self._learning_rate = learning_rate
    self._damping = damping
    self._fix_first_step = fix_first_step
    self._Hv = Hv
    self._init_decay = init_decay
    self._cg_init_ratio = cg_init_ratio
    self._loss_decay = loss_decay
    if hv_method == 0:
      self._Hv = Gv
    elif hv_method == 1:
      self._Hv = Hv
    elif hv_method == 2:
      self._Hv = Gv2
    else:
      self._Hv = Kv

  def _apply_dense(self, grad, var):
    return training_ops.apply_gradient_descent(
        var,
        math_ops.cast(1.0, var.dtype.base_dtype),
        -grad,
        use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, handle):
    return training_ops.resource_apply_gradient_descent(
        handle.handle, math_ops.cast(1.0,
                                     grad.dtype.base_dtype),
        -grad, use_locking=self._use_locking)

  def _prepare(self):
    self._cg_iter_tensor = ops.convert_to_tensor(self._cg_iter,
                                                       name="cg_iter")
    self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                       name="learning_rate")

  def minimize(self, loss, z=None, global_step=None, var_list=None,
               gate_gradients=GATE_OP, aggregation_method=None,
               colocate_gradients_with_ops=False, name=None,
               grad_loss=None):
    """Add operations to minimize `loss` by updating `var_list`.
    This method simply combines calls `compute_gradients()` and
    `apply_gradients()`. If you want to process the gradient before applying
    them call `compute_gradients()` and `apply_gradients()` explicitly instead
    of using this function.
    Args:
      loss: A `Tensor` containing the value to minimize.
      global_step: Optional `Variable` to increment by one after the
        variables have been updated.
      var_list: Optional list of `Variable` objects to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKeys.TRAINABLE_VARIABLES`.
      gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      colocate_gradients_with_ops: If True, try colocating gradients with
        the corresponding op.
      name: Optional name for the returned operation.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
    Returns:
      An Operation that updates the variables in `var_list`.  If `global_step`
      was not `None`, that operation also increments `global_step`.
    Raises:
      ValueError: If some of the variables are not `Variable` objects.
    """
    grads_and_vars = self.compute_gradients(
        loss, var_list=var_list, gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss)
    with ops.name_scope("hessian_free"):
      last_loss = variables.Variable(0.0, trainable=False, dtype=dtypes.float32, name="last_loss")
      smoothed_loss_delta = variables.Variable(0.0, trainable=False, dtype=dtypes.float32, name="smoothed_loss_delta")
      smoothed_quad_loss_delta = variables.Variable(0.0, trainable=False, dtype=dtypes.float32, name="smoothed_quad_loss_delta")
      step = variables.Variable(0, trainable=False, dtype=dtypes.int64, name="step")
      damping = variables.Variable(self._damping, trainable=False, dtype=dtypes.float32, name="damping")

    loss_delta = control_flow_ops.cond(step > 0, lambda: last_loss - loss, lambda: ops.convert_to_tensor(0.0))
    smoothed_loss_assign = state_ops.assign(smoothed_loss_delta, self._loss_decay * smoothed_loss_delta + (1 - self._loss_decay) * loss_delta)

    with ops.control_dependencies([smoothed_loss_assign]):
      '''
      curr_damping = control_flow_ops.cond(step <= 1, \
                        lambda: damping,  \
                        lambda: control_flow_ops.cond(smoothed_loss_delta/(smoothed_quad_loss_delta) > 0.75, lambda: damping * 0.99 , \
                            lambda: control_flow_ops.cond(smoothed_loss_delta/(smoothed_quad_loss_delta) < 0.01, lambda: damping / 0.99   , lambda:damping) ) )
      '''
      curr_damping = 0.99 * damping
      curr_damping = gen_math_ops.maximum(gen_math_ops.minimum(curr_damping, self._damping), self._damping*0.01)
      curr_damping = control_flow_ops.cond(curr_damping <= self._damping*0.01+1e-3 , lambda: ops.convert_to_tensor(self._damping), lambda: curr_damping)
      damping_assign = state_ops.assign(damping, curr_damping)

    vars_with_grad = [v for g, v in grads_and_vars if g is not None]
    tf.summary.scalar('last_loss', last_loss)
    tf.summary.scalar('curr_damping', curr_damping)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('loss_delta', loss_delta)
    tf.summary.scalar('smoothed_loss_delta', smoothed_loss_delta)
    tf.summary.scalar('loss_ratio', smoothed_loss_delta/(smoothed_quad_loss_delta))
    if not vars_with_grad:
      raise ValueError(
          "No gradients provided for any variable, check your graph for ops"
          " that do not support gradients, between variables %s and loss %s." %
          ([str(v) for _, v in grads_and_vars], loss))

    init_deltas = [self._zeros_slot(v, 'init_deltas', self._name) for v in vars_with_grad]

    valid_grads = [-g for g, v in grads_and_vars if g is not None]

    valid_vars_with_grad = list(zip(valid_grads, vars_with_grad))

    slot_update = []
    valid_vars_with_grad, deltas_history, residuals_history, improve_pred = \
      self._conjugate_gradient(loss, z, valid_vars_with_grad, self._cg_iter, self._fix_first_step, init_deltas, curr_damping, self._learning_rate)

    if self._init_decay > 0:
      slot_update = [self._get_or_make_slot(v, g, 'init_deltas', self._name).assign(self._init_decay*g) for g,v in valid_vars_with_grad]

    tf.summary.scalar('improve_pred', improve_pred)
    smoothed_quad_loss_assign = state_ops.assign(smoothed_quad_loss_delta, self._loss_decay * smoothed_quad_loss_delta + (1 - self._loss_decay) * improve_pred)
    tf.summary.scalar('smoothed_quad_loss_delta', smoothed_quad_loss_delta)
    step_assign = state_ops.assign_add(step, 1)
    last_loss_assign = state_ops.assign(last_loss, loss)
    slot_update.append(damping_assign)
    slot_update.append(smoothed_quad_loss_assign)
    slot_update.append(step_assign)
    slot_update.append(last_loss_assign)
    with ops.control_dependencies(slot_update):
      train_op = self.apply_gradients(valid_vars_with_grad, global_step=global_step,
                                name=name)
    return train_op

  def _conjugate_gradient(self, loss, z, grads_and_vars, cg_iter, fix_first_step=False, init_deltas=None, damping=1.0, learning_rate=1.0):
    minus_gradient = [g for g,v in grads_and_vars]
    variables = [v for g,v in grads_and_vars]

    H_vars = [array_ops.zeros_like(g) for g in minus_gradient]
    if init_deltas is not None:
      H_vars = self._Hv(loss, z, variables, init_deltas, damping)

    curr_dirs = [g - self._cg_init_ratio * b for g,b in list(zip(minus_gradient, H_vars))]
    curr_residuals = [g - self._cg_init_ratio * b for g,b in list(zip(minus_gradient, H_vars))]
    deltas = init_deltas if init_deltas is not None else [ array_ops.zeros_like(g) for g in curr_dirs ]

    deltas_history = []
    residuals_history = []
    first_alpha = 1
    for i in range(cg_iter):
      Hvs = self._Hv(loss, z, variables, curr_dirs, damping)

      if len(Hvs) != len(variables):
        raise ValueError("xs and Hvs must have the same length.")

      curr_residuals_flatten = [gen_array_ops.reshape(v, [-1]) for v in curr_residuals]
      curr_dirs_flatten = [gen_array_ops.reshape(v, [-1]) for v in curr_dirs]
      Hvs_flatten = [gen_array_ops.reshape(v, [-1]) for v in Hvs]

      curr_residuals_concat = array_ops.concat(curr_residuals_flatten, 0)
      curr_dirs_concat = array_ops.concat(curr_dirs_flatten, 0)
      Hvs_concat = array_ops.concat(Hvs_flatten, 0)
      alpha = _dot(curr_residuals_concat, curr_residuals_concat) / _dot(curr_dirs_concat, Hvs_concat)
      alpha = control_flow_ops.cond(gen_math_ops.is_finite(alpha), lambda: gen_math_ops.maximum(alpha, 1e-6), lambda : ops.convert_to_tensor(1.0))
      if i == 0 and fix_first_step:
        first_alpha = alpha
      curr_deltas = [d * (alpha / first_alpha) * learning_rate for d in curr_dirs]
      deltas = [ d1 + d0 for d0,d1 in list(zip(curr_deltas, deltas)) ]
      deltas_history.append(curr_deltas)
      residuals_history.append(curr_residuals)
      new_residuals = [r - alpha * v for r,v in list(zip(curr_residuals, Hvs))]
      new_residuals_flatten = [gen_array_ops.reshape(v, [-1]) for v in new_residuals]
      new_residuals_concat = array_ops.concat(new_residuals_flatten, 0)


      beta = _dot(new_residuals_concat, new_residuals_concat) / _dot(curr_residuals_concat, curr_residuals_concat)
      beta = control_flow_ops.cond(gen_math_ops.is_finite(beta), lambda: beta, lambda : ops.convert_to_tensor(0.0))
      #beta = gen_math_ops.maximum(beta, 1e-4)
      new_dirs = [r + beta * d for r,d in list(zip(new_residuals, curr_dirs))]
      curr_dirs = new_dirs
      curr_residuals = new_residuals

    minus_gradient_flatten = [gen_array_ops.reshape(v, [-1]) for v in minus_gradient]
    minus_gradient_concat = array_ops.concat(minus_gradient_flatten, 0)

    deltas_flatten = [gen_array_ops.reshape(v, [-1]) for v in deltas]
    deltas_concat = array_ops.concat(deltas_flatten, 0)

    G_deltas = self._Hv(loss, z, variables, deltas, 0)

    G_deltas_flatten = [gen_array_ops.reshape(v, [-1]) for v in G_deltas]
    G_deltas_concat = array_ops.concat(G_deltas_flatten, 0)
    improve_pred = _dot(deltas_concat, minus_gradient_concat-0.5*G_deltas_concat)
    return list(zip(deltas, variables)), deltas_history, residuals_history, improve_pred

