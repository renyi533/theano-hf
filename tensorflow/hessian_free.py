from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import slot_creator
from tensorflow.python.ops import tensor_array_ops

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

class HessianFreeOptimizer(optimizer.Optimizer):
  """Optimizer that implements the gradient descent algorithm.
  """
  GATE_NONE = 0
  GATE_OP = 1
  GATE_GRAPH = 2
  def __init__(self, cg_iter, learning_rate=1.0, use_locking=False, name="HessianFree"):
    """Construct a new gradient descent optimizer.
    Args:
      learning_rate: A Tensor or a floating point value.  The learning
        rate to use.
      use_locking: If True use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientDescent".
    """
    super(HessianFreeOptimizer, self).__init__(use_locking, name)
    self._cg_iter = cg_iter
    self._learning_rate = learning_rate

  def _apply_dense(self, grad, var):
    return training_ops.apply_gradient_descent(
        var,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, handle):
    return training_ops.resource_apply_gradient_descent(
        handle.handle, math_ops.cast(self._learning_rate_tensor,
                                     grad.dtype.base_dtype),
        grad, use_locking=self._use_locking)

  def _prepare(self):
    self._cg_iter_tensor = ops.convert_to_tensor(self._cg_iter,
                                                       name="cg_iter")
    self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                       name="learning_rate")

  def minimize(self, loss, global_step=None, var_list=None,
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

    vars_with_grad = [v for g, v in grads_and_vars if g is not None]
    if not vars_with_grad:
      raise ValueError(
          "No gradients provided for any variable, check your graph for ops"
          " that do not support gradients, between variables %s and loss %s." %
          ([str(v) for _, v in grads_and_vars], loss))

    valid_grads = [g for g, v in grads_and_vars if g is not None]

    valid_vars_with_grad = list(zip(valid_grads, vars_with_grad))

    valid_vars_with_grad, deltas_history, residuals_history = self._conjugate_gradient(loss, valid_vars_with_grad, self._cg_iter)

    return self.apply_gradients(valid_vars_with_grad, global_step=global_step,
                                name=name)

  def _conjugate_gradient(self, loss, grads_and_vars, cg_iter):
    curr_dirs = [-g for g,v in grads_and_vars]
    curr_residuals = [-g for g,v in grads_and_vars]
    variables = [v for g,v in grads_and_vars]
    deltas = [ array_ops.zeros_like(g) for g in curr_dirs ]
    deltas_history = []
    residuals_history = []

    for i in range(cg_iter):
      Hvs = _hessian_vector_product(loss, variables, curr_dirs)
      if len(Hvs) != len(variables):
        raise ValueError("xs and Hvs must have the same length.")
      alphas = []

      for i in range(len(Hvs)):
        alphas.append( _dot(curr_residuals[i], curr_residuals[i]) / _dot(curr_dirs[i], Hvs[i]) )

      curr_deltas = []
      curr_deltas.append([d * a for d,a in list(zip(curr_dirs,alphas))])
      deltas = [ d1 + d0 for d0,d1 in list(zip(curr_deltas, deltas)) ]
      deltas_history.append(curr_deltas)
      residuals_history.append(curr_residuals)
      new_residuals = [r - a * v for r,a,v in list(zip(curr_residuals, alphas, Hvs))]
      betas =[]
      for i in range(len(curr_residuals)):
        betas.append( _dot(new_residuals[i], new_residuals[i]) / _dot(curr_residuals[i], curr_residuals[i]) )
      new_dirs = [r + b * d for r,b,d in list(zip(new_residuals, betas, curr_dirs))]
      curr_dirs = new_dirs
      curr_residuals = new_residuals

    return list(zip(deltas, variables)), deltas_history, residuals_history

