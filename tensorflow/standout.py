from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_nn_ops import *
# pylint: enable=wildcard-import


def standout(after, is_test=False, alpha=0.0, beta=0.0, before=None, seed=None, name=None):  # pylint: disable=invalid-name
  """Computes adaptive dropout.

  Keep the activated value with probability of sigmoid(alpha * before + beta).
  https://papers.nips.cc/paper/5032-adaptive-dropout-for-training-deep-neural-networks

  Args:
    after: The activated tensor.
    is_test: Boolean, to control deterministic dropout is used.
    alpha: A tensor to control the sensitivity of dropout.
    beta: A tensor to control the bias of dropout.
    before: The pre-activated tensor.
    seed: A Python integer. Used to create random seeds. See
      @{tf.set_random_seed}
      for behavior.
    name: A name for this operation (optional).

  Returns:
    A Tensor of the same shape of `x`.
  """
  with ops.name_scope(name, "standout", [after]) as name:
    after = ops.convert_to_tensor(after, name="after")

    if before is None:
      before = after.op.inputs[0]

    keep_prob = math_ops.sigmoid(math_ops.add(math_ops.multiply(alpha, before), beta), name="keep_prob")

    # uniform [keep_prob, 1.0 + keep_prob)
    random_tensor = keep_prob
    random_tensor += random_ops.random_uniform(array_ops.shape(after),
                                               seed=seed,
                                               dtype=after.dtype)
    # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
    binary_tensor = math_ops.floor(random_tensor)
    sample = math_ops.multiply(after, binary_tensor)
    expect = math_ops.multiply(after, keep_prob)

    is_test = ops.convert_to_tensor(is_test, name="is_test")
    assert is_test.dtype is dtypes.bool
    is_test.get_shape().assert_is_compatible_with(tensor_shape.scalar())
    is_test = math_ops.cast(is_test, after.dtype)
    is_train = math_ops.subtract(1.0, is_test)

    ret = math_ops.add(math_ops.multiply(is_train, sample), math_ops.multiply(is_test, expect))
    ret.set_shape(after.get_shape())
    return ret
