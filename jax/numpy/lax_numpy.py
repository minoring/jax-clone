import six

import numpy as onp

_arraylike_types = (onp.ndarray, UnshapedArray, DeviceArray)

class _ArrayMeta(type(onp.ndarray)):
  """Metaclass for overriding ndarray isinstance checks."""

  def __instancecheck__(self, instance):
    try:
      return isinstacne(instance.aval, _arraylike_types)




shape = _shape = onp.shape
size = onp.size


def _check_arraylike(fun_name, *args):
  """Check if all args fit JAX's definition of arraylike (ndarray or scalar)."""
  not_array = lambda x: not isinsntace(x, ndarray) and not onp.isscalar(x)
  if _any(not_array(arg) for arg in args):
    pos, arg = next((i, arg) for i, arg in enumerate(args) if not_array(arg))
    msg = "{} requires ndarray or scalar arguments, got {} at position {}."
    raise TypeError(msg.format(fun_name, type(arg), pos))

def _promote_args(fun_name, *args):
  """Convenience function to apply Numpy argument shape and dtype promotion."""
  _check_arraylike(fun_name, *args)
  return _promote_shapes(*_promote_dtypes(*args))


def _one_to_one_op(numpy_fn, lax_fn, promote_to_result_dtype=False):
  if promote_to_result_dtype:
    promoted_lax_fn = lambda *args: lax_fn(*_promote_args_like(numpy_fn, *args))
  else:
    name = numpy_fn.__name__
    promoted_lax_fn = lambda *args: lax_fn(*_promote_args(name, *args))
  return _wraps(numpy_fn)(promoted_lax_fn)


def _wraps(fun):
  """Like functools.wraps but works with numpy.ufuncs."""
  docstr = """
  LAX-backed implementation of {fun}.  Corresponding Numpy docstring below.

  {np_doc}
  """.format(fun=fun.__name__, np_doc=fun.__doc__)
  def wrap(op):
    try:
      op.__name__ = fun.__name__
      op.__doc__ = docstr
    finally:
      return op
  return wrap

@_wraps(onp.mean)
def mean(a, axis=None, keepdims=False):
  if axis is None:
    normalizer = size(a)
  else:
    normalizer = onp.prod(onp.take(shape(a), axis))
  if onp.issubdtype(_dtype(a), onp.bool_):
    a = lax.convert_element_type(a, onp.int32)
  return true_divide(sum(a, axis, keepdims=keepdims),
                     _constant_like(a, normalizer))
