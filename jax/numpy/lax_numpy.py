import numpy as onp

import jax.lax as lax

ndim = _ndim = onp.ndim
size = onp.size


def _wraps(fun):
  """Like functools.wraps but work with numpy.ufuncs."""
  docstr = """
  LAX-backed implementation of {fun}. Corresponding Numpy docstring below.

  {np_doc}
  """.format(fun=fun.__name__, np_doc=fun.__doc__)

  def wrap(op):
    try:
      op.__name__ = fun.__name__
      op.__doc__ = docstr
    finally:
      return op

  return wrap


@_wraps(onp.reshape)
def reshape(a, newshape, order='C'):
  if order == 'C' or order is None:
    dims = None
  elif order == 'F':
    dims = onp.arange(ndim(a))[::-1]
  elif order == 'A':
    dims = onp.arange(ndim(a))[::-1] if isfortran(a) else onp.arange(ndim(a))
  else:
    raise ValueError("Unexpected value for 'order' argument: {}.".format(order))

  dummy_val = onp.broadcast_to(0, a.shape)  # zero strides
  computed_newshape = onp.reshape(dummy_val, newshape).shape
  return lax.reshape(a, computed_newshape, dims)


@_wraps(onp.ravel)
def ravel(a, order="C"):
  if order == "K":
    raise NotImplementedError("Ravel not implemented for order='K'.")
  return reshape(a, (size(a),), order)


@_wraps(onp.argmax)
def argmax(a, axis=None):
  if axis is None:
    a = ravel(a)
    axis = 0
  return _argminmax(max, a, axis)


@_wraps(onp.argmin)
def argmin(a, axis=None):
  if axis is None:
    a = ravel(a)
    axis = 0
  return _argminmax(min, a, axis)
