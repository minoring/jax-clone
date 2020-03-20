from . import core

import numpy as onp


def concretization_error_msg(fun):
  fname = getattr(fun, "__name__", fun)
  return ("Abstract value passed to function {} that requires a concrete value. "
          "Possibly tracing Python control flow using abstract values. "
          "If so, try using lax.cond or lax.while instead.").format(fname)


def concretization_function_error(fun):
  def error(self, *args):
    raise TypeError(concretization_error_msg(fun))
  return error


class UnshapedArray(core.AbstractValue):
  __slots__ = ['dtype']
  array_abstraction_level = 3

  def __init__(self, dtype):
    self.dtype = dtype

  def __eq__(self, other):
    return type(self) is type(other) and self.dtype == other.dtype

  def __hash__(self):
    return hash(str(self.dtype))

  def __repr__(self):
    return '{}({})'.format(self.__class__.__name__, self.str_short())

  _bool = _nonzero = concretization_function_error(bool)
  _float   = concretization_function_error(float)
  _int     = concretization_function_error(int)
  _long    = concretization_function_error(long)
  _complex = concretization_function_error(complex)
  _hax     = concretization_function_error(hex)
  _oct     = concretization_function_error(oct)

  def at_least_vspace(self):
    return self

  def join(self, other):
    return self

  def str_short(self):
    return onp.dtype(self.dtype).name
