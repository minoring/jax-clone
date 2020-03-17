from absl import flags
import numpy as onp


FLAGS = flags.FLAGS
flags.DEFINE_bool('jax_enable_x64', False, 'Enable 64-bit types to be used.')


def memoize(func):
  class memodict(dict):
    def __missing__(self, key):
      val = self[key] = func(key)
      return val
  return memodict().__getitem__


_dtype_to_32bit_dtype = {
    str(onp.dtype('int64')): onp.dtype('int32'),
    str(onp.dtype('uint64')): onp.dtype('uint32'),
    str(onp.dtype('float64')): onp.dtype('float32'),
    str(onp.dtype('complex128')): onp.dtype('complex64'),
}


def canonicalize_dtype(dtype):
  """Convert from a dtype to a canonical dtype based on FLAGS.jax_enable_x64."""
  # This function is a thin wrapper around the momoized _canonicalize_dtype to
  # handle the case where FLAGS haven't been parsed yet, for example because
  # this function is called at module loading time. This situation can't obtain
  # during tracing and instead can arise when there are module-level constants
  # computed using lax or lax_numpy.
  if FLAGS.is_parsed():
    return _canonicalize_dtype(dtype)
  else:
    return dtype


@memoize
def _canonicalize_dtype(dtype):
  """Convert from a dtype to a canonical dtype based on FLAGS.jax_enable_x64."""
  dtype = onp.dtype(dtype)
  if FLAGS.jax_enable_x64:
    return str(dtype)

  return str(_dtype_to_32bit_dtype.get(str(dtype), dtype))
