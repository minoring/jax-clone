import functools


def partial(fun, *args, **kwargs):
  wrapped = functools.partial(fun, *args, **kwargs)
  functools.update_wrapper(wrapped, fun)
  wrapped._bound_args = args # TODO: What is bound args?
  return wrapped
