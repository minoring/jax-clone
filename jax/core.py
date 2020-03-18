skip_checks = True  # not __debug__


class Primitive(object):

  def __init__(self, name):
    self.name = name

  def __repr__(self):
    return '{}'.format(self.name)

  def bind(self, *args, **kwargs):
    assert skip_checks or all(
        isinstance(arg, Tracer) or valid_jaxtype(arg) for arg in args), args
    top_trace = find_top_trace(args)
    if top_trace is None:
      return self.impl(*args, **kwargs)

    tracers = map(top_trace.full_raise, args)
    out_tracer = top_trace.process_primitive(self, tracers, kwargs)
    return full_lower(out_tracer)

  def def_impl(self, impl):
    self.impl = impl
    return impl

  def def_custom_bind(self, bind):
    self.bind = bind
    return bind

  def impl(self, *args, **kwargs):
    raise NotImplementedError("Evaluation rule for '{}' not implemented".format(
        self.name))
