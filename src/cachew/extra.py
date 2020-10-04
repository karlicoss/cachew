# todo Ideally, needs doublewraps as well? also typing helpers
def mcachew(*args, **kwargs):
    """
    Stands for 'Maybe cachew'.
    Defensive wrapper around @cachew to make it an optional dependency.
    """
    try:
        import cachew
    except ModuleNotFoundError:
        import warnings
        warnings.warn('cachew library not found. You might want to install it to speed things up. See https://github.com/karlicoss/cachew')
        return lambda orig_func: orig_func
    else:
        return cachew.cachew(*args, **kwargs)


from contextlib import contextmanager
@contextmanager
def disabled_cachew():
    from . import settings
    orig = settings.ENABLE
    try:
        settings.ENABLE = False
        yield
    finally:
        settings.ENABLE = orig
