# fmt: off
def fix_sqlalchemy_StatementError_str() -> None:
    # see https://github.com/sqlalchemy/sqlalchemy/issues/5632
    import sqlalchemy
    v = sqlalchemy.__version__
    if v != '1.3.19':
        # sigh... will still affect smaller versions.. but patching code to remove import dynamically would be far too mad
        return

    from sqlalchemy.util import compat
    from sqlalchemy.exc import StatementError as SE

    def _sql_message(self, as_unicode):
        details = [self._message(as_unicode=as_unicode)]
        if self.statement:
            # pylint: disable=no-member
            if not as_unicode and not compat.py3k:  # type: ignore[attr-defined]
                # pylint: disable=no-member
                stmt_detail = "[SQL: %s]" % compat.safe_bytestring(  # type: ignore[attr-defined]
                    self.statement
                )
            else:
                stmt_detail = "[SQL: %s]" % self.statement
            details.append(stmt_detail)
            if self.params:
                if self.hide_parameters:
                    details.append(
                        "[SQL parameters hidden due to hide_parameters=True]"
                    )
                else:
                    # NOTE: this will still cause issues
                    from sqlalchemy.sql import util

                    params_repr = util._repr_params(
                        self.params, 10, ismulti=self.ismulti
                    )
                    details.append("[parameters: %r]" % params_repr)
        code_str = self._code_str()
        if code_str:
            details.append(code_str)
        return "\n".join(["(%s)" % det for det in self.detail] + details)

    SE._sql_message = _sql_message  # type: ignore[method-assign,assignment]



import sys
import types
import functools
def _get_annotations(obj, *, globals=None, locals=None, eval_str=False):
    if isinstance(obj, type):
        # class
        obj_dict = getattr(obj, '__dict__', None)
        if obj_dict and hasattr(obj_dict, 'get'):
            ann = obj_dict.get('__annotations__', None)
            if isinstance(ann, types.GetSetDescriptorType):
                ann = None
        else:
            ann = None

        obj_globals = None
        module_name = getattr(obj, '__module__', None)
        if module_name:
            module = sys.modules.get(module_name, None)
            if module:
                obj_globals = getattr(module, '__dict__', None)
        obj_locals = dict(vars(obj))
        unwrap = obj
    elif isinstance(obj, types.ModuleType):
        # module
        ann = getattr(obj, '__annotations__', None)
        obj_globals = getattr(obj, '__dict__')
        obj_locals = None
        unwrap = None
    elif callable(obj):
        # this includes types.Function, types.BuiltinFunctionType,
        # types.BuiltinMethodType, functools.partial, functools.singledispatch,
        # "class funclike" from Lib/test/test_inspect... on and on it goes.
        ann = getattr(obj, '__annotations__', None)
        obj_globals = getattr(obj, '__globals__', None)
        obj_locals = None
        unwrap = obj
    else:
        raise TypeError(f"{obj!r} is not a module, class, or callable.")

    if ann is None:
        return {}

    if not isinstance(ann, dict):
        raise ValueError(f"{obj!r}.__annotations__ is neither a dict nor None")

    if not ann:
        return {}

    if not eval_str:
        return dict(ann)

    if unwrap is not None:
        while True:
            if hasattr(unwrap, '__wrapped__'):
                unwrap = unwrap.__wrapped__  # type: ignore[union-attr]
                continue
            if isinstance(unwrap, functools.partial):
                unwrap = unwrap.func  # type: ignore[assignment]
                continue
            break
        if hasattr(unwrap, "__globals__"):
            obj_globals = unwrap.__globals__  # type: ignore[union-attr]

    if globals is None:
        globals = obj_globals
    if locals is None:
        locals = obj_locals

    return_value = {key:
        value if not isinstance(value, str) else eval(value, globals, locals)
        for key, value in ann.items() }
    return return_value


if sys.version_info[:2] < (3, 10):
    get_annotations = _get_annotations
else:
    from inspect import get_annotations

# fmt: on
