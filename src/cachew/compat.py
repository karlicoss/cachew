import sys

def fix_sqlalchemy_StatementError_str() -> None:
    # see https://github.com/sqlalchemy/sqlalchemy/issues/5632
    import sqlalchemy
    v = sqlalchemy.__version__  # type: ignore[attr-defined]
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

                    params_repr = util._repr_params(  # type: ignore[attr-defined]
                        self.params, 10, ismulti=self.ismulti
                    )
                    details.append("[parameters: %r]" % params_repr)
        code_str = self._code_str()
        if code_str:
            details.append(code_str)
        return "\n".join(["(%s)" % det for det in self.detail] + details)

    SE._sql_message = _sql_message  # type: ignore[attr-defined,assignment]
