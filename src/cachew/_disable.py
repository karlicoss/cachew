import fnmatch
import logging
import os


# Could cache this, but it might be worth not to so the user can change it on the fly.
def parse_disabled_modules(logger: logging.Logger | None = None) -> list[str]:
    # e.g. CACHEW_DISABLE=my.browser:my.reddit
    disabled = os.environ.get('CACHEW_DISABLE', '')
    if disabled.strip() == '':
        return []
    if ',' in disabled and logger:
        logger.warning(
            'CACHEW_DISABLE contains a comma, but this expects a $PATH-like, colon-separated list; '
            f'try something like CACHEW_DISABLE={disabled.replace(",", ":")}'
        )
    # Remove any empty strings in case the user did something like CACHEW_DISABLE=my.module:$CACHEW_DISABLE.
    return [p for p in disabled.split(':') if p.strip() != '']


def matches_disabled_module(module_name: str, pattern: str) -> bool:
    '''
    >>> matches_disabled_module('my.browser', 'my.browser')
    True
    >>> matches_disabled_module('my.browser', 'my.*')
    True
    >>> matches_disabled_module('my.browser', 'my')
    True
    >>> matches_disabled_module('my.browser', 'my.browse*')
    True
    >>> matches_disabled_module('my.browser.export', 'my.browser')
    True
    >>> matches_disabled_module('mysomething.else', '*')  # CACHEW_DISABLE='*' disables everything
    True
    >>> matches_disabled_module('my.browser', 'my.br?????')  # fnmatch supports unix-like patterns
    True
    >>> matches_disabled_module('my.browser', 'my.browse')
    False
    >>> matches_disabled_module('mysomething.else', 'my')  # since not at '.' boundary, doesn't match
    False
    >>> matches_disabled_module('mysomething.else', '')
    False
    >>> matches_disabled_module('my.browser', 'my.browser.export')
    False
    '''

    module_parts = module_name.split('.')
    pattern_parts = pattern.split('.')

    # e.g. if pattern is 'module.submod.inner_module' and module is just 'module.submod', there is no possible way for it to match.
    if len(module_parts) < len(pattern_parts):
        return False

    for mp, pp in zip(module_parts, pattern_parts, strict=False):
        if fnmatch.fnmatch(mp, pp):
            continue
        return False
    return True


def module_is_disabled(module_name: str, logger: logging.Logger) -> bool:
    disabled_modules = parse_disabled_modules(logger)
    for pat in disabled_modules:
        if matches_disabled_module(module_name, pat):
            disabled = os.environ.get('CACHEW_DISABLE', '')
            logger.debug(f"caching disabled for {module_name} (matched '{pat}' from 'CACHEW_DISABLE={disabled}')")
            return True
    return False
