Can put this in the README.md once its been tested a bit

### Disable through Environment Variables

To disable a `cachew` function in some module, you can use the `CACHEW_DISABLE` environment variable. This is a colon-delimited (like a `$PATH`) list of modules to disable. It disables modules given some name recursively, and supports [unix-style globs](https://docs.python.org/3/library/fnmatch.html)

For example, say you were using [HPI](https://github.com/karlicoss/HPI) which internally uses a snippet like `mcachew` above. You may want to enable `cachew` for _most_ modules, but disable them for specific ones. For example take:

```
my/browser
├── active_browser.py
├── all.py
├── common.py
└── export.py
my/reddit
├── __init__.py
├── all.py
├── common.py
├── pushshift.py
└── rexport.py
```

To disable `cachew` in all of these files: `export CACHEW_DISABLE=my.browser:my.reddit` (disables for all submodules)

To disable just for a particular module: `export CACHEW_DISABLE='my.browser.export'`

Similarly to `$PATH` manipulations, you can do this in your shell configuration incrementally:

```
CACHEW_DISABLE='my.reddit.rexport'
if some condition...; then
    CACHEW_DISABLE="my.browser.export:$CACHEW_DISABLE"
fi
export CACHEW_DISABLE
```

You can also use globs, e.g. `CACHEW_DISABLE='my.*.gdpr`

To disable `cachew` everywhere, you could set `export CACHEW_DISABLE='*'`
