import re
from functools import partial
import operator
from typing import Iterator

def argscan(args: list[str]) -> Iterator[str]:
    """Yield argument flags, stripping them from `args`.

    `args` is expected to be an array of command line arguments,
    without an initial program name, essentially like `sys.argv[1:]`.
    (But see the warning below regarding how to call `argscan`).

    A _flag_ is

    - a `-` followed by a single letter or digit,
    - a `--` followed by one or more letters or digits

    argscan recognizes *batched* single letter flags, ie `-ab3d` and
    yields them as `-a`, `-b` `-3`, and so on.

    argscan does not know which flags are valid: it merely recognizes
    flags, and yields them to the caller.

    argscan does not know how many arguments a flag should consume. It
    is up to the client to save the necessary arguments (if any) *and*
    remove them from the front of `args` before letting argscan resume
    to find the next flag.

    argscan does not recognize the traditional unix `-n3` format for
    giving a value `3` to the `-n` flag. Use `-n 3` instead.

    argscan stops scanning for flags when

    - args becomes empty;
    - it encounters a word not matching one of the flag patterns
      above: this word and all remaining words in args are left for
      the client to process as desired; or
    - it encounters the word `--` to explicitly indicate the end of
      the flags. In this case, argscan *does not* yield `--` to the
      caller, but silently removes it from the arg array.

    NOTE: This follows from the above, but to be clear, `argscan` does
    not scan the entire argument list for flags: it stops when one of
    the conditions mentioned above occurs. This is consistent with
    unix conventions, and although of course there are exceptions, is
    more than sufficient for the large majority of cases. The client
    is also free to consume one or more arguments when argscan exits,
    and re-invoke it to find more arguments, but this is not an
    expected use case.

    WARNING: This routine strips option flags from the front of `args`
    as it yields them, and so the client must be able to inspect the
    modified `args` to extract any arguments associated with the flag.
    So, **DO NOT** call this function as `argscan(sys.argv[1:])`.
    Instead, do something like

    ```py
        args = sys.argv[1:]
        for flag in argscan(args):
            ...
    ```

    so that the code can inspect args (which has modified by argscan)
    as needed.
    """
    while args and args[0].startswith('-'):
        arg = args.pop(0)
        if arg in ('--'):
            break
        elif re.match(r'^-[A-Za-z0-9]{2,}$', arg):
            args[:0] = [ '-' + ch for ch in arg[1:] ]
        else:
            yield arg
