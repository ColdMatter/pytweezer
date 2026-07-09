"""Generic mechanism for building simulated ("dummy") device backends.

Historically each driver hand-copied its real backend class into a parallel
``Simulated``/``Dummy`` class, reimplementing only the methods someone
remembered to. That drifts out of sync silently (missing methods, typo'd
names, stale signatures) as the real class evolves.

:func:`simulate` instead derives a simulated class's method surface from the
real class at *class definition time* (never touching an instance, so it
never talks to hardware): any public method the hand-written simulated class
doesn't already define itself gets a safe, logging, no-op stub. Only methods
with genuinely interesting fake behavior (e.g. synthesizing camera frames)
need to be hand-written; everything else stays in sync automatically as the
real class changes.

Stubs return plain ``None``/dict/list values (never a ``unittest.mock.Mock``)
so they remain serializable by ``sipyco.pyon`` when served over RPC.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Iterable

from pytweezer.logging_utils import get_logger

logger = get_logger("simulated device")


def public_methods(real_cls: type, *, exclude: Iterable[str] = ()) -> dict[str, Callable]:
    """Return ``{name: unbound function}`` for every public method of ``real_cls``.

    Inspects the class itself, not an instance, so building this mapping
    never runs the real class's ``__init__`` or touches hardware.
    """
    excluded = set(exclude)
    return {
        name: func
        for name, func in inspect.getmembers(real_cls, inspect.isfunction)
        if not name.startswith("_") and name not in excluded
    }


def _default_stub(name: str, default_return: Any = None) -> Callable:
    def stub(self, *args, **kwargs):
        logger.debug(
            "%s: %s called (args=%r, kwargs=%r)",
            type(self).__name__,
            name,
            args,
            kwargs,
        )
        return default_return() if callable(default_return) else default_return

    stub.__name__ = name
    return stub


def _provides(cls: type, name: str) -> bool:
    """True if ``cls`` already defines ``name`` itself or via a (non-``object``)
    base class. Lets :func:`simulate` compose with a shared simulated base:
    methods a base already implements are left intact instead of being
    clobbered by a no-op stub.
    """
    for klass in cls.__mro__:
        if klass is object:
            continue
        if name in klass.__dict__:
            return True
    return False


def simulate(
    real_cls: type,
    *,
    exclude: Iterable[str] = (),
    defaults: dict[str, Any] | None = None,
) -> Callable[[type], type]:
    """Class decorator: fill in any public method of ``real_cls`` missing from
    the decorated class with a logging, PYON-safe no-op stub.

    Methods already defined directly on the decorated class are left alone,
    so hand-written fake behavior always wins over the auto-generated stub.
    ``defaults`` maps method name to a return value (or a zero-arg callable
    producing one, e.g. ``dict``/``list``, so mutable defaults are fresh per
    call) for methods where returning bare ``None`` would be a poor fake.
    """
    defaults = defaults or {}

    def decorator(cls: type) -> type:
        for name in public_methods(real_cls, exclude=exclude):
            if _provides(cls, name):
                continue
            setattr(cls, name, _default_stub(name, defaults.get(name)))
        cls._simulates = real_cls
        return cls

    return decorator
