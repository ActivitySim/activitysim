# ActivitySim
# See full license in LICENSE.txt.
import logging
import warnings

from orca import orca

_DECORATED_STEPS = {}
_DECORATED_TABLES = {}
_DECORATED_COLUMNS = {}
_DECORATED_INJECTABLES = {}
_BROADCASTS = []


# we want to allow None (any anyting else) as a default value, so just choose an improbable string
_NO_DEFAULT = "throw error if missing"

logger = logging.getLogger(__name__)


def step():
    def decorator(func):
        name = func.__name__

        logger.debug("inject step %s" % name)

        assert not _DECORATED_STEPS.get(name, False), (
            "step '%s' already decorated." % name
        )
        if _DECORATED_STEPS.get(name, False):
            warnings.warn(
                f"step {name!r} already exists, ignoring default implementation."
            )
        else:
            _DECORATED_STEPS[name] = func
            orca.add_step(name, func)

        return func

    return decorator


def injectable(cache=False, override=False):
    def decorator(func):
        name = func.__name__

        logger.debug("inject injectable %s" % name)

        # insist on explicit override to ensure multiple definitions occur in correct order
        assert override or not _DECORATED_INJECTABLES.get(name, False), (
            "injectable '%s' already defined. not overridden" % name
        )

        _DECORATED_INJECTABLES[name] = {"func": func, "cache": cache}

        orca.add_injectable(name, func, cache=cache)

        return func

    return decorator


def add_injectable(name, injectable, cache=False):
    logger.critical(f"ADD-INJECTABLE: {name}")
    return orca.add_injectable(name, injectable, cache=cache)


def get_table(name, default=_NO_DEFAULT):
    logger.critical(f"GET-TABLE: {name}")
    if orca.is_table(name) or default == _NO_DEFAULT:
        return orca.get_table(name)
    else:
        return default
