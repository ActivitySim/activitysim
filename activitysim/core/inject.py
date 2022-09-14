# ActivitySim
# See full license in LICENSE.txt.
import logging

from orca import orca

_DECORATED_STEPS = {}
_DECORATED_TABLES = {}
_DECORATED_COLUMNS = {}
_DECORATED_INJECTABLES = {}


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
        _DECORATED_STEPS[name] = func

        orca.add_step(name, func)

        return func

    return decorator


def table():
    def decorator(func):
        name = func.__name__

        logger.debug("inject table %s" % name)

        assert not _DECORATED_TABLES.get(name, False), (
            "table '%s' already decorated." % name
        )
        _DECORATED_TABLES[name] = func

        orca.add_table(name, func)

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


def merge_tables(target, tables, columns=None):
    return orca.merge_tables(target, tables, columns)


def add_step(name, func):
    return orca.add_step(name, func)


def add_table(table_name, table, replace=False):
    """
    Add new table and raise assertion error if the table already exists.
    Silently replace if replace=True.
    """
    if (
        not replace
        and orca.is_table(table_name)
        and orca.table_type(table_name) == "dataframe"
    ):
        logger.warning("inject add_table replacing existing table %s" % table_name)
        assert False

    # FIXME - should add table.copy() instead, so it can't be modified behind our back?
    return orca.add_table(table_name, table, cache=False)


# fixme remove?
def add_column(table_name, column_name, column, cache=False):
    return orca.add_column(table_name, column_name, column, cache=cache)


def add_injectable(name, injectable, cache=False):
    return orca.add_injectable(name, injectable, cache=cache)


def broadcast(
    cast, onto, cast_on=None, onto_on=None, cast_index=False, onto_index=False
):
    return orca.broadcast(
        cast,
        onto,
        cast_on=cast_on,
        onto_on=onto_on,
        cast_index=cast_index,
        onto_index=onto_index,
    )


def get_table(name, default=_NO_DEFAULT):

    if orca.is_table(name) or default == _NO_DEFAULT:
        return orca.get_table(name)
    else:
        return default


def is_injectable(name):

    return orca.is_injectable(name)


def get_injectable(name, default=_NO_DEFAULT):

    if is_injectable(name) or default == _NO_DEFAULT:
        return orca.get_injectable(name)
    else:
        return default


def remove_injectable(name):

    orca._INJECTABLES.pop(name, None)


def reinject_decorated_tables():
    """
    reinject the decorated tables (and columns)
    """

    logger.info("reinject_decorated_tables")

    # need to clear any non-decorated tables that were added during the previous run
    orca._TABLES.clear()
    orca._COLUMNS.clear()
    orca._TABLE_CACHE.clear()
    orca._COLUMN_CACHE.clear()

    for name, func in _DECORATED_TABLES.items():
        logger.debug("reinject decorated table %s" % name)
        orca.add_table(name, func)

    for column_key, args in _DECORATED_COLUMNS.items():
        table_name, column_name = column_key
        logger.debug("reinject decorated column %s.%s" % (table_name, column_name))
        orca.add_column(table_name, column_name, args["func"], cache=args["cache"])

    for name, args in _DECORATED_INJECTABLES.items():
        logger.debug("reinject decorated injectable %s" % name)
        orca.add_injectable(name, args["func"], cache=args["cache"])


def clear_cache():
    return orca.clear_cache()


def set_step_args(args=None):

    assert isinstance(args, dict) or args is None
    orca.add_injectable("step_args", args)


def get_step_arg(arg_name, default=_NO_DEFAULT):

    args = orca.get_injectable("step_args")

    assert isinstance(args, dict)
    if arg_name not in args and default == _NO_DEFAULT:
        raise "step arg '%s' not found and no default" % arg_name

    return args.get(arg_name, default)


def dump_state():

    print("_DECORATED_STEPS", list(_DECORATED_STEPS.keys()))
    print("orca._STEPS", list(orca._STEPS.keys()))
