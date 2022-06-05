# from . import archive_outputs
from pypyr.context import Context
from pypyr.errors import KeyNotInContextError

from . import cmd, py


def get_formatted_or_default(self: Context, key: str, default):
    try:
        return self.get_formatted(key)
    except (KeyNotInContextError, KeyError):
        return default
    except TypeError:
        return self.get(key)
    except Exception as err:
        raise ValueError(f"extracting {key} from context") from err


Context.get_formatted_or_default = get_formatted_or_default


def get_formatted_or_raw(self: Context, key: str):
    try:
        return self.get_formatted(key)
    except TypeError:
        return self.get(key)
    except Exception as err:
        raise ValueError(f"extracting {key} from context") from err


Context.get_formatted_or_raw = get_formatted_or_raw
