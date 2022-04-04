from . import archive_outputs
from . import cmd
from . import py

from pypyr.context import Context
from pypyr.errors import KeyNotInContextError

def get_formatted_or_default(self:Context, key:str, default):
    try:
        return self.get_formatted(key)
    except (KeyNotInContextError, KeyError):
        return default

Context.get_formatted_or_default = get_formatted_or_default
