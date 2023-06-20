from __future__ import annotations


class PipelineError(ValueError):
    """General class for errors in using a Pipeline."""


class StateAccessError(PipelineError):
    """Error trying to access a pipeline feature that is not yet initialized."""


class TableTypeError(TypeError):
    """Unable to return data in the format requested."""


class DuplicateWorkflowNameError(ValueError):
    """More than one workflow function is defined with the same name"""


class DuplicateWorkflowTableError(ValueError):
    """More than one loadable table is defined with the same name"""


class DuplicateLoadableObjectError(ValueError):
    """More than one loadable object is defined with the same name"""


class SettingsFileNotFoundError(FileNotFoundError):
    def __init__(self, file_name, configs_dir):
        self.file_name = file_name
        self.configs_dir = configs_dir

    def __str__(self):
        return repr(f"Settings file '{self.file_name}' not found in {self.configs_dir}")


class CheckpointFileNotFoundError(FileNotFoundError):
    """The checkpoints file is not found."""


class CheckpointNameNotFoundError(KeyError):
    """The checkpoint_name is not found."""


class TableNameNotFound(KeyError):
    """The table_name is not found."""


class MissingNameError(KeyError):
    """The name is not found."""


class ReadOnlyError(IOError):
    """This object is read-only."""


class MissingInputTableDefinition(RuntimeError):
    """An input table definition was expected but not found."""
