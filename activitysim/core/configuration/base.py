from __future__ import annotations

from typing import Any, Union  # noqa: F401

from activitysim.core import configuration

try:
    from pydantic import BaseModel as PydanticBase
except ModuleNotFoundError:

    class PydanticBase:
        pass


class PydanticReadable(PydanticBase):
    @classmethod
    def read_settings_file(
        cls,
        filesystem: "configuration.FileSystem",
        file_name,
        mandatory=True,
        include_stack=False,
        configs_dir_list=None,
    ) -> PydanticReadable:
        # pass through to read_settings_file, requires validator_class and provides type hinting for IDE's
        return filesystem.read_settings_file(
            file_name,
            mandatory,
            include_stack,
            configs_dir_list,
            validator_class=cls,
        )
