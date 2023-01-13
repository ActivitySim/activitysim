from typing import Any, Union  # noqa: F401

try:
    from pydantic import BaseModel as PydanticBase
except ModuleNotFoundError:

    class PydanticBase:
        pass
