# import orca
# import logging
#
# from ..pipeline import Whale
# from ..exceptions import DuplicateLoadableTableError
#
# logger = logging.getLogger(__name__)
#
# def _injectable(cache=False):
#     def decorator(func):
#         name = func.__name__
#         logger.debug(f"found loadable object {name}")
#         if name in Whale._LOADABLE_OBJECTS:
#             raise DuplicateLoadableTableError(name)
#         Whale._LOADABLE_OBJECTS[name] = (func, cache)
#         return func
#
#     return decorator
#
