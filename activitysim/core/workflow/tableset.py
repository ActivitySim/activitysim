#
# import pandas as pd
# import logging
# from typing import Mapping
# from ..exceptions import PipelineAccessError, DuplicateLoadableTableError
#
# logger = logging.getLogger(__name__)
#
#
#
#
# def loadable_table(func):
#     """
#     Decorator for functions that initialize tables.
#
#     The function being decorated should have a single arguments: `whale`.
#
#     Parameters
#     ----------
#     func
#
#     Returns
#     -------
#     func
#     """
#     from ..pipeline import Whale
#     name = func.__name__
#     logger.debug(f"found loadable table {name}")
#     if name in Whale._LOADABLE_TABLES:
#         raise DuplicateLoadableTableError(name)
#     Whale._LOADABLE_TABLES[name] = func
#     return func
#
#
#
# class Tableset:
#
#     def __init__(self):
#         self.tables = {}
#         self.unsaved_tables = set()
#         self.saveable_tables = set()
#
#     # def load_table(self, tablename, overwrite=False, swallow_errors=False):
#     #     if tablename in self.tables and not overwrite:
#     #         if swallow_errors:
#     #             return
#     #         raise ValueError(f"table {tablename} already loaded")
#     #     if tablename not in _LOADABLE_TABLES:
#     #         if swallow_errors:
#     #             return
#     #         raise ValueError(f"table {tablename} has no loading function")
#     #     if self.filesystem is None:
#     #         if swallow_errors:
#     #             return
#     #         raise PipelineAccessError("filesystem not attached to tableset")
#     #     if self.settings is None:
#     #         if swallow_errors:
#     #             return
#     #         raise PipelineAccessError("settings not attached to tableset")
#     #     logger.debug(f"loading table {tablename}")
#     #     t = _LOADABLE_TABLES[tablename](self, self.filesystem, self.settings)
#     #     self.store_data(tablename, t)
#     #     return t
#
#     #
#     # def get_frame(self, tablename):
#     #     t = self.tables.get(tablename, None)
#     #     if t is None:
#     #         t = self.load_table(tablename, swallow_errors=True)
#     #     if t is None:
#     #         raise KeyError(tablename)
#     #     if isinstance(t, pd.DataFrame):
#     #         return t
#     #     raise TypeError(f"cannot convert {tablename} to DataFrame")
#
#     def store_data(self, name, data, saveable=True):
#         self.tables[name] = data
#         if saveable or name in self.saveable_tables:
#             self.saveable_tables.add(name)
#             self.unsaved_tables.add(name)
#
#     def update(self, other, all_saveable=False):
#         if isinstance(other, Tableset):
#             for tablename, t in other.tables.items():
#                 is_saveable = tablename in self.saveable_tables or tablename in other.saveable_tables
#                 self.store_data(tablename, t, saveable=is_saveable)
#         elif isinstance(other, Mapping):
#             for tablename, t in other.items():
#                 is_saveable = all_saveable or (tablename in self.saveable_tables)
#                 self.store_data(tablename, t, saveable=is_saveable)
