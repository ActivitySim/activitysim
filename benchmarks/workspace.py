# # -*- coding: utf-8 -*-
# # import os
# # import ruamel.yaml
# # from pathlib import Path
# #
# #
# # class Workspace:
# #     """
# #     Stores information about the activitysim benchmarking workspace in
# #     ~/.asv-activitysim.yaml file.
# #     """
# #
# #     def __init__(self, directory=None):
# #         self.directory = directory
# #
# #     @staticmethod
# #     def get_cfg_file_path():
# #         return os.path.expanduser('~/.asv-activitysim.yaml')
# #
# #     @classmethod
# #     def load(cls, path=None):
# #         settings = {}
# #         if path is None:
# #             path = cls.get_cfg_file_path()
# #         if os.path.isfile(path):
# #             yaml = ruamel.yaml.YAML()
# #             settings = yaml.load(Path(path))
# #         self = cls(
# #             directory=settings.get('directory', None)
# #         )
# #         if self.directory:
# #             os.makedirs(self.directory, exist_ok=True)
# #         return self
# #
# #     def save(self, path=None):
# #         if path is None:
# #             path = self.get_cfg_file_path()
# #         yaml = ruamel.yaml.YAML()
# #         yaml.dump({
# #             'directory': self.directory,
# #         }, Path(path))
# #
# #
# # workspace = Workspace.load()
#
# directory = None