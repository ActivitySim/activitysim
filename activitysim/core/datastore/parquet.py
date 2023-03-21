from __future__ import annotations

import gzip
import logging
import os
import pickle
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import xarray as xr
import yaml
from sharrow import DataTree, Relationship
from sharrow.dataset import construct, from_zarr_with_attr

from activitysim.core.datastore._base import CheckpointStore, timestamp
from activitysim.core.exceptions import ReadOnlyError

logger = logging.getLogger(__name__)

# name of the checkpoint dict keys
# (which are also columns in the checkpoints dataframe stored in hte pipeline store)
TIMESTAMP = "timestamp"
CHECKPOINT_NAME = "checkpoint_name"
NON_TABLE_COLUMNS = [CHECKPOINT_NAME, TIMESTAMP]

# name used for storing the checkpoints dataframe to the pipeline store
CHECKPOINT_TABLE_NAME = "checkpoints"

LAST_CHECKPOINT = "_"

# name of the first step/checkpoint created when the pipeline is started
INITIAL_CHECKPOINT_NAME = "init"
FINAL_CHECKPOINT_NAME = "final"


def _read_parquet(filename, index_col=None) -> xr.Dataset:
    import pyarrow.parquet as pq
    from sharrow.dataset import from_table

    if not isinstance(filename, (Path, zipfile.Path)):
        filename = Path(filename)
    try:
        with filename.open("rb") as f:
            content = pq.read_table(f)
    except FileNotFoundError:
        if isinstance(filename, zipfile.Path):
            filename2 = zipfile.Path(
                filename.root, Path(filename.at).with_suffix(".pickle.gz").as_posix()
            )
        else:
            filename2 = filename.with_suffix(".pickle.gz")
        if filename2.exists():
            with filename2.open("rb") as f:
                return pickle.loads(gzip.decompress(f.read()))
        else:
            raise
    if index_col is not None:
        index = content.column(index_col)
        content = content.drop([index_col])
    else:
        index = None
    x = from_table(content, index=index, index_name=index_col or "index")
    return x


class ParquetStore(CheckpointStore):
    """Storage interface for parquet-based table storage.

    This interface will fall back to storing tables in a gzipped pickle if
    the parquet format fails (as might happen if datatypes for some columns
    are not homogenous and values are stored as "object").
    """

    extension = ".pipeline"
    metadata_filename: str = "metadata.yaml"
    checkpoint_subdir: str = "checkpoints"
    LATEST = "_"

    @staticmethod
    def _to_parquet(df: pd.DataFrame, filename, *args, **kwargs):
        try:
            df.to_parquet(filename, *args, **kwargs)
        except (pa.lib.ArrowInvalid, pa.lib.ArrowTypeError) as err:
            logger.error(
                f"Problem writing to {filename}\n" f"{err}\n" f"falling back to pickle"
            )
            # fallback to pickle, compatible with more dtypes
            df.to_pickle(str(Path(filename).with_suffix(".pickle.gz")))

    def __init__(
        self,
        directory: Path,
        mode: str = "a",
        *,
        gitignore: bool = True,
        complib: str = "NOTSET",
    ):
        """Initialize a storage interface for parquet-based table storage.

        Parameters
        ----------
        directory : Path
            The file directory for this ParquetStore. If this location does not
            include a ".pipeline" or ".zip" suffix, one is added.
        mode : {"a", "r"}, default "a"
            Mode to open this store, "a"ppend or "r"ead-only.  Zipped stores
            can only be opened in read-only mode.
        gitignore : bool, default True
            If not opened in read-only mode, should a ".gitignore" file be added
            with a global wildcard (**)?  Doing so will help prevent this store
            from being accidentally committed to git.
        """
        super().__init__(mode=mode)
        directory = Path(directory)
        if directory.suffix == ".zip":
            if mode != "r":
                raise ValueError("can only open a Zip parquet store as read-only.")
        elif directory.suffix != self.extension:
            directory = directory.with_suffix(self.extension)
        self._directory = directory
        self.complib = complib
        if self._mode != "r":
            self._directory.mkdir(parents=True, exist_ok=True)
            if gitignore and not self._directory.joinpath(".gitignore").exists():
                self._directory.joinpath(".gitignore").write_text("**\n")
        self._checkpoints = {}
        self._checkpoint_order = []

    @property
    def filename(self) -> Path:
        """The directory location of this ParquetStore."""
        return self._directory

    def _store_table_path(self, table_name, checkpoint_name):
        if checkpoint_name:
            return self._directory.joinpath(table_name, f"{checkpoint_name}.parquet")
        else:
            return self._directory.joinpath(f"{table_name}.parquet")

    def write_data(
        self,
        name: str,
        checkpoint_name: str,
        data: xr.Dataset | pd.DataFrame,
        overwrite: bool = True,
    ):
        if self.is_readonly:
            raise ReadOnlyError("store is read-only")
        if isinstance(data, pd.DataFrame):
            data = construct(data)
        if len(data.dims) == 1:
            target = self._parquet_name(name, checkpoint_name)
            if overwrite and target.is_file():
                os.unlink(target)
            if overwrite and target.with_suffix(".pickle.gz").is_file():
                os.unlink(target.with_suffix(".pickle.gz"))
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                if self.complib == "NOTSET":
                    data.single_dim.to_parquet(str(target))
                else:
                    data.single_dim.to_parquet(str(target), compression=self.complib)
            except (pa.lib.ArrowInvalid, pa.lib.ArrowTypeError) as err:
                logger.error(
                    f"Problem writing to {target}\n"
                    f"{err}\n"
                    f"falling back to pickle"
                )
                # fallback to pickle, compatible with more dtypes

                with gzip.open(target.with_suffix(".pickle.gz"), "w") as f:
                    pickle.dump(data, f)
        else:
            # zarr is used if ndim > 1
            target = self._zarr_subdir(name, checkpoint_name)
            if overwrite and target.is_dir():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
            data.to_zarr_with_attr(target)

    def list_checkpoint_names(self) -> list[str]:
        """Get a list of all checkpoint names in this store."""
        if not self._checkpoint_order:
            self.read_metadata()
        return list(self._checkpoint_order)

    @property
    def is_readonly(self) -> bool:
        return self._mode == "r"

    @property
    def is_open(self) -> bool:
        return self._directory is not None and self._directory.is_dir()

    def close(self) -> None:
        """Close this store."""
        pass

    def make_zip_archive(self, output_filename) -> Path:
        """
        Compress this pipeline into a zip archive.

        Parameters
        ----------
        output_filename

        Returns
        -------
        Path
            Filename of the resulting zipped store.
        """
        output_filename = Path(output_filename)
        if output_filename.suffix != ".zip":
            output_filename = output_filename.with_suffix(".zip")
        with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self._directory):
                files = [f for f in files if not f[0] == "."]
                for f in files:
                    arcname = Path(root).joinpath(f).relative_to(self._directory)
                    zipf.write(Path(root).joinpath(f), arcname=arcname)

        return output_filename

    def wipe(self):
        """
        Remove this store, including all subdirectories.
        """
        if self.is_readonly:
            raise ReadOnlyError("store is readonly")
        walked = list(os.walk(self._directory))
        while walked:
            root, dirs, files = walked.pop(-1)
            for f in files:
                if f.endswith(".parquet"):
                    os.unlink(os.path.join(root, f))
            # after removing all parquet files, is this directory basically empty?
            should_drop_root = True
            file_list = {f for f in Path(root).glob("**/*") if f.is_file()}
            for f in file_list:
                if f not in {".gitignore", ".DS_Store"}:
                    should_drop_root = False
            if should_drop_root:
                os.rmdir(root)

    def _zarr_subdir(self, table_name, checkpoint_name):
        if self.filename.suffix == ".zip":
            basepath = zipfile.Path(self.filename)
        else:
            basepath = self.filename
        rel_path = Path(table_name, checkpoint_name).with_suffix(".zarr")
        return basepath.joinpath(rel_path)

    def _parquet_name(self, table_name, checkpoint_name):
        if self.filename.suffix == ".zip":
            basepath = zipfile.Path(self.filename)
        else:
            basepath = self.filename
        rel_path = Path(table_name, checkpoint_name).with_suffix(".parquet")
        return basepath.joinpath(rel_path)

    def make_checkpoint(self, checkpoint_name: str, overwrite: bool = True) -> None:
        if self._mode == "r":
            raise ReadOnlyError
        to_be_checkpointed = self.to_be_checkpointed()
        new_checkpoint = {
            "timestamp": timestamp(),
            "tables": {},
            "relationships": [],
        }
        # remove checkpoint name from ordered list if it already exists
        while checkpoint_name in self._checkpoint_order:
            self._checkpoint_order.remove(checkpoint_name)
        # add checkpoint name at end ordered list
        self._checkpoint_order.append(checkpoint_name)
        for table_name, table_data in to_be_checkpointed.items():
            self.write_data(table_name, checkpoint_name, table_data)
            self._update_dataset(
                table_name, table_data, last_checkpoint=checkpoint_name
            )
        for table_name, table_data in self._tree.subspaces_iter():
            inventory = {"data_vars": {}, "coords": {}}
            for varname, vardata in table_data.items():
                inventory["data_vars"][varname] = {
                    "last_checkpoint": vardata.attrs.get("last_checkpoint", "MISSING"),
                    "dtype": str(vardata.dtype),
                }
            for varname, vardata in table_data.coords.items():
                _cp = checkpoint_name
                # coords in every checkpoint with any content
                if table_name not in to_be_checkpointed:
                    _cp = vardata.attrs.get("last_checkpoint", "MISSING")
                inventory["coords"][varname] = {
                    "last_checkpoint": _cp,
                    "dtype": str(vardata.dtype),
                }
            new_checkpoint["tables"][table_name] = inventory
        for r in self._tree.list_relationships():
            new_checkpoint["relationships"].append(r.to_dict())
        self._checkpoints[checkpoint_name] = new_checkpoint
        self._write_checkpoint(checkpoint_name, new_checkpoint)
        self._write_metadata()

    def _write_checkpoint(self, name: str, checkpoint: dict):
        if self._mode == "r":
            raise ReadOnlyError
        checkpoint_metadata_target = self.filename.joinpath(
            self.checkpoint_subdir, f"{name}.yaml"
        )
        if checkpoint_metadata_target.exists():
            n = 1
            while checkpoint_metadata_target.with_suffix(f".{n}.yaml").exists():
                n += 1
            os.rename(
                checkpoint_metadata_target,
                checkpoint_metadata_target.with_suffix(f".{n}.yaml"),
            )
        checkpoint_metadata_target.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_metadata_target, "w") as f:
            yaml.safe_dump(checkpoint, f)

    def _write_metadata(self):
        if self._mode == "r":
            raise ReadOnlyError
        metadata_target = self.filename.joinpath(self.metadata_filename)
        if metadata_target.exists():
            n = 1
            while metadata_target.with_suffix(f".{n}.yaml").exists():
                n += 1
            os.rename(metadata_target, metadata_target.with_suffix(f".{n}.yaml"))
        with open(metadata_target, "w") as f:
            metadata = dict(
                datastore_format_version=1,
                checkpoint_order=self._checkpoint_order,
            )
            if self.do_not_persist:
                metadata["do_not_persist"] = self.do_not_persist
            yaml.safe_dump(metadata, f)

    def read_metadata(self, checkpoints=None):
        """
        Read storage metadata.

        Parameters
        ----------
        checkpoints : str | list[str], optional
            Read only these checkpoints.  If not provided, only the latest
            checkpoint metadata is read. Set to "*" to read all.
        """
        if self.filename.suffix == ".zip":
            basepath = zipfile.Path(self.filename)
        else:
            basepath = self.filename
        with basepath.joinpath(self.metadata_filename).open() as f:
            metadata = yaml.safe_load(f)
        datastore_format_version = metadata.get("datastore_format_version", "missing")
        if datastore_format_version == 1:
            self._checkpoint_order = metadata["checkpoint_order"]
        else:
            raise NotImplementedError(f"{datastore_format_version=}")
        self.do_not_persist = set(metadata.get("do_not_persist", []))
        if checkpoints is None or checkpoints == self.LATEST:
            checkpoints = [self._checkpoint_order[-1]]
        elif isinstance(checkpoints, str):
            if checkpoints == "*":
                checkpoints = list(self._checkpoint_order)
            else:
                checkpoints = [checkpoints]
        for c in checkpoints:
            with basepath.joinpath(self.checkpoint_subdir, f"{c}.yaml").open() as f:
                self._checkpoints[c] = yaml.safe_load(f)

    def restore_checkpoint(self, checkpoint_name: str = LAST_CHECKPOINT):
        if checkpoint_name == LAST_CHECKPOINT:
            if not self._checkpoint_order:
                self.read_metadata()
            if not self._checkpoint_order:
                raise ValueError("no checkpoints found")
            checkpoint_name = self._checkpoint_order[-1]
        if checkpoint_name not in self._checkpoints:
            try:
                self.read_metadata(checkpoint_name)
            except FileNotFoundError as err:
                raise KeyError(checkpoint_name) from err
        checkpoint = self._checkpoints[checkpoint_name]
        self._tree = DataTree(root_node_name=False)
        for table_name, table_def in checkpoint["tables"].items():
            if table_name == "timestamp":
                continue
            t = xr.Dataset()
            opened_targets = {}
            coords = table_def.get("coords", {})
            if len(coords) == 1:
                index_name = list(coords)[0]
            else:
                index_name = None
            for coord_name, coord_def in coords.items():
                target = self._zarr_subdir(table_name, coord_def["last_checkpoint"])
                key = str(target)
                if key not in opened_targets:
                    if target.exists():
                        opened_targets[key] = from_zarr_with_attr(target)
                    else:
                        # zarr not found, try parquet
                        target2 = self._parquet_name(
                            table_name, coord_def["last_checkpoint"]
                        )
                        try:
                            opened_targets[key] = _read_parquet(target2, index_name)
                        except FileNotFoundError as err:
                            raise FileNotFoundError(target) from None
                t = t.assign_coords({coord_name: opened_targets[key][coord_name]})
            data_vars = table_def.get("data_vars", {})
            for var_name, var_def in data_vars.items():
                if var_def["last_checkpoint"] == "MISSING":
                    raise ValueError(f"missing checkpoint for {table_name}.{var_name}")
                target = self._zarr_subdir(table_name, var_def["last_checkpoint"])
                key = str(target)
                if key not in opened_targets:
                    if target.exists():
                        opened_targets[key] = from_zarr_with_attr(target)
                    else:
                        # zarr not found, try parquet
                        target2 = self._parquet_name(
                            table_name, var_def["last_checkpoint"]
                        )
                        try:
                            opened_targets[key] = _read_parquet(target2, index_name)
                        except FileNotFoundError:
                            raise FileNotFoundError(target) from None
                t = t.assign({var_name: opened_targets[key][var_name]})
            self._tree.add_dataset(table_name, t)
        for r in checkpoint["relationships"]:
            self._tree.add_relationship(Relationship(**r))
