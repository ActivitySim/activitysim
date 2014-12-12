# Licensed under the Apache License, v2.0
# See activitysim/omx/LICENSE.txt
# Modified from the original OMX library available at
# https://github.com/osPlanning/omx/tree/dev/api/python/omx

import numpy as np
import tables

from .exceptions import ShapeError

OMX_VERSION = '0.2'


class OMXFile(tables.File):
    def __init__(self, *args, **kwargs):
        super(OMXFile, self).__init__(*args, **kwargs)
        self._shape = None

    def version(self):
        if 'OMX_VERSION' in self.root._v_attrs:
            return self.root._v_attrs['OMX_VERSION']
        else:
            return None

    def create_matrix(
            self, name, atom=None, shape=None, title='', filters=None,
            chunkshape=None, byteorder=None, createparents=False, obj=None,
            attrs=None):
        """Create OMX Matrix (CArray) at root level. User must pass in either
           an existing numpy matrix, or a shape and an atom type."""

        # If object was passed in, make sure its shape is correct
        if (self.shape() is not None and
                obj is not None and
                obj.shape != self.shape()):
            raise ShapeError(
                '%s has shape %s but this file requires shape %s' %
                (name, obj.shape, self.shape()))

        # Create the HDF5 array
        if tables.__version__.startswith('3'):
            matrix = self.create_carray(
                self.root.data, name, atom, shape, title, filters,
                chunkshape, byteorder, createparents, obj)

        # Store shape if we don't have one yet
        if self._shape is None:
            storeshape = np.array(matrix.shape, dtype='int32')
            self.root._v_attrs['SHAPE'] = storeshape
            self._shape = matrix.shape

        # attributes
        if attrs:
            for key in attrs:
                matrix.attrs[key] = attrs[key]

        return matrix

    def shape(self):
        """Return the one and only shape of all matrices in this File"""

        # If we already have the shape, just return it
        if self._shape:
            return self._shape

        # If shape is already set in root node attributes, grab it
        if 'SHAPE' in self.root._v_attrs:
            # Shape is stored as a numpy.array:
            arrayshape = self.root._v_attrs['SHAPE']
            # which must be converted to a tuple:
            realshape = tuple(arrayshape)
            self._shape = realshape
            return self._shape

        # Inspect the first CArray object to determine its shape
        if len(self) > 0:
            self._shape = self.iter_nodes(
                self.root.data, 'CArray').next().shape

            # Store it if we can
            if self._iswritable():
                storeshape = np.array(self._shape, dtype='int32')
                self.root._v_attrs['SHAPE'] = storeshape

            return self._shape

        else:
            return None

    def list_matrices(self):
        """Return list of Matrix names in this File"""
        return [
            node.name for node in self.list_nodes(self.root.data, 'CArray')]

    def list_all_attributes(self):
        """
        Return combined list of all attributes used for
        any Matrix in this File

        """
        all_tags = set()
        for m in self.iter_nodes(self.root.data, 'CArray'):
            all_tags.update(m.attrs._v_attrnamesuser)
        return sorted(all_tags)

    # MAPPINGS -----------------------------------------------
    def list_mappings(self):
        try:
            return [m.name for m in self.list_nodes(self.root.lookup)]
        except:
            return []

    def delete_mapping(self, title):
        try:
            self.remove_node(self.root.lookup, title)
        except:
            raise LookupError('No such mapping: ' + title)

    def mapping(self, title):
        """Return dict containing key:value pairs for specified mapping. Keys
           represent the map item and value represents the array offset."""
        try:
            # fetch entries
            entries = []
            entries.extend(self.get_node(self.root.lookup, title)[:])

            # build reverse key-lookup
            keymap = {}
            for i in range(len(entries)):
                keymap[entries[i]] = i

            return keymap

        except:
            raise LookupError('No such mapping: ' + title)

    def create_mapping(self, title, entries, overwrite=False):
        """Create an equivalency index, which maps a raw data dimension to
           another integer value. Once created, mappings can be referenced by
           offset or by key."""

        # Enforce shape-checking
        if self.shape():
            if not len(entries) in self._shape:
                raise ShapeError('Mapping must match one data dimension')

        # Handle case where mapping already exists:
        if title in self.list_mappings():
            if overwrite:
                self.delete_mapping(title)
            else:
                raise LookupError(title + ' mapping already exists.')

        # Create lookup group under root if it doesn't already exist.
        if 'lookup' not in self.root:
            self.create_group(self.root, 'lookup')

        # Write the mapping!
        mymap = self.create_array(
            self.root.lookup, title, atom=tables.UInt16Atom(),
            shape=(len(entries),))
        mymap[:] = entries

        return mymap

    def __getitem__(self, key):
        """Return a matrix by name, or a list of matrices by attributes"""
        return self.get_node(self.root.data, key)

    def get_matrices_by_attr(self, key, value):
        """
        Returns a list of matrices that have an attribute matching
        a certain value.

        Parameters
        ----------
        key : str
            Attribute name to match.
        value : object
            Attribute value to match.

        Returns
        -------
        matrices : list

        """
        return [
            m for m in self
            if key in m.attrs and m.attrs[key] == value]

    def __len__(self):
        return len(self.list_nodes(self.root.data, 'CArray'))

    def __setitem__(self, key, dataset):
        # checks to see if it is already a tables instance, and if so,
        # copies it
        if isinstance(dataset, tables.CArray):
            return dataset.copy(self.root.data, key)
        else:
            # We need to determine atom and shape from the object that's
            # been passed in.
            # This assumes 'dataset' is a numpy object.
            atom = tables.Atom.from_dtype(dataset.dtype)
            shape = dataset.shape

            return self.create_matrix(key, atom, shape, obj=dataset)

    def __delitem__(self, key):
        self.remove_node(self.root.data, key)

    def __iter__(self):
        """Iterate over the keys in this container"""
        return self.iter_nodes(self.root.data, 'CArray')

    def __contains__(self, item):
        return item in self.root.data


def open_omxfile(
        filename, mode='r', title='', root_uep='/',
        filters=tables.Filters(
            complevel=1, shuffle=True, fletcher32=False, complib='zlib'),
        shape=None, **kwargs):
    """Open or create a new OMX file. New files will be created with default
       zlib compression enabled."""

    f = OMXFile(filename, mode, title, root_uep, filters, **kwargs)

    # add omx structure if file is writable
    if mode != 'r':
        # version number
        if 'OMX_VERSION' not in f.root._v_attrs:
            f.root._v_attrs['OMX_VERSION'] = OMX_VERSION

        # shape
        if shape:
            storeshape = np.array(shape, dtype='int32')
            f.root._v_attrs['SHAPE'] = storeshape

        # /data and /lookup folders
        if 'data' not in f.root:
            f.create_group(f.root, "data")
        if 'lookup' not in f.root:
            f.create_group(f.root, "lookup")

    return f
