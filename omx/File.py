# OMX package
# release 1

import numpy as np
import tables  # requires pytables >= 2.4

from Exceptions import *


class File(tables.File):

    def __init__(self, f,m,t,r,f1, **kwargs):
        tables.File.__init__(self,f,m,t,r,f1,**kwargs)
        self._shape = None

    def version(self):
        if 'OMX_VERSION' in self.root._v_attrs:
            return self.root._v_attrs['OMX_VERSION']
        else:
            return None


    def createMatrix(self, name, atom=None, shape=None, title='', filters=None,
                     chunkshape=None, byteorder=None, createparents=False, obj=None,
                     attrs=None):
        """Create OMX Matrix (CArray) at root level. User must pass in either
           an existing numpy matrix, or a shape and an atom type."""

        # If object was passed in, make sure its shape is correct
        if self.shape() != None and obj != None and obj.shape != self.shape():
            raise ShapeError('%s has shape %s but this file requires shape %s' %
                (name, obj.shape, self.shape()))
                
        # Create the HDF5 array
        if tables.__version__.startswith('3'):
            matrix = self.createCArray(self.root.data, name, atom, shape, title, filters,
                                       chunkshape, byteorder, createparents, obj)
        else:
            # this version is tables 2.4-compatible:
            matrix = self.createCArray(self.root.data, name, atom, shape, title, filters,
                                       chunkshape, byteorder, createparents)
            if (obj != None):
                matrix[:] = obj

        # Store shape if we don't have one yet
        if self._shape == None:
            storeshape = np.array([matrix.shape[0],matrix.shape[1]], dtype='int32')
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
            realshape = (array[0],array[1])
            self._shape = realshape
            return self._shape

        # Inspect the first CArray object to determine its shape
        if len(self) > 0:
            self._shape = self.iterNodes(self.root.data,'CArray').next().shape

            # Store it if we can
            if self._isWritable():
                storeshape = np.array(
                    [self._shape[0],self._shape[1]],
                    dtype='int32')
                self.root._v_attrs['SHAPE'] = storeshape

            return self._shape

        else:
            return None


    def listMatrices(self):
        """Return list of Matrix names in this File"""
        return [node.name for node in self.listNodes(self.root.data,'CArray')]


    def listAllAttributes(self):
        """Return combined list of all attributes used for any Matrix in this File"""
        all_tags = set()
        for m in self.listNodes(self.root,'CArray'):
            if m.attrs != None:
                all_tags.update(m.attrs._v_attrnames)
        return sorted(list(all_tags))


    # MAPPINGS -----------------------------------------------
    def listMappings(self):
        try:
            return [m.name for m in self.listNodes(self.root.lookup)]
        except:
            return []


    def deleteMapping(self, title):
        try:
            self.removeNode(self.root.lookup, title)
        except:
            raise LookupError('No such mapping: '+title)


    def mapping(self, title):
        """Return dict containing key:value pairs for specified mapping. Keys
           represent the map item and value represents the array offset."""
        try:
            # fetch entries
            entries = []
            entries.extend(self.getNode(self.root.lookup, title)[:])

            # build reverse key-lookup
            keymap = {}
            for i in range(len(entries)):
                keymap[entries[i]] = i

            return keymap

        except:
            raise LookupError('No such mapping: '+title)

    def mapentries(self, title):
        """Return entries[] with key for each array offset."""
        try:
            # fetch entries
            entries = []
            entries.extend(self.getNode(self.root.lookup, title)[:])

            return (keymap,entries)

        except:
            raise LookupError('No such mapping: '+title)


    def createMapping(self, title, entries, overwrite=False):
        """Create an equivalency index, which maps a raw data dimension to
           another integer value. Once created, mappings can be referenced by
           offset or by key."""

        # Enforce shape-checking
        if self.shape():
            if not len(entries) in self._shape:
                raise ShapeError('Mapping must match one data dimension')

        # Handle case where mapping already exists:
        if title in self.listMappings():
            if overwrite:
                self.deleteMapping(title)
            else:
                raise LookupError(title+' mapping already exists.')

        # Create lookup group under root if it doesn't already exist.
        if 'lookup' not in self.root:
            self.createGroup(self.root, 'lookup')

        # Write the mapping!
        mymap = self.createArray(self.root.lookup, title, atom=tables.UInt16Atom(),
                                 shape=(len(entries),) )
        mymap[:] = entries

        return mymap


    # The following functions implement Python list/dictionary lookups. ----
    def __getitem__(self,key):
        """Return a matrix by name, or a list of matrices by attributes"""

        if isinstance(key, str):
            return self.getNode(self.root.data, key)

        if 'keys' not in dir(key):
            raise LookupError('Key %s not found' % key)

        # Loop through key/value pairs
        mats = self.listNodes(self.root.data, 'CArray')
        for a in key.keys():
            mats = self._getMatricesByAttribute(a, key[a], mats)

        return mats


    def _getMatricesByAttribute(self, key, value, matrices=None):

        answer = []

        if matrices==None:
            matrices = self.listNodes(self.root.data,'CArray')

        for m in matrices:
            if m.attrs == None: continue

            # Only test if key is present in matrix attributes
            if key in m.attrs._v_attrnames and m.attrs[key] == value:
                answer.append(m)

        return answer


    def __len__(self):
        return len(self.listNodes(self.root.data, 'CArray'))


    def __setitem__(self, key, dataset):
        # We need to determine atom and shape from the object that's been passed in.
        # This assumes 'dataset' is a numpy object.
        atom = tables.Atom.from_dtype(dataset.dtype)
        shape = dataset.shape

        #checks to see if it is already a tables instance, and if so, just copies it
        if dataset.__class__.__name__ == 'CArray':
            return dataset.copy(self.root.data, key)
        else:
            return self.createMatrix(key, atom, shape, obj=dataset)


    def __delitem__(self, key):
        self.removeNode(self.root.data, key)


    def __iter__(self):
        """Iterate over the keys in this container"""
        return self.iterNodes(self.root.data, 'CArray')


    def __contains__(self, item):
        return item in self.root.data._v_children

