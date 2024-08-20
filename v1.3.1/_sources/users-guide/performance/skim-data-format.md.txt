# Skims Data File Formatting

ActivitySim formally uses the [open matrix (OMX) data format](https://github.com/osPlanning/omx/wiki)
for skims data. This format is a widely accepted data interchange format for
transportation models. The format is based on the open-source file storage
technology [HDF5](https://www.hdfgroup.org/solutions/hdf5/).

When using the OMX format "[i]t is strongly recommended, but not required, that
OMX files be compressed using the 'zlib' compression filter with compression level 1."
([ref](https://github.com/osPlanning/omx/wiki/Specification#hdf5-attributes)).
This recommendation is based on the fact that the OMX format is designed to be
widely interoperable, and the zlib compression filter is widely available on many
platforms.

The ActivitySim consortium has found that, while widely available, the zlib compression
filter can be slow to read and write, and can be a bottleneck in the runtime performance
of ActivitySim models.  Larger regional models have complex skims data that can be
100 gigabyte in size or more, and the time to read and write these files can be significant.

ActivitySim models are typically only ever run on "standard" hardware,
which uses Windows, MacOS, or Linux operating systems.  This means that the zlib
compression filter is not strictly necessary for interoperability, as many other faster
and more efficient compression technologies are available on these platforms.
A recently standardized such technology is the
"[blosc2:zstd](https://www.blosc.org/posts/zstd-has-just-landed-in-blosc/)"
compression filter, which performs significantly better than zlib in typical
transportation modeling applications, and similar to zlib, is an open-source technology.

The ActivitySim consortium recommends storing skim data in the OMX format
using "blosc2:zstd" compression instead of "zlib".  This is a high-performance compression filter
that is available on all major platforms, and is significantly faster than zlib. Ideally, the
model developer should configure the network assignment or skimming tool used to produce
skims to directly write skims data files using the "blosc2:zstd" compression filter, which
should improve the runtime performance of that tool as well.  If this is not possible,
standard "zlib" format skims can be converted to "blosc2:zstd" compression using the
"[wring](https://jpn--.github.io/wring/)" command line tool, although such conversion
will likely take almost as much time as simply using "zlib" skims.
