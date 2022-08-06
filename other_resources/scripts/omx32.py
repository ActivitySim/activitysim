import argparse
import os

import numpy as np
import openmatrix as omx
import pandas as pd

parser = argparse.ArgumentParser(description="crop PSRC raw_data")
parser.add_argument(
    "input", metavar="input_file_name", type=str, nargs=1, help=f"input omx file"
)

parser.add_argument(
    "output", metavar="output_file_name", type=str, nargs=1, help=f"output omx file"
)

args = parser.parse_args()


input_file_name = args.input[0]
output_file_name = args.output[0]


#
# skims
#
skim_data_type = np.float32

omx_in = omx.open_file(input_file_name, "r")
print(f"omx_in shape {omx_in.shape()}")

omx_out = omx.open_file(output_file_name, "w")

for mapping_name in omx_in.listMappings():
    offset_map = omx_in.mapentries(mapping_name)
    omx_out.create_mapping(mapping_name, offset_map)


for mat_name in omx_in.list_matrices():

    # make sure we have a vanilla numpy array, not a CArray
    m = np.asanyarray(omx_in[mat_name])
    print(f"{mat_name} {m.shape} {type(m[0,0])}")

    omx_out[mat_name] = m.astype(skim_data_type)


omx_in.close()
omx_out.close()
