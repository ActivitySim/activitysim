
## Scripts

### Create Asim TM1 inputs from CT-RAMP TM1 model data

  - Open a DOS prompt in the mtc tm1 skims folder
  - Run mtc_tm1_omx_export.s to convert Cube matrices to OMX (you need Cube 6.4.3 or 6.4.2 + OMXLib-x64.dll)
  - Run build_omx.py to build one OMX file. Requires skim_manifest.csv.
  - Run mtc_inputs.py to build mtc_asim.h5 (taz data and syn pop files)

### Other scripts
  - create_sf_example.py - create SF county only MTC TM1 example inputs - land use, syn pop, and skims - for testing the entire system with full functionality but less memory requirements.
  - make_pipeline_output.py - create table of pipeline table fields by creator for the rst docs
  - verify_results.py - compare results for each submodel against TM1 results, see verification page in the wiki
  - create_abmviz_inputs.py - create abmviz input files (this script is not yet complete)