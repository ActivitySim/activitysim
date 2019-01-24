
## Scripts

### Create Asim TM1 inputs from CT-RAMP TM1 model data

  - Open a DOS prompt in the mtc tm1 skims folder
  - Run mtc_tm1_omx_export.s to convert Cube matrices to OMX (you need Cube 6.4.3 or 6.4.2 + OMXLib-x64.dll)
  - Run build_omx.py to build one OMX file. Requires skim_manifest.csv.
  - Run mtc_inputs.py to build mtc_asim.h5 (taz data and syn pop files)

### Other scripts
  - stack_mode_choice.py - reformat the MTC TM1 mode choice UEC Excel files into a more straightforward csv file.  Unfortunately, there were some manual steps - for tour mode choice you have to run it twice since there are two slightly different variable sets that seem to occur. (this script is out-of-date)
  - stack_mode_choice2.py - has a couple of custom edits for trip_mode_choice rather than tour_mode choice. (this script is out-of-date)
  - create_sf_example.py - create SF county only MTC TM1 example inputs - land use, syn pop, and skims - for testing the entire system with full functionality but less memory requirements.
  - make_pipeline_output.py - create table of pipeline table fields by creator for the rst docs
  - create_abmviz_inputs.py - create abmviz input files (this script is not yet complete)