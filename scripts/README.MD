
## Scripts
Potentially out-of-date scripts used for various data processing activities:
  - stack_mode_choice.py - reformat the very complicated MTC TM1 mode choice UEC Excel files into a more straightforward csv file.  Unfortunately, there were some manual steps - for tour mode choice you have to run it twice since there are two slightly different variable sets that seem to occur.  Hopefully we never will have to do this again so I'm not documenting this throughly.  Not sure I could if I wanted to.
  - stack_mode_choice2.py - has a couple of custom edits for trip_mode_choice rather than tour_mode choice.
  - data_mover.ipynb - create HDF5 example data store from raw CSV files
  - build_omx.py - create one big OMX file for all the MTC TM1 skims.  Requires skim_manifest.csv.
  - create_sf_example.py - create SF county only MTC TM1 example inputs - land use, syn pop, and skims - for testing the entire system with full functionality but less memory requirements.
  - SandagNetworkLOS.py - convert SANDAG network los files on MTC box account to ActivitySim NetworkLOS format