
# Running the SANDAG multiple-zone system test
  - Download the SANDAG test data from MTC's ActivitySim Box account
  - Run `python import_data.py` to convert the SANDAG test data to ActivitySim format
    - update script line: `folder = "C:/projects/sandag-asim/toRSG/"`
    - update script line: `output_folder = "C:/projects/sandag-asim/toRSG/output/"`
  - From the example_multi folder, run `python simulation.py > log.txt` to run the tests
    - update script line: `DATA_REPO = "C:/projects/sandag-asim/toRSG/output/"`
    - update script line: `vectorTestSize = 10000` to set the query vector length
  - Review results in `log.txt`
  - `create_examples.py` is not required for running the example; it is used to create subsets of data for testing
