# create bat file to run all examples from the examples folder
# copies commented out lines to a bat file for running
# python create_run_all_examples.py > run_all_examples.bat
# ben.stabler@rsginc.com 04/05/21

runnable_line_signature = "  # "  # yes, hacky for now
examples_file_name = "example_manifest.yaml"

example_file = open(examples_file_name, "r")
lines = example_file.readlines()
for line in lines:
    if runnable_line_signature in line:
        runnable_line = line.replace(runnable_line_signature, "").replace("\n", "")
        print(runnable_line)
