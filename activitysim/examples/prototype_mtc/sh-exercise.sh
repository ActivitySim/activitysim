asys create -e example_mtc_full -d .
cd example_mtc_full

asys run -c configs_sh_compile -c configs -d data -o output

asys run -c configs_sh -c configs_chunktrain -c configs -d data -o output

asys run -c configs_sh -c configs_production -c configs -d data -o output