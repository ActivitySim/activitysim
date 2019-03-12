# Full dataset skims

2732722 households in full dataset

# - 10% sample
households_sample_size: 2875192
2875192
chunk_size: 1500000000
num_processes: 1

# 10% sample OSX single process

#python simulation.py -d ancillary_data -d /Users/jeff.doyle/work/activitysim-data/mtc_tm1/data -c configs -c ../example/configs
python simulation.py

# 10% sample OSX single process - 3 shadow price iterations

python simulation.py -o output_sp -c configs_sp -c ../example/configs

# 100% sample OSX single process - 10 shadow price iterations ctramp method

python simulation.py -o output_sp_ctramp -c configs_sp_ctramp -c ../example/configs

# 100% sample OSX single process - 10 shadow price iterations daysim method

python simulation.py -o output_sp_daysim -c configs_sp_daysim -c ../example/configs