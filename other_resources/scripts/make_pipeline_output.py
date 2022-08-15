# create table of pipeline table fields by creator
# Ben Stabler, ben.stabler@rsginc.com, 06/06/18
# C:\projects\development\activitysim\example>python ../scripts/make_pipeline_output.py

import pandas as pd

pipeline_filename = "output\\pipeline.h5"
out_fields_filename = "output\\pipeline_fields.csv"

# get pipeline tables
pipeline = pd.io.pytables.HDFStore(pipeline_filename)
checkpoints = pipeline["/checkpoints"]
p_tables = pipeline.keys()
p_tables.remove("/checkpoints")
p_tables_tables = [i.split("/")[1] for i in p_tables]
p_tables_cps = [i.split("/")[2] for i in p_tables]
p_tables = pd.DataFrame({"table": p_tables_tables, "cp": p_tables_cps})

# join timestamps and sort
cp_times = checkpoints[["checkpoint_name", "timestamp"]]
cp_times = cp_times.set_index("checkpoint_name")
p_tables = p_tables.join(cp_times, on="cp")
p_tables = p_tables.sort_values(by=["table", "timestamp"])

# build table of fields for each table by creator
# TABLE, FIELD, DTYPE, CREATOR, NCOL, NROW
tables_fields = dict()
for i in range(len(p_tables)):

    cur_table = p_tables["table"].iloc[i]
    cur_cp = p_tables["cp"].iloc[i]

    print("process " + cur_table + "/" + cur_cp)

    cur_table_data = pipeline["/" + cur_table + "/" + cur_cp]
    cur_table_data_fields = cur_table_data.dtypes.index
    cur_table_data_dtypes = cur_table_data.dtypes.values
    cur_table_data_nrow = len(cur_table_data)
    cur_table_data_ncol = len(cur_table_data.columns)

    table_data = pd.DataFrame(
        {
            "TABLE": cur_table,
            "FIELD": cur_table_data_fields,
            "DTYPE": cur_table_data_dtypes,
            "CREATOR": cur_cp,
            "NCOL": cur_table_data_ncol,
            "NROW": cur_table_data_nrow,
        }
    )
    table_data = table_data[
        ["TABLE", "FIELD", "DTYPE", "CREATOR", "NCOL", "NROW"]
    ]  # reorder

    if cur_table not in tables_fields.keys():
        tables_fields[cur_table] = table_data
    else:
        # determine new fields and append to table
        existing_fields = tables_fields[cur_table].FIELD
        new_fields = table_data[~table_data.FIELD.isin(existing_fields)]
        tables_fields[cur_table] = tables_fields[cur_table].append(new_fields)

# write final table
table_names = sorted(tables_fields.keys())
for i in range(len(table_names)):
    if i == 0:
        final_fields = tables_fields[table_names[i]]
    else:
        final_fields = final_fields.append(tables_fields[table_names[i]])
final_fields.to_csv(out_fields_filename)
