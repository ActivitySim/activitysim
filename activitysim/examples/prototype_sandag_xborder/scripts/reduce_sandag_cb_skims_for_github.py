# remove unused skims from full scale sandag cross border skim files to
# reduce file size for upload to GitHub since there's a 2GB file size limit
# run this script and then run these repack commands to reduce file size

# Ben.Stabler@rsginc.com, 10/18/21

# h5repack -i traffic_skims_xborder_EA.omx -o traffic_skims_xborder_EA_repacked.omx
# h5repack -i traffic_skims_xborder_AM.omx -o traffic_skims_xborder_AM_repacked.omx
# h5repack -i traffic_skims_xborder_MD.omx -o traffic_skims_xborder_MD_repacked.omx
# h5repack -i traffic_skims_xborder_PM.omx -o traffic_skims_xborder_PM_repacked.omx
# h5repack -i traffic_skims_xborder_EV.omx -o traffic_skims_xborder_EV_repacked.omx

import tables

time_periods = ["EA", "AM", "MD", "PM", "EV"]

skims_to_remove = [
    "/data/HOV2_H_HOVDIST",
    "/data/HOV2_H_REL",
    "/data/HOV2_H_TOLLDIST",
    "/data/HOV2_L_HOVDIST",
    "/data/HOV2_L_REL",
    "/data/HOV2_L_TOLLDIST",
    "/data/HOV2_M_HOVDIST",
    "/data/HOV2_M_REL",
    "/data/HOV2_M_TOLLDIST",
    "/data/HOV3_H_HOVDIST",
    "/data/HOV3_H_REL",
    "/data/HOV3_H_TOLLDIST",
    "/data/HOV3_L_HOVDIST",
    "/data/HOV3_L_REL",
    "/data/HOV3_L_TOLLDIST",
    "/data/HOV3_M_HOVDIST",
    "/data/HOV3_M_REL",
    "/data/HOV3_M_TOLLDIST",
    "/data/SOV_NT_H_REL",
    "/data/SOV_NT_H_TOLLDIST",
    "/data/SOV_NT_L_REL",
    "/data/SOV_NT_L_TOLLDIST",
    "/data/SOV_NT_M_REL",
    "/data/SOV_NT_M_TOLLDIST",
    "/data/SOV_TR_H_DIST",
    "/data/SOV_TR_H_REL",
    "/data/SOV_TR_H_TIME",
    "/data/SOV_TR_H_TOLLCOST",
    "/data/SOV_TR_H_TOLLDIST",
    "/data/SOV_TR_L_DIST",
    "/data/SOV_TR_L_REL",
    "/data/SOV_TR_L_TIME",
    "/data/SOV_TR_L_TOLLCOST",
    "/data/SOV_TR_L_TOLLDIST",
    "/data/SOV_TR_M_DIST",
    "/data/SOV_TR_M_REL",
    "/data/SOV_TR_M_TIME",
    "/data/SOV_TR_M_TOLLCOST",
    "/data/SOV_TR_M_TOLLDIST",
    "/data/TRK_H_DIST",
    "/data/TRK_H_TIME",
    "/data/TRK_H_TOLLCOST",
    "/data/TRK_L_DIST",
    "/data/TRK_L_TIME",
    "/data/TRK_L_TOLLCOST",
    "/data/TRK_M_DIST",
    "/data/TRK_M_TIME",
    "/data/TRK_M_TOLLCOST",
]

for time_period in time_periods:
    f = tables.open_file("traffic_skims_xborder_" + time_period + ".omx", "a")
    for skim in skims_to_remove:
        print(skim + "__" + time_period)
        f.remove_node(skim + "__" + time_period)
    f.close()
