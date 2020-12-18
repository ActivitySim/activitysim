
x=read.csv("C:/projects/activitysim/activitysim/examples/example_multiple_zone/output_3_marin_full/final_tours.csv")

#hh=read.csv("C:/projects/activitysim/activitysim/examples/example_multiple_zone/output_3_marin_full/final_households.csv")
#x$hhzone = hh$home_zone_id[match(x$household_id,hh$household_id)]
#table(x$tour_mode, x$hhzone==x$destination)


#mode share
tm2_mode_codes = c("DRIVEALONEFREE","SHARED2FREE","SHARED2FREE","SHARED3FREE","SHARED3FREE","WALK","BIKE","WALK_TRANSIT","DRIVE_TRANSIT","DRIVE_TRANSIT","TAXI")
names(tm2_mode_codes) = c(1,3,4,6,7,9,10,11,12,13,15)
x$tm2_MODE = tm2_mode_codes[match(x$tm2_tour_mode, names(tm2_mode_codes))]
tm2_ms  = as.data.frame(round(table(x$tm2_MODE) / nrow(x),2))

write.csv(tm2_ms, "c:/projects/tm2_ms.csv", row.names=F)


asim_ms = as.data.frame(round(table(x$tour_mode) / nrow(x),4))
write.csv(asim_ms, "c:/projects/asim_ms.csv", row.names=F)

#taps

taps=read.csv("C:/projects/activitysim/activitysim/examples/example_multiple_zone/data_3_marin_full/tap_data.csv")

#tap count
tm2_out_btap = sort(table(x$tm2_out_btap))
asim_out_btap = sort(table(x$od_btap))

taps = merge(taps, as.data.frame(tm2_out_btap), by.x="TAP", by.y="Var1")
taps = merge(taps, as.data.frame(asim_out_btap), by.x="TAP", by.y="Var1")

write.csv(taps, "c:/projects/taps.csv", row.names=F)



