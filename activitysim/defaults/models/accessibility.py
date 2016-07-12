# ActivitySim
# See full license in LICENSE.txt.

import logging
import os

import orca
import pandas as pd
import numpy as np

from activitysim import activitysim as asim
from activitysim import tracing


class ASkims(object):

    def __init__(self, skims, omx, length, transpose=False):
        self.skims = skims
        self.omx = omx
        self.length = length
        self.transpose = transpose

    def __getitem__(self, key):

        try:
            data = self.skims.get_skim(key).data
        except KeyError:
            omx_key = '__'.join(key)
            tracing.info(__name__,
                         message="ASkims loading %s from omx as %s" % (key, omx_key,))
            data = self.omx[omx_key]

        data = data[:self.length, :self.length]

        if self.transpose:
            return data.transpose()
        else:
            return data

    def get_from_omx(self, key, v):
        # treat this as a callback - override depending on how you store skims in the omx file
        #
        # from activitysim import skim as askim
        # from types import MethodType
        # askim.Skims3D.get_from_omx = MethodType(get_from_omx, None, askim.Skims3D)

        omx_key = key + '__' + v
        # print "my_get_from_omx - key: '%s' v: '%s', omx_key: '%s'" % (key, v, omx_key)
        return self.omx[omx_key]


@orca.injectable()
def accessibility_spec(configs_dir):
    f = os.path.join(configs_dir, 'configs', "accessibility.csv")
    return asim.read_model_spec(f).fillna(0)


@orca.step()
def compute_accessibility(skims, omx_file, land_use, trace_od, trace_hh_id):
    """
    I am not quite sure how to describe what this does
    """

    tracing.info(__name__,
                 "Running accessibility_simulate")

    land_use_df = land_use.to_frame()

    skim = ASkims(skims, omx_file, len(land_use_df.index))
    skim_t = ASkims(skims, omx_file, len(land_use_df.index), transpose=True)

    skim_shape = skim['DISTWALK'].shape
    tracing.info(__name__,
                 "skim shape %s land_use len %s" % (skim_shape, len(land_use_df.index)))

    # dispersion parameters
    dispersion_parameter_automobile = -0.05
    dispersion_parameter_transit = -0.05
    dispersion_parameter_walk = -1.00
    # maximum walk distance in miles
    maximum_walk_distance = 3.0
    # perceived minute of in-vehicle time for every minute of out-of-vehicle time
    out_of_vehicle_time_weight = 2.0

    # -----
    # retailEmp = zi.1.RETEMPN[j]
    # totalEmp  = zi.1.TOTEMP[j]
    # -----

    retailEmp = np.asanyarray(land_use_df['RETEMPN'])
    retailEmp[np.isnan(retailEmp)] = 0

    totalEmp = np.asanyarray(land_use_df['TOTEMP'])
    totalEmp[np.isnan(totalEmp)] = 0

    # -----
    #  auPkTime = mi.1.TOLLTIMEDA[j] + mi.3.TOLLTIMEDA.T[j]
    #
    #  auOpTime = mi.2.TOLLTIMEDA[j] + mi.2.TOLLTIMEDA.T[j]
    #
    #  auPkRetail = retailEmp * exp(_kAuto * auPkTime)
    #  auOpRetail = retailEmp * exp(_kAuto * auOpTime)
    #
    #  auPkTotal  = totalEmp  * exp(_kAuto * auPkTime)
    #  auOpTotal  = totalEmp  * exp(_kAuto * auOpTime)
    # -----

    # set the automobile level-of-service variables origin/destination time
    # assume peak occurs in AM for outbound and PM for inbound
    auPkTime = skim[('SOVTOLL_TIME', 'AM')] + skim_t[('SOVTOLL_TIME', 'PM')]

    # assume midday occurs entirely in the midday period
    auOpTime = skim[('SOVTOLL_TIME', 'MD')] + skim_t[('SOVTOLL_TIME', 'MD')]

    # retailEmp and totalEmp broadcast to populate each orig row with copy of dest values
    auPkRetail = retailEmp * np.exp(auPkTime * dispersion_parameter_automobile)
    auPkTotal = totalEmp * np.exp(auPkTime * dispersion_parameter_automobile)

    auOpRetail = retailEmp * np.exp(auOpTime * dispersion_parameter_automobile)
    auOpTotal = totalEmp * np.exp(auOpTime * dispersion_parameter_automobile)

    # -----
    # inVehicleTime    = mi.4.IVT[j]
    # outOfVehicleTime = @token_out_of_vehicle_time_weight@
    #                    * (mi.4.IWAIT[j] + mi.4.XWAIT[j] + mi.4.WACC[j]
    #                      + mi.4.WAUX[j] + mi.4.WEGR[j])
    # trPkTime_od      = (inVehicleTime + outOfVehicleTime)/100.0
    #
    # inVehicleTime    = mi.6.IVT.T[j] ##### trnskmPM_wlk_trn_wlk
    # outOfVehicleTime = @token_out_of_vehicle_time_weight@
    #                    * (mi.6.IWAIT.T[j] + mi.6.XWAIT.T[j] + mi.6.WACC.T[j]
    #                       + mi.6.WAUX.T[j] + mi.6.WEGR.T[j])
    # trPkTime_do     = (inVehicleTime + outOfVehicleTime)/100.0
    #
    # trPkTime = trPkTime_od + trPkTime_do
    #
    # if(trPkTime_od > 0 && trPkTime_do > 0)
    #
    #    trPkRetail = retailEmp * exp(_kTran * trPkTime)
    #    trPkTotal  = totalEmp  * exp(_kTran * trPkTime)
    #
    # endif
    # -----

    # set the peak transit level-of-service variables (separately for origin and destination)

    # peak, origin-to-destination, assume outbound occurs in AM peak
    inVehicleTime = skim[('WLK_TRN_WLK_IVT', 'AM')]
    outOfVehicleTime = skim[('WLK_TRN_WLK_IWAIT', 'AM')] \
        + skim[('WLK_TRN_WLK_XWAIT', 'AM')] \
        + skim[('WLK_TRN_WLK_WACC', 'AM')] \
        + skim[('WLK_TRN_WLK_WAUX', 'AM')] \
        + skim[('WLK_TRN_WLK_WEGR', 'AM')]
    trPkTime_od = (inVehicleTime + out_of_vehicle_time_weight * outOfVehicleTime)

    # FIXME - are these times really minutes x 100?
    # FIXME - are they also that in the other skims?
    # convert minutes x 100 to minutes
    trPkTime_od = trPkTime_od / 100.0

    # peak, destination-to-origin, assume inbound occurs in PM peak
    inVehicleTime = skim_t[('WLK_TRN_WLK_IVT', 'PM')]
    outOfVehicleTime = skim_t[('WLK_TRN_WLK_IWAIT', 'PM')] \
        + skim_t[('WLK_TRN_WLK_XWAIT', 'PM')] \
        + skim_t[('WLK_TRN_WLK_WACC', 'PM')] \
        + skim_t[('WLK_TRN_WLK_WAUX', 'PM')] \
        + skim_t[('WLK_TRN_WLK_WEGR', 'PM')]
    trPkTime_do = (inVehicleTime + out_of_vehicle_time_weight * outOfVehicleTime)

    # FIXME - are these times really minutes x 100?
    # convert minutes x 100 to minutes
    trPkTime_do = trPkTime_do / 100.0

    # peak round-trip time
    trPkTime = trPkTime_od + trPkTime_do

    #  compute the decay function for peak transit accessibility if a round trip path is available
    # (zero otherwise)
    rt_available = (trPkTime_od > 0) & (trPkTime_do > 0)
    trPkDecay = np.empty(trPkTime.shape)
    trPkDecay[rt_available] = np.exp(trPkTime[rt_available] * dispersion_parameter_transit)
    trPkDecay[~rt_available] = 0

    trPkRetail = retailEmp * trPkDecay
    trPkTotal = totalEmp * trPkDecay

    # -----
    # inVehicleTime    = mi.5.IVT[j] ##### trnskmMD_wlk_trn_wlk
    # outOfVehicleTime = @token_out_of_vehicle_time_weight@
    #  * (mi.5.IWAIT[j] + mi.5.XWAIT[j] + mi.5.WACC[j] + mi.5.WAUX[j] + mi.5.WEGR[j])
    # trOpTime_od     = (inVehicleTime + outOfVehicleTime)/100.0
    #
    # inVehicleTime    = mi.5.IVT.T[j]
    # outOfVehicleTime = @token_out_of_vehicle_time_weight@
    #  * (mi.5.IWAIT.T[j] + mi.5.XWAIT.T[j] + mi.5.WACC.T[j] + mi.5.WAUX.T[j] + mi.5.WEGR.T[j])
    # trOpTime_do     = (inVehicleTime + outOfVehicleTime)/100.0
    #
    # trOpTime = trOpTime_od + trOpTime_do
    #
    # FIXME - this assumes DO always available if OD is.
    # FIXME - may not matter but probably should be:
    # FIXME - if(trOpTime_od > 0 && trOpTime_do > 0)
    # if(trOpTime>0)
    #
    #    trOpRetail = retailEmp * exp(_kTran * trOpTime)
    #    trOpTotal  = totalEmp  * exp(_kTran * trOpTime)
    #
    # endif
    # -----

    # set the off-peak transit level-of-service variables
    # (separately for the origin and destination)

    # off-peak, origin-to-destination, assume outbound occurs in the MD time period
    inVehicleTime = skim[('WLK_TRN_WLK_IVT', 'MD')]
    outOfVehicleTime = skim[('WLK_TRN_WLK_IWAIT', 'MD')] \
        + skim[('WLK_TRN_WLK_XWAIT', 'MD')] \
        + skim[('WLK_TRN_WLK_WACC', 'MD')] \
        + skim[('WLK_TRN_WLK_WAUX', 'MD')] \
        + skim[('WLK_TRN_WLK_WEGR', 'MD')]
    trOpTime_od = (inVehicleTime + out_of_vehicle_time_weight * outOfVehicleTime)
    # FIXME - are these times really minutes x 100?
    trOpTime_od = trOpTime_od / 100.0

    # off-peak, destination-to-origin, assume it's the same time as the origin-to-destination
    inVehicleTime = skim_t[('WLK_TRN_WLK_IVT', 'MD')]
    outOfVehicleTime = skim_t[('WLK_TRN_WLK_IWAIT', 'MD')] \
        + skim_t[('WLK_TRN_WLK_XWAIT', 'MD')] \
        + skim_t[('WLK_TRN_WLK_WACC', 'MD')] \
        + skim_t[('WLK_TRN_WLK_WAUX', 'MD')] \
        + skim_t[('WLK_TRN_WLK_WEGR', 'MD')]
    trOpTime_do = (inVehicleTime + out_of_vehicle_time_weight * outOfVehicleTime)
    # FIXME - are these times really minutes x 100?
    trOpTime_do = trOpTime_do / 100.0

    # off-peak, round-trip time
    trOpTime = trOpTime_od + trOpTime_do

    #  compute the decay function for peak transit accessibility if a round trip path is available
    # (zero otherwise)

    # FIXME - doing this way because mtc_accessibility.job does...
    one_way = np.logical_xor((trOpTime_od > 0) & (trOpTime_do > 0), (trOpTime > 0))
    if np.sum(one_way) > 0:
        tracing.warn(__name__, "OP transit %s one way" % (np.sum(one_way),))
        od_pairs = np.transpose(np.nonzero(one_way)) + 1
        tracing.write_array(od_pairs, "one_way_transit", fmt='%d')

    rt_available = (trOpTime > 0)
    # rt_available = (trOpTime_od > 0) & (trOpTime_do > 0)

    trOpDecay = np.empty(trOpTime.shape)
    trOpDecay[rt_available] = np.exp(trOpTime[rt_available] * dispersion_parameter_transit)
    trOpDecay[~rt_available] = 0

    trOpRetail = retailEmp * trOpDecay
    trOpTotal = totalEmp * trOpDecay

    # -----
    # nmDist = mi.7.DISTWALK[j] + mi.7.DISTWALK.T[j] ##### nonmotskm
    #
    # if(nmDist <= @token_maximum_walk_distance@)
    #
    #    nmRetail = retailEmp * exp(_kWalk * nmDist)
    #    nmTotal  = totalEmp  * exp(_kWalk * nmDist)
    #
    # endif
    # -----

    nmDist = skim['DISTWALK'] + skim_t['DISTWALK']

    #  compute the decay function for peak transit accessibility if a round trip path is available
    # (zero otherwise)
    rt_available = (nmDist <= maximum_walk_distance)

    nmDecay = np.empty(nmDist.shape)
    nmDecay[rt_available] = np.exp(nmDecay[rt_available] * dispersion_parameter_walk)
    nmDecay[~rt_available] = 0

    nmRetail = retailEmp * trOpDecay
    nmTotal = totalEmp * trOpDecay

    # FIXME - endjloop

    # np.fill_diagonal(auPkRetail, 0.0)
    # np.fill_diagonal(auPkTotal, 0.0)
    # np.fill_diagonal(auOpRetail, 0.0)
    # np.fill_diagonal(auOpTotal, 0.0)
    # np.fill_diagonal(trPkRetail, 0.0)
    # np.fill_diagonal(trPkTotal, 0.0)
    # np.fill_diagonal(trOpRetail, 0.0)
    # np.fill_diagonal(trOpTotal, 0.0)
    # np.fill_diagonal(nmRetail, 0.0)
    # np.fill_diagonal(nmTotal, 0.0)

    lnAuPkRetail = np.log(np.sum(auPkRetail, axis=1) + 1)
    lnAuPkTotal = np.log(np.sum(auPkTotal, axis=1) + 1)
    lnAuOpRetail = np.log(np.sum(auOpRetail, axis=1) + 1)
    lnAuOpTotal = np.log(np.sum(auOpTotal, axis=1) + 1)
    lnTrPkRetail = np.log(np.sum(trPkRetail, axis=1) + 1)
    lnTrPkTotal = np.log(np.sum(trPkTotal, axis=1) + 1)
    lnTrOpRetail = np.log(np.sum(trOpRetail, axis=1) + 1)
    lnTrOpTotal = np.log(np.sum(trOpTotal, axis=1) + 1)
    lnNmRetail = np.log(np.sum(nmRetail, axis=1) + 1)
    lnNmTotal = np.log(np.sum(nmTotal, axis=1) + 1)

    index = land_use_df.index
    orca.add_column("accessibility", "xAUTOPEAKRETAIL", pd.Series(lnAuPkRetail, index=index))
    orca.add_column("accessibility", "xAUTOPEAKTOTAL",  pd.Series(lnAuPkTotal, index=index))
    orca.add_column("accessibility", "xAUTOOFFPEAKRETAIL", pd.Series(lnAuOpRetail, index=index))
    orca.add_column("accessibility", "xAUTOOFFPEAKTOTAL",  pd.Series(lnAuOpTotal, index=index))
    orca.add_column("accessibility", "xTRANSITPEAKRETAIL", pd.Series(lnTrPkRetail, index=index))
    orca.add_column("accessibility", "xTRANSITPEAKTOTAL",  pd.Series(lnTrPkTotal, index=index))
    orca.add_column("accessibility", "xTRANSITOFFPEAKRETAIL", pd.Series(lnTrOpRetail, index=index))
    orca.add_column("accessibility", "xTRANSITOFFPEAKTOTAL",  pd.Series(lnTrOpTotal, index=index))
    orca.add_column("accessibility", "xNONMOTORIZEDRETAIL",   pd.Series(lnNmRetail, index=index))
    orca.add_column("accessibility", "xNONMOTORIZEDTOTAL",    pd.Series(lnNmTotal, index=index))

    if trace_od:
        tracing.info(__name__,
                     "trace origin = %s, dest = %s" % (trace_od[0], trace_od[1]))

        tracing.trace_df(land_use_df, "accessibility.land_use",
                         column_labels=['label', 'orig_taz', 'dest_taz'])

        tracing.info(__name__,
                     message="dispersion_parameter_automobile = %s"
                             % (dispersion_parameter_automobile,))
        tracing.info(__name__,
                     message="dispersion_parameter_transit = %s"
                             % (dispersion_parameter_transit,))
        tracing.info(__name__,
                     message="dispersion_parameter_walk = %s"
                             % (dispersion_parameter_walk,))
        tracing.info(__name__,
                     message="maximum_walk_distance = %s"
                             % (maximum_walk_distance,))
        tracing.info(__name__,
                     message="out_of_vehicle_time_weight = %s"
                             % (out_of_vehicle_time_weight,))

        tracing.trace_df(land_use_df['RETEMPN'], "accessibility.RETEMPN", slicer='NONE')
        tracing.trace_df(land_use_df['TOTEMP'], "accessibility.TOTEMP", slicer='NONE')

        tracing.trace_array(__name__, retailEmp, "retailEmp")
        tracing.trace_array(__name__, totalEmp, "totalEmp")
        tracing.trace_array(__name__, auPkTime, "auPkTime")
        tracing.trace_array(__name__, auPkRetail, "auPkRetail")
        tracing.trace_array(__name__, auPkTotal, "auPkTotal")
        tracing.trace_array(__name__, trPkTime_od, "trPkTime_od")
        tracing.trace_array(__name__, trPkTime_do, "trPkTime_do")
        tracing.trace_array(__name__, trOpTime_od, "trOpTime_od")
        tracing.trace_array(__name__, trOpTime_do, "trOpTime_do")
        tracing.trace_array(__name__, trPkTime, "trOpTime")
        tracing.trace_array(__name__, trPkDecay, "trOpDecay")
        tracing.trace_array(__name__, trOpRetail, "trOpRetail")
        tracing.trace_array(__name__, trOpTotal, "trOpTotal")
        tracing.trace_array(__name__, nmDist, "nmDist")
        tracing.trace_array(__name__, nmRetail, "nmRetail")
        tracing.trace_array(__name__, nmTotal, "nmTotal")

        tracing.trace_df(orca.get_table('accessibility').to_frame(), "accessibility",
                         column_labels=['label', 'orig_taz', 'dest_taz'])

        # tracing.trace_df(orca.get_table('accessibility').to_frame(), "accessibility.full",
        #                  slicer='NONE', transpose=False)

        tracing.trace_df(orca.get_table('persons_merged').to_frame(), "persons_merged")
