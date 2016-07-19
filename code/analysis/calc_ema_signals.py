import sys
import os
from itertools import izip
sys.path.append('/home/egrois/git/code/pybt')
sys.path.append('/home/egrois/git/code/preprocess')
sys.path.append('/home/egrois/git/code/analysis')
import pybt
reload(pybt)
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
from datetime import datetime,time, timedelta
import timeit
import pandas as pd
import utils
reload(utils)
import csv
import math
import re

import mkdt_utils
reload(mkdt_utils)
import tech_anal
reload(tech_anal)
#import calc_trade_signals
#reload(calc_trade_signals)



"""Event-specific parameter setting"""
event_name = "ADP"
instrument_root = '6E'
event_start_time = '08:15:00'
event_time_offset_s = 0  # for example, for Payrolls can be =1.0
spike_duration_s = 2

event_dates = ['2013-01-03', '2013-03-06', '2013-04-03', '2013-05-01', '2013-06-05', '2013-07-03', '2013-07-31', '2013-09-05', '2013-10-02', '2013-10-30', '2013-12-04', '2014-01-08', '2014-02-05', '2014-03-05', '2014-04-02', '2014-04-30', '2014-06-04', '2014-07-02', '2014-07-30', '2014-09-04', '2014-10-01', '2014-11-05', '2014-12-03', '2015-01-07', '2015-02-04', '2015-03-04', '2015-04-01', '2015-05-06', '2015-06-03', '2015-07-01', '2015-08-05', '2015-09-02']
#event_dates = ['2013-07-03']

"""Parameter settings for valuing an entry/trade.  These are specific to ADP and 6E (for now)."""
min_price_increment = 1  # need to have a lookup structure for these
book_depth = 5
nominal_trading_size = 100  # num contracts to go for at each entry
trading_start_time = "08:15:00"
max_entry_time = "08:25:00"
trading_stop_time = "08:27:00"
max_hold_period_s = 120
min_profit_ticks = 4
stop_loss_ticks = 8
exit_depth = 0  # cross book and trade at top of book


spike_5s = {'2013-01-03': -4, '2013-03-06': -4, '2013-04-03': 10, '2013-05-01': 6, '2013-06-05': 26, '2013-07-03': -20, '2013-07-31': -12, '2013-09-05': 5, '2013-10-02': 10, '2013-10-30': 6, '2013-12-04': -22, '2014-01-08': -15, '2014-02-05': 8, '2014-03-05': 9, '2014-04-02': -2, '2014-04-30': -3, '2014-06-04': 6, '2014-07-02': -16, '2014-07-30': 10, '2014-09-04': 2, '2014-10-01': -7, '2014-11-05': -9, '2014-12-03': 3, '2015-01-07': -9, '2015-02-04': 6, '2015-03-04': 5, '2015-04-01': 23, '2015-05-06': 36, '2015-06-03': -17, '2015-07-01': -13, '2015-08-05': 51, '2015-09-02': 13}


spike_5s_pred = {'2013-01-03': -16, '2013-03-06': -10, '2013-04-03': 26, '2013-05-01': 20, '2013-06-05': 18, '2013-07-03': -13, '2013-07-31': -7, '2013-09-05': 7, '2013-10-02': 5, '2013-10-30': 6, '2013-12-04': -20, '2014-01-08': -18, '2014-02-05': 10, '2014-03-05': 12, '2014-04-02': -5, '2014-04-30': -9, '2014-06-04': 18, '2014-07-02': -21, '2014-07-30': 6, '2014-09-04': 9, '2014-10-01': -5, '2014-11-05': -8, '2014-12-03': 8, '2015-01-07': -13, '2015-02-04': 3, '2015-03-04': 3, '2015-04-01': 27, '2015-05-06': 23, '2015-06-03': -1, '2015-07-01': -9, '2015-08-05': 21, '2015-09-02': 7}




def computeTimeOffsetSignals(df, ema_feats_df, start_event_dt, start_dt, stop_dt, start_event_loc, start_loc, stop_loc):
    ema_feats_df.ix[(ema_feats_df.time<start_dt)&(ema_feats_df.time>stop_dt), 'offset_intervals'] = float('NaN')

    for i in range(start_event_loc, stop_loc):
	ema_feats_df.ix[i, 'offset_intervals'] = i - start_event_loc

    return (ema_feats_df, df)

def computeEMAs(df, ema_feats_df, start_event_dt):
    df['microprice_ema_700ms'] = pd.ewma(df['microprice'], halflife=7)
    df['microprice_ema_1200ms'] = pd.ewma(df['microprice'], halflife=12)
    df['microprice_ema_3600ms'] = pd.ewma(df['microprice'], halflife=36)
    df['microprice_ema_6000ms'] = pd.ewma(df['microprice'], halflife=60)
    df['microprice_ema_26000ms'] = pd.ewma(df['microprice'], halflife=260)
    df['microprice_ema_62000ms'] = pd.ewma(df['microprice'], halflife=620)
    df['microprice_ema_150000ms'] = pd.ewma(df['microprice'], halflife=1500)

    pre_event_snapshot_dt = start_event_dt - timedelta(seconds=2.5)
    pre_event_microprice = df[df.time == pre_event_snapshot_dt]['microprice_ema_200ms'][0]

    df['microprice_ema_200ms_ticks'] = df['microprice_ema_200ms'] - pre_event_microprice
    df['microprice_ema_700ms_ticks'] = df['microprice_ema_700ms'] - pre_event_microprice
    df['microprice_ema_1200ms_ticks'] = df['microprice_ema_1200ms'] - pre_event_microprice
    df['microprice_ema_2000ms_ticks'] = df['microprice_ema_2s'] - pre_event_microprice
    df['microprice_ema_3600ms_ticks'] = df['microprice_ema_3600ms'] - pre_event_microprice
    df['microprice_ema_6000ms_ticks'] = df['microprice_ema_6000ms'] - pre_event_microprice
    df['microprice_ema_10000ms_ticks'] = df['microprice_ema_10s'] - pre_event_microprice
    df['microprice_ema_26000ms_ticks'] = df['microprice_ema_26000ms'] - pre_event_microprice
    df['microprice_ema_62000ms_ticks'] = df['microprice_ema_62000ms'] - pre_event_microprice
    df['microprice_ema_150000ms_ticks'] = df['microprice_ema_150000ms'] - pre_event_microprice

    return (ema_feats_df, df)


def computeRelativeDislocation(s, p, norm=1.0):
    return (s - p) / norm
    

def computeSpikeSignals(df, ema_feats_df, start_event_dt, spike_5s, spike_5s_pred, norm):
    ema_feats_df['microprice_ema_200ms_from_spike_5s'] = computeRelativeDislocation(df['microprice_ema_200ms_ticks'], spike_5s, spike_5s)
    ema_feats_df['microprice_ema_700ms_from_spike_5s'] = computeRelativeDislocation(df['microprice_ema_700ms_ticks'], spike_5s, spike_5s)
    #ema_feats_df['microprice_ema_1200ms_from_spike_5s'] = computeRelativeDislocation(df['microprice_ema_1200ms_ticks'], spike_5s, spike_5s)
    ema_feats_df['microprice_ema_2000ms_from_spike_5s'] = computeRelativeDislocation(df['microprice_ema_2000ms_ticks'], spike_5s, spike_5s)
    #ema_feats_df['microprice_ema_3600ms_from_spike_5s'] = computeRelativeDislocation(df['microprice_ema_3600ms_ticks'], spike_5s, spike_5s)
    ema_feats_df['microprice_ema_6000ms_from_spike_5s'] = computeRelativeDislocation(df['microprice_ema_6000ms_ticks'], spike_5s, spike_5s)
    ema_feats_df['microprice_ema_10000ms_from_spike_5s'] = computeRelativeDislocation(df['microprice_ema_10000ms_ticks'], spike_5s, spike_5s)
    ema_feats_df['microprice_ema_26000ms_from_spike_5s'] = computeRelativeDislocation(df['microprice_ema_26000ms_ticks'], spike_5s, spike_5s)
    ema_feats_df['microprice_ema_62000ms_from_spike_5s'] = computeRelativeDislocation(df['microprice_ema_62000ms_ticks'], spike_5s, spike_5s)
    ema_feats_df['microprice_ema_150000ms_from_spike_5s'] = computeRelativeDislocation(df['microprice_ema_150000ms_ticks'], spike_5s, spike_5s)

    ema_feats_df['microprice_ema_200ms_from_spike_pred'] = computeRelativeDislocation(df['microprice_ema_200ms_ticks'], spike_5s_pred, spike_5s_pred)
    #ema_feats_df['microprice_ema_700ms_from_spike_pred'] = computeRelativeDislocation(df['microprice_ema_700ms_ticks'], spike_5s_pred, spike_5s_pred)
    #ema_feats_df['microprice_ema_1200ms_from_spike_pred'] = computeRelativeDislocation(df['microprice_ema_1200ms_ticks'], spike_5s_pred, spike_5s_pred)
    ema_feats_df['microprice_ema_2000ms_from_spike_pred'] = computeRelativeDislocation(df['microprice_ema_2000ms_ticks'], spike_5s_pred, spike_5s_pred)
    #ema_feats_df['microprice_ema_3600ms_from_spike_pred'] = computeRelativeDislocation(df['microprice_ema_3600ms_ticks'], spike_5s_pred, spike_5s_pred)
    ema_feats_df['microprice_ema_6000ms_from_spike_pred'] = computeRelativeDislocation(df['microprice_ema_6000ms_ticks'], spike_5s_pred, spike_5s_pred)
    ema_feats_df['microprice_ema_10000ms_from_spike_pred'] = computeRelativeDislocation(df['microprice_ema_10000ms_ticks'], spike_5s_pred, spike_5s_pred)
    ema_feats_df['microprice_ema_26000ms_from_spike_pred'] = computeRelativeDislocation(df['microprice_ema_26000ms_ticks'], spike_5s_pred, spike_5s_pred)
    ema_feats_df['microprice_ema_62000ms_from_spike_pred'] = computeRelativeDislocation(df['microprice_ema_62000ms_ticks'], spike_5s_pred, spike_5s_pred)
    ema_feats_df['microprice_ema_150000ms_from_spike_pred'] = computeRelativeDislocation(df['microprice_ema_150000ms_ticks'], spike_5s_pred, spike_5s_pred)

    return (ema_feats_df, df)


def computePeakEMADiffs(ema1, ema2, norm=1.0):
    norm = 1.0  # force this for now because may not want normalization for this diff
    diff = (ema1 - ema2) / norm
    ##smooth_diff = pd.ewma(diff, span=10)
    #diff_diff = diff - smooth_diff

    #diff_buy_signal = map(lambda a, b: math.log(-1*a*b+1) if (a<0 and b>0) else math.log(math.log(a*b+1)+1) if (a<0 and b<0) else 0, \
    #                            diff, diff_diff)
    #diff_sell_signal = map(lambda a, b: math.log(-1*a*b+1) if (a>0 and b<0) else math.log(math.log(a*b+1)+1) if (a>0 and b>0) else 0, \
    #                            diff, diff_diff)
    
    #diff_signal = map(lambda b, s: b if b>s else -1.0*s if s>b else 0, diff_buy_signal, diff_sell_signal)  #signed version of buy/sell signals

    return diff


def computeEMADiffSignals(df, ema_feats_df, norm):
    ema_feats_df['diff_ema_200ms_700ms'] = computePeakEMADiffs(df['microprice_ema_200ms_ticks'], df['microprice_ema_700ms_ticks'], df['microprice_ema_700ms_ticks'])  #norm
    #ema_feats_df['diff_ema_200ms_1200ms'] = computePeakEMADiffs(df['microprice_ema_200ms_ticks'], df['microprice_ema_1200ms_ticks'], norm)
    #ema_feats_df['diff_ema_700ms_1200ms'] = computePeakEMADiffs(df['microprice_ema_700ms_ticks'], df['microprice_ema_1200ms_ticks'], norm)
    #ema_feats_df['diff_ema_700ms_2000ms'] = computePeakEMADiffs(df['microprice_ema_700ms_ticks'], df['microprice_ema_2000ms_ticks'], norm)
    #ema_feats_df['diff_ema_1200ms_2000ms'] = computePeakEMADiffs(df['microprice_ema_1200ms_ticks'], df['microprice_ema_2000ms_ticks'], norm)
    ema_feats_df['diff_ema_1200ms_3600ms'] = computePeakEMADiffs(df['microprice_ema_1200ms_ticks'], df['microprice_ema_3600ms_ticks'], df['microprice_ema_3600ms_ticks'])
    #ema_feats_df['diff_ema_1200ms_6000ms'] = computePeakEMADiffs(df['microprice_ema_1200ms_ticks'], df['microprice_ema_6000ms_ticks'], norm)
    #ema_feats_df['diff_ema_2000ms_3600ms'] = computePeakEMADiffs(df['microprice_ema_2000ms_ticks'], df['microprice_ema_3600ms_ticks'], norm)
    #ema_feats_df['diff_ema_2000ms_6000ms'] = computePeakEMADiffs(df['microprice_ema_2000ms_ticks'], df['microprice_ema_6000ms_ticks'], norm)
    ema_feats_df['diff_ema_3600ms_6000ms'] = computePeakEMADiffs(df['microprice_ema_3600ms_ticks'], df['microprice_ema_6000ms_ticks'], df['microprice_ema_6000ms_ticks'])
    #ema_feats_df['diff_ema_2000ms_10000ms'] = computePeakEMADiffs(df['microprice_ema_2000ms_ticks'], df['microprice_ema_10000ms_ticks'], norm)
    ema_feats_df['diff_ema_3600ms_10000ms'] = computePeakEMADiffs(df['microprice_ema_3600ms_ticks'], df['microprice_ema_10000ms_ticks'], df['microprice_ema_10000ms_ticks'])
    ema_feats_df['diff_ema_6000ms_10000ms'] = computePeakEMADiffs(df['microprice_ema_6000ms_ticks'], df['microprice_ema_10000ms_ticks'], df['microprice_ema_10000ms_ticks'])
    ema_feats_df['diff_ema_6000ms_26000ms'] = computePeakEMADiffs(df['microprice_ema_6000ms_ticks'], df['microprice_ema_26000ms_ticks'], df['microprice_ema_26000ms_ticks'])
    ema_feats_df['diff_ema_10000ms_26000ms'] = computePeakEMADiffs(df['microprice_ema_10000ms_ticks'], df['microprice_ema_26000ms_ticks'], df['microprice_ema_26000ms_ticks'])
    ema_feats_df['diff_ema_10000ms_62000ms'] = computePeakEMADiffs(df['microprice_ema_10000ms_ticks'], df['microprice_ema_62000ms_ticks'], df['microprice_ema_62000ms_ticks'])
    ema_feats_df['diff_ema_26000ms_62000ms'] = computePeakEMADiffs(df['microprice_ema_26000ms_ticks'], df['microprice_ema_62000ms_ticks'], df['microprice_ema_62000ms_ticks'])
    ema_feats_df['diff_ema_26000ms_150000ms'] = computePeakEMADiffs(df['microprice_ema_26000ms_ticks'], df['microprice_ema_150000ms_ticks'], df['microprice_ema_150000ms_ticks'])
    ema_feats_df['diff_ema_62000ms_150000ms'] = computePeakEMADiffs(df['microprice_ema_62000ms_ticks'], df['microprice_ema_150000ms_ticks'], df['microprice_ema_150000ms_ticks'])

    return (ema_feats_df, df)


def computeEMAAverageSlope(ema1, ema2, offset1=5, offset2=5):
    avg_slope = ((ema1 - ema1.shift(offset1)) + (ema2 - ema2.shift(offset2))) / 2.0

    avg_slope_buy_signal = map(lambda a: math.log(-1*a+1) if (a<0) else 0, avg_slope)
    avg_slope_sell_signal =  map(lambda a: math.log(a+1) if (a>0) else 0, avg_slope)

    avg_slope_signal = map(lambda b, s: b if b>s else -1.0*s if s>b else 0, avg_slope_buy_signal, avg_slope_sell_signal)  #signed version of buy/sell signals
    
    return avg_slope_signal


def computeEMAAverageSlopeSignals(df, ema_feats_df):
    ema_feats_df['ema_avg_slope_200ms_700ms'] = computeEMAAverageSlope(df['microprice_ema_200ms_ticks'], df['microprice_ema_700ms_ticks'], 3, 4)
    ema_feats_df['ema_avg_slope_200ms_1200ms'] = computeEMAAverageSlope(df['microprice_ema_200ms_ticks'], df['microprice_ema_1200ms_ticks'], 3, 5)
    #ema_feats_df['ema_avg_slope_700ms_1200ms'] = computeEMAAverageSlope(df['microprice_ema_700ms_ticks'], df['microprice_ema_1200ms_ticks'], 4, 5)
    #ema_feats_df['ema_avg_slope_700ms_2000ms'] = computeEMAAverageSlope(df['microprice_ema_700ms_ticks'], df['microprice_ema_2000ms_ticks'], 4, 5)
    #ema_feats_df['ema_avg_slope_1200ms_2000ms'] = computeEMAAverageSlope(df['microprice_ema_1200ms_ticks'], df['microprice_ema_2000ms_ticks'])
    #ema_feats_df['ema_avg_slope_1200ms_3600ms'] = computeEMAAverageSlope(df['microprice_ema_1200ms_ticks'], df['microprice_ema_3600ms_ticks'])
    #ema_feats_df['ema_avg_slope_1200ms_6000ms'] = computeEMAAverageSlope(df['microprice_ema_1200ms_ticks'], df['microprice_ema_6000ms_ticks'])
    #ema_feats_df['ema_avg_slope_2000ms_3600ms'] = computeEMAAverageSlope(df['microprice_ema_2000ms_ticks'], df['microprice_ema_3600ms_ticks'])
    #ema_feats_df['ema_avg_slope_2000ms_6000ms'] = computeEMAAverageSlope(df['microprice_ema_2000ms_ticks'], df['microprice_ema_6000ms_ticks'])
    #ema_feats_df['ema_avg_slope_3600ms_6000ms'] = computeEMAAverageSlope(df['microprice_ema_3600ms_ticks'], df['microprice_ema_6000ms_ticks'])
    #ema_feats_df['ema_avg_slope_2000ms_10000ms'] = computeEMAAverageSlope(df['microprice_ema_2000ms_ticks'], df['microprice_ema_10000ms_ticks'])
    #ema_feats_df['ema_avg_slope_3600ms_10000ms'] = computeEMAAverageSlope(df['microprice_ema_3600ms_ticks'], df['microprice_ema_10000ms_ticks'])
    #ema_feats_df['ema_avg_slope_6000ms_10000ms'] = computeEMAAverageSlope(df['microprice_ema_6000ms_ticks'], df['microprice_ema_10000ms_ticks'])
    #ema_feats_df['ema_avg_slope_6000ms_26000ms'] = computeEMAAverageSlope(df['microprice_ema_6000ms_ticks'], df['microprice_ema_26000ms_ticks'])
    #ema_feats_df['ema_avg_slope_10000ms_26000ms'] = computeEMAAverageSlope(df['microprice_ema_10000ms_ticks'], df['microprice_ema_26000ms_ticks'])
    #ema_feats_df['ema_avg_slope_10000ms_62000ms'] = computeEMAAverageSlope(df['microprice_ema_10000ms_ticks'], df['microprice_ema_62000ms_ticks'])
    #ema_feats_df['ema_avg_slope_26000ms_62000ms'] = computeEMAAverageSlope(df['microprice_ema_26000ms_ticks'], df['microprice_ema_62000ms_ticks'])
    ema_feats_df['ema_avg_slope_26000ms_150000ms'] = computeEMAAverageSlope(df['microprice_ema_26000ms_ticks'], df['microprice_ema_150000ms_ticks'])
    ema_feats_df['ema_avg_slope_62000ms_150000ms'] = computeEMAAverageSlope(df['microprice_ema_62000ms_ticks'], df['microprice_ema_150000ms_ticks'])

    return (ema_feats_df, df)


def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]
def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get cosine value
    cos_ = dot_prod/magA/magB
    
    return cos_

def ang(diffYA, diffYB, diffXA, diffXB):
    vA = [diffYA, diffXA]
    vB = [diffYB, diffXB]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get cosine value
    cos_ = dot_prod/magA/magB
    
    # Get angle in radians 
    angle_radians = math.acos(min(max(cos_,-1.0),1.0))
    cos_ = math.cos(angle_radians - (math.pi*0.5))

    return cos_


def computeAngle(seriesA, seriesB, offsetA=5, offsetB=5):
    angle_series = map(lambda sA_p1y, sA_p2y, sB_p1y, sB_p2y: np.sign(sA_p1y-sB_p1y)*ang((sA_p1y-sA_p2y), (sB_p1y-sB_p2y), offsetA, offsetB),\
                       seriesA, seriesA.shift(offsetA), seriesB, seriesB.shift(offsetB) )
    
    return angle_series


def computeEMAAngles(df, ema_feats_df):
    ema_feats_df['ema_angle_200ms_700ms'] = computeAngle(df['microprice_ema_200ms_ticks'], df['microprice_ema_700ms_ticks'], 3, 4)
    ema_feats_df['ema_angle_200ms_1200ms'] = computeAngle(df['microprice_ema_200ms_ticks'], df['microprice_ema_1200ms_ticks'], 3, 5)
    ema_feats_df['ema_angle_700ms_1200ms'] = computeAngle(df['microprice_ema_700ms_ticks'], df['microprice_ema_1200ms_ticks'], 4, 5)
    #ema_feats_df['ema_angle_700ms_2000ms'] = computeAngle(df['microprice_ema_700ms_ticks'], df['microprice_ema_2000ms_ticks'], 4, 5)
    #ema_feats_df['ema_angle_1200ms_2000ms'] = computeAngle(df['microprice_ema_1200ms_ticks'], df['microprice_ema_2000ms_ticks'])
    #ema_feats_df['ema_angle_1200ms_6000ms'] = computeAngle(df['microprice_ema_1200ms_ticks'], df['microprice_ema_6000ms_ticks'])
    #ema_feats_df['ema_angle_2000ms_6000ms'] = computeAngle(df['microprice_ema_2000ms_ticks'], df['microprice_ema_6000ms_ticks'])
    #ema_feats_df['ema_angle_2000ms_10000ms'] = computeAngle(df['microprice_ema_2000ms_ticks'], df['microprice_ema_10000ms_ticks'])
    #ema_feats_df['ema_angle_6000ms_10000ms'] = computeAngle(df['microprice_ema_6000ms_ticks'], df['microprice_ema_10000ms_ticks'])
    #ema_feats_df['ema_angle_6000ms_26000ms'] = computeAngle(df['microprice_ema_6000ms_ticks'], df['microprice_ema_26000ms_ticks'])
    #ema_feats_df['ema_angle_10000ms_26000ms'] = computeAngle(df['microprice_ema_10000ms_ticks'], df['microprice_ema_26000ms_ticks'])
    #ema_feats_df['ema_angle_10000ms_62000ms'] = computeAngle(df['microprice_ema_10000ms_ticks'], df['microprice_ema_62000ms_ticks'])
    #ema_feats_df['ema_angle_26000ms_62000ms'] = computeAngle(df['microprice_ema_26000ms_ticks'], df['microprice_ema_62000ms_ticks'])
    ema_feats_df['ema_angle_26000ms_150000ms'] = computeAngle(df['microprice_ema_26000ms_ticks'], df['microprice_ema_150000ms_ticks'])
    ema_feats_df['ema_angle_62000ms_150000ms'] = computeAngle(df['microprice_ema_62000ms_ticks'], df['microprice_ema_150000ms_ticks'])
    
    #ema_feats_df['ema_angle_200ms_2000ms'] = computeAngle(df['microprice_ema_200ms_ticks'], df['microprice_ema_2000ms_ticks'], 3, 5)
    ema_feats_df['ema_angle_200ms_6000ms'] = computeAngle(df['microprice_ema_200ms_ticks'], df['microprice_ema_6000ms_ticks'], 3, 5)
    #ema_feats_df['ema_angle_700ms_10000ms'] = computeAngle(df['microprice_ema_700ms_ticks'], df['microprice_ema_10000ms_ticks'], 4, 5)
    #ema_feats_df['ema_angle_700ms_26000ms'] = computeAngle(df['microprice_ema_700ms_ticks'], df['microprice_ema_26000ms_ticks'], 4, 5)

    return (ema_feats_df, df)


def computeRunningMinMaxSignals(df, ema_feats_df, spike_5s, norm):
    max_to_here = pd.expanding_max(df['microprice_ema_200ms_ticks'])
    min_to_here = pd.expanding_min(df['microprice_ema_200ms_ticks'])

    if spike_5s >= 0:
        max_disl = max_to_here
    else:
        max_disl = min_to_here

    ema_feats_df['from_max_disl_ema_200ms_ticks'] = computeRelativeDislocation(df['microprice_ema_200ms_ticks'], max_disl, max_disl)
    ema_feats_df['from_max_disl_ema_2000ms_ticks'] = computeRelativeDislocation(df['microprice_ema_2000ms_ticks'], max_disl, max_disl)
    ema_feats_df['from_max_disl_ema_10000ms_ticks'] = computeRelativeDislocation(df['microprice_ema_10000ms_ticks'], max_disl, max_disl)

    rev_from_max_disl_to_here = df['microprice_ema_200ms_ticks'] - max_disl
    max_rev_to_here = pd.expanding_max(rev_from_max_disl_to_here)
    min_rev_to_here = pd.expanding_min(rev_from_max_disl_to_here)

    if spike_5s >= 0:
        max_rev = max_disl + min_rev_to_here
    else:
        max_rev = max_disl + max_rev_to_here

    ema_feats_df['from_max_rev_ema_200ms_ticks'] = computeRelativeDislocation(df['microprice_ema_200ms_ticks'], max_rev, max_rev)
    ema_feats_df['from_max_rev_ema_2000ms_ticks'] = computeRelativeDislocation(df['microprice_ema_2000ms_ticks'], max_rev, max_rev)
    ema_feats_df['from_max_rev_ema_10000ms_ticks'] = computeRelativeDislocation(df['microprice_ema_10000ms_ticks'], max_rev, max_rev)
        

    return (ema_feats_df, df)


def computeEMALookbackDiffSignals(df, ema_feats_df, norm):
    #ema_feats_df['lookback_diff_ema_200ms'] = (df['microprice_ema_200ms_ticks'] - df['microprice_ema_200ms_ticks'].shift(10)) / df['microprice_ema_200ms_ticks']
    #ema_feats_df['lookback_diff_ema_2000ms'] = (df['microprice_ema_2000ms_ticks'] - df['microprice_ema_2000ms_ticks'].shift(100)) / df['microprice_ema_2000ms_ticks']
    #ema_feats_df['lookback_diff_ema_10000ms'] = (df['microprice_ema_10000ms_ticks'] - df['microprice_ema_10000ms_ticks'].shift(500)) / df['microprice_ema_10000ms_ticks']

    ema_feats_df['lookback_diff_ema_200ms'] = df['microprice_ema_200ms_ticks'].pct_change(periods=10)
    ema_feats_df['lookback_diff_ema_2000ms'] = df['microprice_ema_2000ms_ticks'].pct_change(periods=100)
    ema_feats_df['lookback_diff_ema_10000ms'] = df['microprice_ema_10000ms_ticks'].pct_change(periods=500)

    return (ema_feats_df, df)


def computeRSI(df, n):
    name = "RSI_" + str(n)
    df['UpMoves'] = df['close'] - df['close'].shift(1)
    df['DoMoves'] = df['close'].shift(1) - df['close']
    df['UpI'] = map(lambda u, d: u if (u>d and u>0) else 0, df['UpMoves'], df['DoMoves'])
    df['DoI'] = map(lambda u, d: d if (d>u and d>0) else 0, df['UpMoves'], df['DoMoves'])
    
    df['PosDI'] = pd.ewma(df['UpI'], span = n, min_periods = n - 1) 
    df['NegDI'] = pd.ewma(df['DoI'], span = n, min_periods = n - 1)  
    df['RSI'] = map(lambda posI, negI: posI / (posI + negI), df['PosDI'], df['NegDI'])
    df['RSI_smooth'] = pd.ewma(df['RSI'], span=20) #20
    
    return df


def computeRSISignals(df, ema_feats_df):
    ema_feats_df['RSI'] = computeRSI(df, 50)['RSI']

    return (ema_feats_df, df)


def computeVolumeSignals(df, ema_feats_df):
    df['cum_volume'] = df['abs_vol'].cumsum()

    #ema_feats_df.ix[(ema_feats_df.time<start_dt)&(ema_feats_df.time>stop_dt), 'norm_cum_volume'] = 0
    #ema_feats_df.ix[(ema_feats_df.time>=start_dt)&(ema_feats_df.time<), 'norm_cum_volume']

    df['vol_fast'] = pd.ewma(df['abs_vol'], halflife = 100)
    df['vol_slow'] = pd.ewma(df['abs_vol'], halflife = 2000)
    ema_feats_df['norm_vol_ema'] = df['vol_fast'] / df['vol_slow']

    return (ema_feats_df, df)


def computeImbalance(df, ema_feats_df):
    df['imbalance3_ema'] = pd.ewma(df['imbalance3'], span=25)
    df['imbalance3_ema_long'] = pd.ewma(df['imbalance3'], span=1000)

    ema_feats_df['diff_imbalance_ema_short_long'] = df['imbalance3_ema'] - df['imbalance3_ema_long']
    ema_feats_df['norm_imbalance_ema'] = df['imbalance3_ema'] / df['imbalance3_ema_long']  #1

    return (ema_feats_df, df)


def getTargetUtilities(ema_feats_df, start_dt, max_entry_dt, stop_dt, event_date):
    dir_path = '/local/disk1/temp_eg/trade_util_labels/'
    store_filename = dir_path + event_name + '_' + instrument_root + '_' + event_date.replace('-', '')  + '.h5'
    store = pd.HDFStore(store_filename)
    trades_df = store['trades_df']
    store.close()

    ema_feats_df.ix[(ema_feats_df.time<start_dt)&(ema_feats_df.time>stop_dt), 'buy_trade_utility'] = float('NaN')
    ema_feats_df.ix[(ema_feats_df.time<start_dt)&(ema_feats_df.time>stop_dt), 'sell_trade_utility'] = float('NaN')

    ema_feats_df.ix[(ema_feats_df.time>=start_dt)&(ema_feats_df.time<=stop_dt), 'buy_trade_utility'] = trades_df.ix[ema_feats_df.time, 'buy_duration_penalized_trade_utility']
    ema_feats_df.ix[(ema_feats_df.time>=start_dt)&(ema_feats_df.time<=stop_dt), 'sell_trade_utility'] = trades_df.ix[ema_feats_df.time, 'sell_duration_penalized_trade_utility']

    ema_feats_df.ix[ema_feats_df.time == max_entry_dt, 'buy_trade_utility'] = 0.9 * ema_feats_df.ix[ema_feats_df.time == (max_entry_dt - timedelta(seconds=0.1)), 'buy_trade_utility'][0]
    ema_feats_df.ix[ema_feats_df.time == max_entry_dt, 'sell_trade_utility'] = 0.9 * ema_feats_df.ix[ema_feats_df.time == (max_entry_dt - timedelta(seconds=0.1)), 'sell_trade_utility'][0]

    return ema_feats_df 


def computeEMASignals(df, ema_feats_df, event_date, start_event_dt, start_dt, max_entry_dt, stop_dt, spike_5s, spike_5s_pred):
    start_event_loc = df.index.get_loc(start_event_dt)
    start_loc = df.index.get_loc(start_dt)
    max_entry_loc = df.index.get_loc(max_entry_dt)
    stop_loc = df.index.get_loc(stop_dt)

    norm = float(spike_5s)

    ema_feats_df, df = computeTimeOffsetSignals(df, ema_feats_df, start_event_dt, start_dt, stop_dt, start_event_loc, start_loc, stop_loc)
    ema_feats_df, df = computeEMAs(df, ema_feats_df, start_event_dt)
    #ema_feats_df, df = computeSpikeSignals(df, ema_feats_df, start_event_dt, spike_5s, spike_5s_pred, norm)
    #ema_feats_df, df = computeEMADiffSignals(df, ema_feats_df, norm)
    #ema_feats_df, df = computeEMAAverageSlopeSignals(df, ema_feats_df)
    #ema_feats_df, df = computeEMAAngles(df, ema_feats_df)
    #ema_feats_df, df = computeRunningMinMaxSignals(df, ema_feats_df, spike_5s, norm)
    #ema_feats_df, df = computeEMALookbackDiffSignals(df, ema_feats_df, norm)
    #ema_feats_df, df = computeRSISignals(df, ema_feats_df)
    #ema_feats_df, df = computeVolumeSignals(df, ema_feats_df)
    ema_feats_df, df = computeImbalance(df, ema_feats_df)
    ema_feats_df = getTargetUtilities(ema_feats_df, start_dt, max_entry_dt, stop_dt, event_date)

    return ema_feats_df


def main():
    import timeit


    for event_date in event_dates:
        print event_date
        event_datetime_obj = datetime.strptime(event_date + " " + event_start_time, '%Y-%m-%d %H:%M:%S')
        symbol = mkdt_utils.getSymbol(instrument_root, event_datetime_obj)
        df = mkdt_utils.getMarketDataFrameForTradingDate(event_date, instrument_root, symbol, "100ms")
	if isinstance(df.head(1).time[0], basestring):
            df.time = df.time.apply(mkdt_utils.str_to_dt)
        df.set_index(df['time'], inplace=True)

        start_event_dt = event_datetime_obj
	start_dt = datetime.strptime(event_date + " " + trading_start_time, '%Y-%m-%d %H:%M:%S')
        max_entry_dt = datetime.strptime(event_date + " " + max_entry_time, '%Y-%m-%d %H:%M:%S')
        stop_dt = datetime.strptime(event_date + " " + trading_stop_time, '%Y-%m-%d %H:%M:%S')

	# create dataframe for new features
	ema_feats_df = pd.DataFrame(index = df['time'])
	ema_feats_df['time'] = df['time']

	ema_feats_df = computeEMASignals(df, ema_feats_df, event_date, start_event_dt, start_dt, max_entry_dt, stop_dt, spike_5s[event_date], spike_5s_pred[event_date])
	
	dir_path = '/local/disk1/temp_eg/ema_features/'
        filename = 'ema_feats_' + event_name + '_' + instrument_root + '_' + event_date.replace('-', '')
        store_filename = dir_path + filename + '.h5'
        store = pd.HDFStore(store_filename)
        store['ema_feats_df'] = ema_feats_df
        store.close()

        print "completed processing date: ", event_date


if __name__ == "__main__": main()

