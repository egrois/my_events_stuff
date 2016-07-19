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
import calc_trade_signals
reload(calc_trade_signals)



"""Event-specific parameter setting"""
event_name = "ADP"
instrument_root = '6E'
event_start_time = '08:15:00'
event_time_offset_s = 0  # for example, for Payrolls can be =1.0
spike_duration_s = 2

event_dates = ['2013-01-03', '2013-03-03', '2013-04-03', '2013-05-01', '2013-06-05', '2013-07-03', '2013-07-31', '2013-09-05', '2013-10-02', '2013-10-30', '2013-12-04', '2014-01-08', '2014-02-05', '2014-03-05', '2014-04-02', '2014-04-30', '2014-06-04', '2014-07-02', '2014-07-30', '2014-09-04', '2014-10-01', '2014-11-05', '2014-12-03', '2015-01-07', '2015-02-04', '2015-03-04', '2015-04-01', '2015-05-06', '2015-06-03', '2015-07-01', '2015-08-05', '2015-09-02']
#event_dates = ['2013-07-03']

"""Parameter settings for valuing an entry/trade.  These are specific to ADP and 6E (for now)."""
min_price_increment = 1  # need to have a lookup structure for these
book_depth = 5
nominal_trading_size = 100  # num contracts to go for at each entry
trading_start_time = "08:15:01"
max_entry_time = "08:25:00"
trading_stop_time = "08:27:00"
max_hold_period_s = 120
min_profit_ticks = 4
stop_loss_ticks = 8
exit_depth = 0  # cross book and trade at top of book



spike_2s_ticks_predictions = {'2013-01-03': -16, '2013-03-06': -10, '2013-04-03': 25, '2013-05-01': 21, '2013-06-05': 14, '2013-07-03': -11, '2013-07-31': -8, '2013-09-05': 7, '2013-10-02': 5, '2013-10-30': 4, '2013-12-04': -16, '2014-01-08': -16, '2014-02-05': 8, '2014-03-05': 12, '2014-04-02': -4, '2014-04-30': -9, '2014-06-04': 10, '2014-07-02': -13, '2014-07-30': 2, '2014-09-04': 5, '2014-10-01': -6, '2014-11-05': -8, '2014-12-03': 9, '2015-01-07': -8, '2015-02-04': 5, '2015-03-04': 5, '2015-04-01': 31, '2015-05-06': 22, '2015-06-03': 2, '2015-07-01': -7, '2015-08-05': 27, '2015-09-02': 10}


def computeTimeOffsetSignals(df, feats_df, abstr_feats_df):
    abstr_feats_df['offset_intervals'] = feats_df['offset_intervals']

    return abstr_feats_df


def computeSpikeSignals(df, feats_df, abstr_feats_df):
    abstr_feats_df['mp_ema_fast_from_spike_1s'] = feats_df['microprice_ema_200ms_from_spike_1s_ticks']
    abstr_feats_df['mp_ema_med_from_spike_1s'] = feats_df['microprice_ema_2s_from_spike_1s_ticks']
    abstr_feats_df['mp_ema_slow_from_spike_1s'] = feats_df['microprice_ema_10s_from_spike_1s_ticks']
    abstr_feats_df['mp_ema_fast_from_spike_2s'] = feats_df['microprice_ema_200ms_from_spike_2s_ticks']
    abstr_feats_df['mp_ema_med_from_spike_2s'] = feats_df['microprice_ema_2s_from_spike_2s_ticks']
    abstr_feats_df['mp_ema_slow_from_spike_2s'] = feats_df['microprice_ema_10s_from_spike_2s_ticks']

    abstr_feats_df['mp_ema_fast_from_spike_2s_pred'] = feats_df['microprice_ema_200ms_from_spike_2s_ticks_pred']
    abstr_feats_df['mp_ema_med_from_spike_2s_pred'] = feats_df['microprice_ema_2s_from_spike_2s_ticks_pred']
    abstr_feats_df['mp_ema_slow_from_spike_2s_pred'] = feats_df['microprice_ema_10s_from_spike_2s_ticks_pred']

    abstr_feats_df['spike_1s_from_spike_2s_pred'] = abstr_feats_df['mp_ema_slow_from_spike_1s'] - abstr_feats_df['mp_ema_slow_from_spike_2s_pred']
    abstr_feats_df['spike_2s_from_spike_2s_pred'] = abstr_feats_df['mp_ema_slow_from_spike_2s'] - abstr_feats_df['mp_ema_slow_from_spike_2s_pred']

    return abstr_feats_df
        

def computeMicropriceSignals(df, feats_df, abstr_feats_df):
    df['diff_microprice_ema_fast_med'] = df['microprice_ema_200ms'] - df['microprice_ema_2s']
    df['diff_microprice_ema_fast_med_smooth'] =  pd.ewma(df['diff_microprice_ema_fast_med'], span=10)
    df['diff_diff_microprice_ema_fast_med'] = df['diff_microprice_ema_fast_med'] - df['diff_microprice_ema_fast_med_smooth']

    df['test_mp_buy_signal'] = map(lambda a, b: math.log(-1*a*b+1) if (a<0 and b>0) else math.log(math.log(a*b+1)+1) if (a<0 and b<0) else 0, \
                                df['diff_microprice_ema_fast_med'], df['diff_diff_microprice_ema_fast_med'])
    df['test_mp_sell_signal'] = map(lambda a, b: math.log(-1*a*b+1) if (a>0 and b<0) else math.log(math.log(a*b+1)+1) if (a>0 and b>0) else 0, \
                                df['diff_microprice_ema_fast_med'], df['diff_diff_microprice_ema_fast_med'])

    abstr_feats_df['mp_signal'] = map(lambda b, s: b if b>s else -1.0*s if s>b else 0, df['test_mp_buy_signal'], df['test_mp_sell_signal'])  #signed version of buy/sell signals

    return abstr_feats_df    


def computeEMAAverageSlopeSignals(df, feats_df, abstr_feats_df):
    offset = 5  # num intervals to lookback to compute slope
  
    df['microprice_ema_fast_med_avg_slope'] = ((df['microprice_ema_200ms'] - df['microprice_ema_200ms'].shift(offset)) \
                                               + (df['microprice_ema_2s'] - df['microprice_ema_2s'].shift(offset))) / float(2)
    
    df['test_mp_avg_slope_buy_signal'] = map(lambda a: math.log(-1*a+1) if (a<0) else 0, df['microprice_ema_fast_med_avg_slope'])
    df['test_mp_avg_slope_sell_signal'] = map(lambda a: math.log(a+1) if (a>0) else 0, df['microprice_ema_fast_med_avg_slope'])

    abstr_feats_df['ema_avg_slope_signal'] = map(lambda b, s: b if b>s else -1.0*s if s>b else 0, df['test_mp_avg_slope_buy_signal'], df['test_mp_avg_slope_sell_signal'])  #signed version of buy/sell signals  

    return abstr_feats_df


def computeMACDSignals(df, feats_df, abstr_feats_df):
    df = tech_anal.MACD(df, 'microprice', 60, 150, 40)
    df['MACDdiff_smooth'] = pd.ewma(df['MACDdiff'], span=25)
    df['diff_MACDdiff_MACDdiff_smooth'] = df['MACDdiff'] - df['MACDdiff_smooth']
    df['cos_slope_MACDdiff'] = map(lambda a: math.cos(3*a), df['diff_MACDdiff_MACDdiff_smooth'])
    df['test_MACDdiff_signal'] = df['cos_slope_MACDdiff'] * df['MACDdiff']

    abstr_feats_df['abstr_MACDdiff_signal'] = map(lambda a: -1*np.sign(a)*math.log(np.sign(a)*a+1), df['test_MACDdiff_signal'])

    return abstr_feats_df


def computeImbalanceSignals(df, feats_df, abstr_feats_df):
    df['ema_fast_imbalance3'] = pd.ewma(df['imbalance3'], span=15)

    abstr_feats_df['abstr_imbalance_signal'] = map(lambda a: np.sign(a)*math.log(np.sign(a)*a+1), df['ema_fast_imbalance3'])

    return abstr_feats_df


def computeDirectionalVolumeFeats(df, feats_df, abstr_feats_df):
    df['diff_direct_vol_fast_med'] = feats_df['diff_200ms_2s_askbid_vol_ema']  # same as df['diff_direct_vol_fast_med']
    df['diff_direct_vol_med_slow'] = feats_df['diff_2s_10s_askbid_vol_ema']  # same as df['diff_direct_vol_med_slow']
    abstr_feats_df['diff_dir_vol_signal'] = df['diff_direct_vol_fast_med']

    df['dir_vol_ratio_2s'] = df['ask_vol_ema_2s'] / df['bid_vol_ema_2s']
    df['dir_vol_ratio_10s'] = df['ask_vol_ema_10s'] / df['bid_vol_ema_10s']
    df['diff_dir_vol_ratios_med_slow'] = df['dir_vol_ratio_2s'] / df['dir_vol_ratio_10s']
    df['diff_dir_vol_ratios_med_slow_smooth'] = pd.ewma(df['diff_dir_vol_ratios_med_slow'], span=15)
    df['diff_dir_vol_ratios_med_slow_smooth_more'] = pd.ewma(df['diff_dir_vol_ratios_med_slow'], span=60)
    df['diff_diff_dir_vol_ratios'] = df['diff_dir_vol_ratios_med_slow_smooth'] - df['diff_dir_vol_ratios_med_slow_smooth_more']
    abstr_feats_df['ratios_dir_vol_signal'] = df['diff_diff_dir_vol_ratios']

    df['norm_diff_vol_fast_med'] = feats_df['diff_norm_200ms_2s_askbid_vol_ema']  # same as df['norm_diff_vol_fast_med']
    df['norm_diff_vol_med_slow'] = feats_df['diff_norm_2s_10s_askbid_vol_ema']  # same as df['norm_diff_vol_med_slow']
    df['norm_diff_vol_fast_med_smooth'] = pd.ewma(df['norm_diff_vol_fast_med'], span=15)
    df['norm_diff_vol_med_slow_smooth'] = pd.ewma(df['norm_diff_vol_med_slow'], span=15)
    abstr_feats_df['norm_dir_vol_fast_med_signal'] = df['norm_diff_vol_fast_med_smooth']
    abstr_feats_df['norm_dir_vol_med_slow_signal'] = df['norm_diff_vol_med_slow_smooth']

    return abstr_feats_df


def getTargetUtilities(feats_df, abstr_feats_df):
    abstr_feats_df['buy_trade_utility'] = feats_df['buy_trade_utility']
    abstr_feats_df['sell_trade_utility'] = feats_df['sell_trade_utility']

    return abstr_feats_df


def computeAbstractFeatures(df, feats_df, abstr_feats_df):
    abstr_feats_df = computeTimeOffsetSignals(df, feats_df, abstr_feats_df)
    abstr_feats_df = computeSpikeSignals(df, feats_df, abstr_feats_df)
    abstr_feats_df = computeMicropriceSignals(df, feats_df, abstr_feats_df) 
    abstr_feats_df = computeEMAAverageSlopeSignals(df, feats_df, abstr_feats_df)
    abstr_feats_df = computeMACDSignals(df, feats_df, abstr_feats_df)
    abstr_feats_df = computeImbalanceSignals(df, feats_df, abstr_feats_df) 
    abstr_feats_df = computeDirectionalVolumeFeats(df, feats_df, abstr_feats_df)

    abstr_feats_df = getTargetUtilities(feats_df, abstr_feats_df)
    
    return abstr_feats_df


def retrieveBaseFeatsForDate(event_date, start_dt, max_entry_dt, stop_dt):
    dir_path = '/local/disk1/temp_eg/features/'
    filename = 'feats_' + event_name + '_' + instrument_root + '_' + event_date.replace('-', '')
    store_filename = dir_path + filename + '.h5'
    store = pd.HDFStore(store_filename)
    feats_df = store['feats_df']
    store.close()
    
    no_lookback_cols = [col for col in feats_df.columns if ('_0' in col or 'utility' in col)]

    ff = feats_df[no_lookback_cols]
    clean_cols = [col.replace('_0', '') for col in no_lookback_cols]
    ff.columns = clean_cols 

    return ff


def main():
    import timeit
    for event_date in event_dates:
        print event_date
        event_datetime_obj = datetime.strptime(event_date + " " + event_start_time, '%Y-%m-%d %H:%M:%S')
        symbol = mkdt_utils.getSymbol(instrument_root, event_datetime_obj)
        df = mkdt_utils.getMarketDataFrameForTradingDate(event_date, instrument_root, symbol, "100ms")
        df.set_index(df['time'], inplace=True)
	
	#print type(df) 
	#print df.describe()

        start_dt = datetime.strptime(event_date + " " + trading_start_time, '%Y-%m-%d %H:%M:%S')
        max_entry_dt = datetime.strptime(event_date + " " + max_entry_time, '%Y-%m-%d %H:%M:%S')
        stop_dt = datetime.strptime(event_date + " " + trading_stop_time, '%Y-%m-%d %H:%M:%S')

	# retrieve base features
        base_feats_df = retrieveBaseFeatsForDate(event_date, start_dt, max_entry_dt, stop_dt)

	#create dataframe for abstract features
	abstr_feats_df = pd.DataFrame(index = df['time'])
	abstr_feats_df['time'] = df['time']

	#print type(abstr_feats_df)

        abstr_feats_df = computeAbstractFeatures(df, base_feats_df, abstr_feats_df)

	#print type(abstr_feats_df)
	#print abstr_feats_df.describe()
	#print abstr_feats_df.head(50)

	dir_path = '/local/disk1/temp_eg/abstr_features/'
        filename = 'abstr_feats_' + event_name + '_' + instrument_root + '_' + event_date.replace('-', '')
        store_filename = dir_path + filename + '.h5'
        store = pd.HDFStore(store_filename)
        store['abstr_feats_df'] = abstr_feats_df
        store.close()
        
        print "completed processing date: ", event_date

	

if __name__ == "__main__": main()	
	

