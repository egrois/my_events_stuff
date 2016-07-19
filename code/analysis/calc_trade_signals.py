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
import re

import mkdt_utils
reload(mkdt_utils)
import tech_anal
reload(tech_anal)



"""Event-specific parameter setting"""
event_name = "ADP"
instrument_root = '6E'
event_start_time = '08:15:00'
event_time_offset_s = 0  # for example, for Payrolls can be =1.0
spike_duration_s = 2

event_dates = ['2013-01-03', '2013-03-06', '2013-04-03', '2013-05-01', '2013-06-05', '2013-07-03', '2013-07-31', '2013-09-05', '2013-10-02', '2013-10-30', '2013-12-04', '2014-01-08', '2014-02-05', '2014-03-05', '2014-04-02', '2014-04-30', '2014-06-04', '2014-07-02', '2014-07-30', '2014-09-04', '2014-10-01', '2014-11-05', '2014-12-03', '2015-01-07', '2015-02-04', '2015-03-04', '2015-04-01', '2015-05-06', '2015-06-03', '2015-07-01', '2015-08-05', '2015-09-02']
#event_dates = ['2013-07-03']
#event_dates = ['2015-08-05']


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



spike_2s_ticks_predictions = {'2013-01-03': -16, '2013-03-06': -10, '2013-04-03': 25, '2013-05-01': 21, '2013-06-05': 14, '2013-07-03': -11, '2013-07-31': -8, '2013-09-05': 7, '2013-10-02': 5, '2013-10-30': 4, '2013-12-04': -16, '2014-01-08': -16, '2014-02-05': 8, '2014-03-05': 12, '2014-04-02': -4, '2014-04-30': -9, '2014-06-04': 10, '2014-07-02': -13, '2014-07-30': 2, '2014-09-04': 5, '2014-10-01': -6, '2014-11-05': -8, '2014-12-03': 9, '2015-01-07': -8, '2015-02-04': 5, '2015-03-04': 5, '2015-04-01': 31, '2015-05-06': 22, '2015-06-03': 2, '2015-07-01': -7, '2015-08-05': 27, '2015-09-02': 10}




df_cols_to_npar_map = ['time','abs_vol','buy_vol','sell_vol','high','low','open','close','midquote','spread','microprice','microprice2','top_bid_price','bid_size_0','bid_size_1','bid_size_2','bid_size_3','bid_size_4','top_ask_price','ask_size_0','ask_size_1','ask_size_2','ask_size_3','ask_size_4','bid_num_0','bid_num_1','bid_num_2','bid_num_3','bid_num_4','ask_num_0','ask_num_1','ask_num_2','ask_num_3','ask_num_4','imbalance','imbalance2','imbalance3','ask_vol_ema_200ms','ask_vol_ema_2s','ask_vol_ema_10s','bid_vol_ema_200ms','bid_vol_ema_2s','bid_vol_ema_10s','microprice_ema_200ms','microprice_ema_2s','microprice_ema_10s','trade_price_ema_200ms','trade_price_ema_2s','trade_price_ema_10s']


def i_df_col(col_name):
    return df_cols_to_npar_map.index(col_name)


def computeTimeOffsetFeats(df, feats_df, start_dt, stop_dt, start_loc, stop_loc):
    #feats_df.ix[(feats_df.time<start_dt)&(feats_df.time>stop_dt), 'time_offset_s'] = float('NaN')
    feats_df.ix[(feats_df.time<start_dt)&(feats_df.time>stop_dt), 'offset_intervals'] = float('NaN')

    
    # the stuff here is messed up!
    for i in range(0, start_loc):
    for i in range(start_loc-10, start_loc):  # dealing with this first second as special caase, but will need to change when start_time more correctly set to ~100ms.
	#feats_df.ix[i, 'time_offset_s'] = (i - start_loc + 10) * 0.1
	#feats_df.ix[i, 'offset_intervals'] = i - start_loc + 10
	feats_df.ix[i, 'offset_intervals'] = i 

    for i in range(start_loc, stop_loc):
    for i in range(start_loc, stop_loc):
	#feats_df.ix[i, 'time_offset_s'] = 1.0 + (i - start_loc) * 0.1  # 1.0 accounts for waiting 1s before starting to trade, 0.1 is length of interval in seconds
        #feats_df.ix[i, 'offset_intervals'] = 10 + i - start_loc  # 10 accounts for waiting 1s before starting to trade
	feats_df.ix[i, 'offset_intervals'] = i

    return feats_df


def computeSpikeFeats(df, feats_df, start_dt, start_loc):  # hard-coded for start time at 1s.  Will need to change if different start time.
    pre_event_snapshot_dt = start_dt - timedelta(seconds=2)
    pre_event_snapshot_loc = df.index.get_loc(pre_event_snapshot_dt)
    #print "pre_event_snapshot_dt:", pre_event_snapshot_dt
    pre_event_microprice = df[df.time == pre_event_snapshot_dt]['microprice_ema_2s'][0]

    spike_1s_time_dt = start_dt                          # not independent of start time, so not correct
    spike_2s_time_dt = start_dt + timedelta(seconds=1)   # not independent of start time, so not correct

    1s_loc = 10
    2s_loc = 20 

    high_1s = df.ix[pre_event_snapshot_loc : 1s_loc, 'high'].max()
    #print df.ix[pre_event_snapshot_loc : start_loc, 'high']
    low_1s = df.ix[pre_event_snapshot_loc : 1s_loc, 'low'].min()
    high_2s = max(high_1s, df.ix[start_loc+1 : 2s_loc, 'high'].max())
    low_2s = min(low_1s, df.ix[start_loc+1 : 2s_loc, 'low'].min())

    high_1s_ticks = (high_1s - pre_event_microprice) / float(min_price_increment)
    low_1s_ticks = (low_1s - pre_event_microprice) / float(min_price_increment)
    high_2s_ticks = (high_2s - pre_event_microprice) / float(min_price_increment)
    low_2s_ticks = (low_2s - pre_event_microprice) / float(min_price_increment)

    if abs(high_1s_ticks) >= abs(low_1s_ticks):
	spike_1s_ticks = high_1s_ticks
    else:
        spike_1s_ticks = low_1s_ticks

    if abs(high_2s_ticks) >= abs(low_2s_ticks):
        spike_2s_ticks = high_2s_ticks
    else:
        spike_2s_ticks = low_2s_ticks

    #spike_1s_ticks = (spike_1s_price - pre_event_microprice) / float(min_price_increment)
    #spike_2s_ticks = (spike_2s_price - pre_event_microprice) / float(min_price_increment)

    #print start_dt
    #print high_1s, low_1s, high_2s, low_2s
    #print "pre_event_microprice:", pre_event_microprice
    #print "spike_1s_ticks:", spike_1s_ticks
    #print "spike_2s_ticks:", spike_2s_ticks 

    #feats_df.ix[(feats_df.time<start_dt)&(feats_df.time>stop_dt), 'spike_1s_ticks'] = float('NaN')
    #feats_df.ix[(feats_df.time<start_dt)&(feats_df.time>stop_dt), 'spike_2s_ticks'] = float('NaN')

    #feats_df.ix[(feats_df.time>=start_dt)&(feats_df.time<=stop_dt), 'spike_1s_ticks'] = (spike_1s_price - pre_event_microprice) / float(min_price_increment)
    #feats_df.ix[(feats_df.time>=start_dt)&(feats_df.time<=stop_dt), 'spike_2s_ticks'] = (spike_2s_price - pre_event_microprice) / float(min_price_increment)

    df['microprice_ema_200ms_ticks'] = (df['microprice_ema_200ms'] - pre_event_microprice) / float(min_price_increment)
    df['microprice_ema_2s_ticks'] = (df['microprice_ema_2s'] - pre_event_microprice) / float(min_price_increment)
    df['microprice_ema_10s_ticks'] = (df['microprice_ema_10s'] - pre_event_microprice) / float(min_price_increment)

    feats_df['microprice_ema_200ms_from_spike_1s_ticks'] = df['microprice_ema_200ms_ticks'] - spike_1s_ticks
    feats_df['microprice_ema_200ms_from_spike_2s_ticks'] = df['microprice_ema_200ms_ticks'] - spike_2s_ticks
    feats_df['microprice_ema_2s_from_spike_1s_ticks'] = df['microprice_ema_2s_ticks'] - spike_1s_ticks
    feats_df['microprice_ema_2s_from_spike_2s_ticks'] = df['microprice_ema_2s_ticks'] - spike_2s_ticks
    feats_df['microprice_ema_10s_from_spike_1s_ticks'] = df['microprice_ema_10s_ticks'] - spike_1s_ticks
    feats_df['microprice_ema_10s_from_spike_2s_ticks'] = df['microprice_ema_10s_ticks'] - spike_2s_ticks

    event_date = start_dt.strftime('%Y-%m-%d')
    spike_2s_ticks_pred = spike_2s_ticks_predictions[event_date]

    feats_df['microprice_ema_200ms_from_spike_2s_ticks_pred'] = df['microprice_ema_200ms_ticks'] - spike_2s_ticks_pred
    feats_df['microprice_ema_2s_from_spike_2s_ticks_pred'] = df['microprice_ema_2s_ticks'] - spike_2s_ticks_pred
    feats_df['microprice_ema_10s_from_spike_2s_ticks_pred'] = df['microprice_ema_10s_ticks'] - spike_2s_ticks_pred

    return feats_df


def computeMicropriceEMAdiffsFeats(df, feats_df): 
    #feats_df['microprice_ema_200ms_ticks'] = df['microprice_ema_200ms']
    #feats_df['microprice_ema_2s_ticks'] = df['microprice_ema_2s']
    #feats_df['microprice_ema_10s_ticks'] = df['microprice_ema_10s']

    feats_df['diff_microprice_ema_200ms_2s'] = df['microprice_ema_200ms'] - df['microprice_ema_2s']  #1
    feats_df['diff_microprice_ema_2s_10s'] = df['microprice_ema_2s'] - df['microprice_ema_10s']  #1

    feats_df['d_diff_microprice_ema_200ms_2s'] = feats_df['diff_microprice_ema_200ms_2s'] - feats_df['diff_microprice_ema_200ms_2s'].shift(1)  #1
    feats_df['d_diff_microprice_ema_2s_10s'] = feats_df['diff_microprice_ema_2s_10s'] - feats_df['diff_microprice_ema_2s_10s'].shift(5)  #1

    return feats_df


def computeImbalanceEMAFeats(df, feats_df):
    df['imbalance3_ema'] = pd.ewma(df['imbalance3'], span=25)
    df['imbalance3_ema_long'] = pd.ewma(df['imbalance3'], span=1000)
    
    #feats_df['diff_imbalance_ema_short_long'] = df['imbalance3_ema'] - df['diff_imbalance_ema_short_long']  #1
    feats_df['diff_imbalance_ema_short_long'] = df['imbalance3_ema'] - df['imbalance3_ema_long']  #1

    #feats_df['norm_imbalance_ema'] = df['imbalance3_ema'] / df['diff_imbalance_ema_short_long']  #1
    feats_df['norm_imbalance_ema'] = df['imbalance3_ema'] / df['imbalance3_ema_long']  #1

    return feats_df


def computeBidAskVolumeFeats(df, feats_df):
    df['askbid_vol_ema_200ms'] = df['ask_vol_ema_200ms'] - df['bid_vol_ema_200ms']
    df['askbid_vol_ema_2s'] = df['ask_vol_ema_2s'] - df['bid_vol_ema_2s']
    df['askbid_vol_ema_10s'] = df['ask_vol_ema_10s'] - df['bid_vol_ema_10s']
    feats_df['diff_200ms_2s_askbid_vol_ema'] = df['askbid_vol_ema_200ms'] - df['askbid_vol_ema_2s']  #1
    feats_df['diff_2s_10s_askbid_vol_ema'] = df['askbid_vol_ema_2s'] - df['askbid_vol_ema_10s']  #1
    feats_df['d_diff_200ms_2s_askbid_vol_ema'] = feats_df['diff_200ms_2s_askbid_vol_ema'] - feats_df['diff_200ms_2s_askbid_vol_ema'].shift(1)  #1
    feats_df['d_diff_2s_10s_askbid_vol_ema'] = feats_df['diff_2s_10s_askbid_vol_ema'] - feats_df['diff_2s_10s_askbid_vol_ema'].shift(2)  #1

    feats_df['norm_askbid_vol_ema_200ms'] = (df['ask_vol_ema_200ms'] - df['bid_vol_ema_200ms']) / (df['ask_vol_ema_200ms'] + df['bid_vol_ema_200ms'])  #1
    feats_df['norm_askbid_vol_ema_2s'] = (df['ask_vol_ema_2s'] - df['bid_vol_ema_2s']) / (df['ask_vol_ema_2s'] + df['bid_vol_ema_2s'])  #1
    feats_df['norm_askbid_vol_ema_10s'] = (df['ask_vol_ema_10s'] - df['bid_vol_ema_10s']) / (df['ask_vol_ema_10s'] + df['bid_vol_ema_10s'])  #1
    feats_df['diff_norm_200ms_2s_askbid_vol_ema'] = feats_df['norm_askbid_vol_ema_200ms'] - feats_df['norm_askbid_vol_ema_2s']  #1
    feats_df['diff_norm_2s_10s_askbid_vol_ema'] = feats_df['norm_askbid_vol_ema_2s'] - feats_df['norm_askbid_vol_ema_10s']  #1
    
    df['smooth_norm_askbid_vol_ema_200ms'] = pd.ewma(feats_df['norm_askbid_vol_ema_200ms'], span=25)
    df['smooth_norm_askbid_vol_ema_2s'] = pd.ewma(feats_df['norm_askbid_vol_ema_2s'], span=25)
    df['smooth_norm_askbid_vol_ema_10s'] = pd.ewma(feats_df['norm_askbid_vol_ema_10s'], span=25)
    feats_df['smooth_diff_norm_200ms_2s_askbid_vol_ema'] = pd.ewma(feats_df['diff_norm_200ms_2s_askbid_vol_ema'], span=25)  #1
    feats_df['smooth_diff_norm_2s_10s_askbid_vol_ema'] = pd.ewma(feats_df['diff_norm_2s_10s_askbid_vol_ema'], span=25)  #1

    feats_df['d_smooth_diff_norm_200ms_2s_askbid_vol_ema'] = feats_df['smooth_diff_norm_200ms_2s_askbid_vol_ema'] - feats_df['smooth_diff_norm_200ms_2s_askbid_vol_ema'].shift(1)  #1
    feats_df['d_smooth_diff_norm_2s_10s_askbid_vol_ema'] = feats_df['smooth_diff_norm_2s_10s_askbid_vol_ema'] - feats_df['smooth_diff_norm_2s_10s_askbid_vol_ema'].shift(5)  #1

    return feats_df


def computeMACDFeats(df, feats_df):
    df = tech_anal.MACD(df, 'microprice', 60, 150, 40)
    # include MACDdiff here
    feats_df['MACDdiff'] = df['MACDdiff']
    feats_df['d_MACDdiff'] = df['MACDdiff'] - df['MACDdiff'].shift(5)

    return feats_df


def getTargetUtilities(feats_df, start_dt, max_entry_dt, stop_dt, event_date):
    dir_path = '/local/disk1/temp_eg/trade_util_labels/'
    store_filename = dir_path + event_name + '_' + instrument_root + '_' + event_date.replace('-', '')  + '.h5'
    store = pd.HDFStore(store_filename)
    trades_df = store['trades_df']
    store.close()

    feats_df.ix[(feats_df.time<start_dt)&(feats_df.time>stop_dt), 'buy_trade_utility'] = float('NaN')
    feats_df.ix[(feats_df.time<start_dt)&(feats_df.time>stop_dt), 'sell_trade_utility'] = float('NaN')

    feats_df.ix[(feats_df.time>=start_dt)&(feats_df.time<=stop_dt), 'buy_trade_utility'] = trades_df.ix[feats_df.time, 'buy_duration_penalized_trade_utility']
    feats_df.ix[(feats_df.time>=start_dt)&(feats_df.time<=stop_dt), 'sell_trade_utility'] = trades_df.ix[feats_df.time, 'sell_duration_penalized_trade_utility']

    # This is a temporary solution to provide a valid utility value for the a buy/sell trade occurrinf exactly at max_entry_time.  
    # The proper change needs to be made in the utility computation script by extending the main iteration loop upper bound to max_entry_loc+1.
    # Then the two lines below should be removed.
    feats_df.ix[feats_df.time == max_entry_dt, 'buy_trade_utility'] = 0.9 * feats_df.ix[feats_df.time == (max_entry_dt - timedelta(seconds=0.1)), 'buy_trade_utility'][0]
    feats_df.ix[feats_df.time == max_entry_dt, 'sell_trade_utility'] = 0.9 * feats_df.ix[feats_df.time == (max_entry_dt - timedelta(seconds=0.1)), 'sell_trade_utility'][0]
    #print "buy_trade_utility at ", max_entry_dt, feats_df.ix[feats_df.time == max_entry_dt, 'buy_trade_utility'][0]
    #print "sell_trade_utility at ", max_entry_dt, feats_df.ix[feats_df.time == max_entry_dt, 'sell_trade_utility'][0]

    return feats_df


def computeFeatures(df, feats_df, start_dt, max_entry_dt, stop_dt, event_date):
    start_loc = df.index.get_loc(start_dt)
    max_entry_loc = df.index.get_loc(max_entry_dt)
    stop_loc = df.index.get_loc(stop_dt)

    feats_df = computeTimeOffsetFeats(df, feats_df, start_dt, stop_dt, start_loc, stop_loc)
    feats_df = computeSpikeFeats(df, feats_df, start_dt, start_loc)
    feats_df = computeMicropriceEMAdiffsFeats(df, feats_df)
    feats_df = computeImbalanceEMAFeats(df, feats_df)
    feats_df = computeBidAskVolumeFeats(df, feats_df)
    feats_df = computeMACDFeats(df, feats_df)

    # get features for prior intervals 
    all_feats = {}
    for i in range(0,10):
        all_feats['_' + str(i)] = feats_df.shift(i)  # e.g., '_0'

    for key, ff in all_feats.iteritems():
        ff.rename(columns=lambda x: x+key, inplace=True)
        ff.drop('time'+key, axis=1, inplace=True)

    ff_all = pd.concat(all_feats.values(), axis=1, join_axes=[feats_df.index])
    
    #delete all of this
    #for i in range(0,10):
    #    time_col = 'time_' + str(i)
    #    if time_col in train_df.columns:
    #        train_df.drop(time_col, axis=1, inplace=True)
    #    if time_col in test_df.columns:
    #        test_df.drop(time_col, axis=1, inplace=True)

    #feats_df = ff_all.select(lambda x: not re.search('time', x), axis=1)
    #feats_df.drop('time', axis=1, inplace=True)
    feats_df = ff_all
    feats_df['time'] = df['time']
    feats_df = getTargetUtilities(feats_df, start_dt, max_entry_dt, stop_dt, event_date)

    return feats_df


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

	data_npar = df[df_cols_to_npar_map].values

	# create separate dataframe for features
	feats_df = pd.DataFrame(index = df['time'])
	feats_df['time'] = df['time']

	start_dt = datetime.strptime(event_date + " " + trading_start_time, '%Y-%m-%d %H:%M:%S')
	max_entry_dt = datetime.strptime(event_date + " " + max_entry_time, '%Y-%m-%d %H:%M:%S')
	stop_dt = datetime.strptime(event_date + " " + trading_stop_time, '%Y-%m-%d %H:%M:%S')

	feats_df = computeFeatures(df, feats_df, start_dt, max_entry_dt, stop_dt, event_date)
	
	dir_path = '/local/disk1/temp_eg/features/'
	filename = 'feats_' + event_name + '_' + instrument_root + '_' + event_date.replace('-', '')
        store_filename = dir_path + filename + '.h5'
	store = pd.HDFStore(store_filename)
        store['feats_df'] = feats_df
        store.close()

if __name__ == "__main__": main()

