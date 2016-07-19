import sys
import os
from itertools import izip
sys.path.append('/home/egrois/git/code/pybt')
sys.path.append('/home/egrois/git/code/preprocess')
sys.path.append('/home/egrois/git/code/analysis')
sys.path.append('/home/egrois/git/code/backtest')
import pybt
reload(pybt)
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
from datetime import datetime,time, timedelta
import pandas as pd
import utils
reload(utils)
import matplotlib.finance

import csv
import math

import mkdt_utils
reload(mkdt_utils)
import tech_anal
reload(tech_anal)

import simulator
reload(simulator)



"""Event-specific parameter setting"""
event_name = "ADP"
instrument_root = '6E'
event_start_time = '08:15:00'
event_time_offset_s = 0  # for example, for Payrolls can be =1.0
spike_duration_s = 2

event_dates = ['2013-04-03', '2013-05-01', '2013-06-05', '2013-07-03', '2013-07-31', '2013-09-05', '2013-10-02', '2013-10-30', '2013-12-04', '2014-01-08', '2014-02-05', '2014-03-05', '2014-04-02', '2014-04-30', '2014-06-04', '2014-07-02', '2014-07-30', '2014-09-04', '2014-10-01', '2014-11-05', '2014-12-03', '2015-01-07', '2015-02-04', '2015-03-04', '2015-04-01', '2015-05-06', '2015-06-03', '2015-07-01', '2015-08-05', '2015-09-02']
#event_dates = ['2013-04-03']
event_dates = ['2013-05-01']
event_dates = ['2013-05-01', '2013-06-05', '2013-07-03']

spike_5s_pred = {'2013-01-03': -16, '2013-03-06': -10, '2013-04-03': 26, '2013-05-01': 20, '2013-06-05': 18, '2013-07-03': -13, '2013-07-31': -7, '2013-09-05': 7, '2013-10-02': 5, '2013-10-30': 6, '2013-12-04': -20, '2014-01-08': -18, '2014-02-05': 10, '2014-03-05': 12, '2014-04-02': -5, '2014-04-30': -9, '2014-06-04': 18, '2014-07-02': -21, '2014-07-30': 6, '2014-09-04': 9, '2014-10-01': -5, '2014-11-05': -8, '2014-12-03': 8, '2015-01-07': -13, '2015-02-04': 3, '2015-03-04': 3, '2015-04-01': 27, '2015-05-06': 23, '2015-06-03': -1, '2015-07-01': -9, '2015-08-05': 21, '2015-09-02': 7}

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




from enum import Enum
class TradeDirection(Enum):
    bullish = 1
    bearish = 2
    neutral = 3

class MarketTrend(Enum):
    bullish = 1
    bearish = 2
    hyper_bullish = 3
    hyper_bearish = 4
    neutral = 5



class Strategy:
    dollar_value_per_price_level = 0
    df = None
    log_df = None
    start_dt = None
    max_entry_dt = None
    stop_dt = None
    start_loc = 0
    max_entry_loc = 0
    stop_loc = 0
    mySim = None

    cur_loc = 0

    _pos = 0
    _cash = 0
    _vol = 0
    _ltp = 0
    _pnl = 0

    spike_5s_pred = 0
    tradeDirection = TradeDirection.neutral
    spike_ticks = 0

    trend = MarketTrend.neutral
    trend2 = MarketTrend.neutral
    trend3 = MarketTrend.neutral
    last_trend = MarketTrend.neutral
    atr_stop = np.nan    


    def __init__(self, df, start_dt, max_entry_dt, stop_dt, Sim, dollar_value_per_price_level, spike_5s_pred):
        self.df = df
        self.log_df = pd.DataFrame(index = df['time'], columns=['buy_entry', 'sell_entry', 'buy_exit', 'sell_exit', 'buy_exit_invert', 'sell_exit_invert'])
	self.log_df.fillna(np.nan)
        self.log_df['time'] = df['time']
        self.mySim = Sim
	self.dollar_value_per_price_level = dollar_value_per_price_level
	
	self.start_dt = start_dt
        self.start_loc = df.index.get_loc(start_dt)
        self.max_entry_dt = max_entry_dt
        self.max_entry_loc = df.index.get_loc(max_entry_dt)
        self.stop_dt = stop_dt
        self.stop_loc = df.index.get_loc(stop_dt)


        self.spike_5s_pred = spike_5s_pred
        self.df = self.compute_data(self.df)  # compute additional studies and signals to be used by the strategy        

	self.trend = MarketTrend.neutral
	self.trend2 = MarketTrend.neutral
	self.trend3 = MarketTrend.neutral


    def start(self):
        self.cur_loc  = self.mySim.start_sim(self.start_dt)
	#print "start:", self.cur_loc
	self.trend2 = MarketTrend.neutral

        while (self.cur_loc < self.stop_loc):
	    #print "while:", self.cur_loc
	    self.cur_loc, fills = self.onTick()
	    self.update(self.cur_loc, fills)       

	    if fills:
	        print "        pos: ", self._pos
                print "        PnL: ", self._pnl

		print "        atr_high_bound: ", self.df.ix[self.cur_loc-1, 'atr_high_bound']
		print "        atr_low_bound: ", self.df.ix[self.cur_loc-1, 'atr_low_bound']
		print "        atr_stop: ", self.log_df.ix[self.cur_loc-1, 'atr_stop']
		print "        atr_stop_bot: ", self.log_df.ix[self.cur_loc-1, 'atr_stop_bot']
		print "        atr_stop_top: ", self.log_df.ix[self.cur_loc-1, 'atr_stop_top']
		print "        trend: ", self.df.ix[self.cur_loc-1, 'trend']
		print "        trend2: ", self.trend2
		print "        trend3: ", self.trend3

        return self.log_df


    def update(self, mkt_data_loc, fills):
        #self.cur_loc = mkt_data_loc
        self.process_fills(fills)
        self._pnl = self.computePnL()

        #self.onTick()


    def computeEMAs(self, df):
        df['microprice_ema_700ms'] = pd.ewma(df['microprice'], halflife=7)
        df['microprice_ema_1200ms'] = pd.ewma(df['microprice'], halflife=12)
        df['microprice_ema_3600ms'] = pd.ewma(df['microprice'], halflife=36)
        df['microprice_ema_6000ms'] = pd.ewma(df['microprice'], halflife=60)
        df['microprice_ema_26000ms'] = pd.ewma(df['microprice'], halflife=260)
        df['microprice_ema_62000ms'] = pd.ewma(df['microprice'], halflife=620)
        df['microprice_ema_150000ms'] = pd.ewma(df['microprice'], halflife=1500)    

        return df


    def computeMACD(self, df):
	df = tech_anal.MACD(df, 'microprice', 60, 150, 40)	
	df['MACDsmooth'] = pd.ewma(df['MACD'], span=15)

	return df
    
    
    def computeATR(self, df, n):
	df['high_s'] = map(lambda h, mp, mq, a: h if (not math.isnan(h)) else min(max(mp, mq), a), df['high'], df['microprice'], df['midquote'], df['top_ask_price'])    
        df['low_s'] = map(lambda l, mp, mq, b: l if (not math.isnan(l)) else max(min(mp, mq), b), df['low'], df['microprice'], df['midquote'], df['top_bid_price'])
        df['close_s'] = map(lambda c, mq: c if (not math.isnan(c)) else mq, df['close'], df['midquote'])
        df['prev_close_s'] = map(lambda c, mq: c if (not math.isnan(c)) else mq, df['close'].shift(1), df['midquote'].shift(1))
    
        df['tr1'] = df['high_s'] - df['low_s']
        df['tr2'] = map(lambda a, b: abs(a - b), df['high_s'], df['prev_close_s'])
        df['tr3'] = map(lambda a, b: abs(a - b), df['low_s'], df['prev_close_s'])
    
        df['tr'] = map(lambda tr1, tr2, tr3: max(tr1, max(tr2, tr3)), df['tr1'], df['tr2'], df['tr3'])
        df['atr'] = pd.ewma(df['tr'], span=n, min_periods=n)
    
	#df['atr_high_bound'] = map(lambda base, offset: base + min(12.0 * offset, 30), df['close_s'], df['atr'])  # offset is basically ticks
        #df['atr_low_bound'] = map(lambda base, offset: base - min(12.0 * offset, 30), df['close_s'], df['atr'])
	#df['atr_high_bound'] = df['close_s'] + 10.0 * df['atr']
	#df['atr_low_bound'] = df['close_s'] - 10.0 * df['atr']

	df['atr_high_bound'] = df['microprice_ema_200ms'] + 10.0 * df['atr']
	df['atr_low_bound'] = df['microprice_ema_200ms'] - 10.0 * df['atr']

        return df


    def computeATRTrailingStop(self, df, log_df, trend, last_trend):
	if trend == MarketTrend.bullish or trend == MarketTrend.hyper_bullish:
	    if last_trend != MarketTrend.bullish and last_trend != MarketTrend.hyper_bullish:
		log_df.ix[self.cur_loc, 'atr_stop'] = df.ix[self.cur_loc, 'atr_low_bound']
	    else:
		if np.isnan(log_df.ix[self.cur_loc-1, 'atr_stop']):
		    log_df.ix[self.cur_loc, 'atr_stop'] = df.ix[self.cur_loc, 'atr_low_bound']
		else:
		    log_df.ix[self.cur_loc, 'atr_stop'] = max(log_df.ix[self.cur_loc-1, 'atr_stop'], df.ix[self.cur_loc, 'atr_low_bound'])
	elif trend == MarketTrend.bearish or trend == MarketTrend.hyper_bearish:
	    if last_trend != MarketTrend.bearish and last_trend != MarketTrend.hyper_bearish:
		log_df.ix[self.cur_loc, 'atr_stop'] = df.ix[self.cur_loc, 'atr_high_bound']
	    else:
		if np.isnan(log_df.ix[self.cur_loc-1, 'atr_stop']):
		    log_df.ix[self.cur_loc, 'atr_stop'] = df.ix[self.cur_loc, 'atr_high_bound']
		else:
		    log_df.ix[self.cur_loc, 'atr_stop'] = min(log_df.ix[self.cur_loc-1, 'atr_stop'], df.ix[self.cur_loc, 'atr_high_bound'])    

	return df, log_df


    def computeNewATRTrailingStop(self, df, log_df, trend, last_trend):
	#if np.isnan(log_df.ix[self.cur_loc-1, 'atr_stop_bot']) and np.isnan(log_df.ix[self.cur_loc-1, 'atr_stop_top']):
	#    log_df.ix[self.cur_loc, 'atr_stop_bot'] = df.ix[self.cur_loc, 'atr_low_bound']
	#    log_df.ix[self.cur_loc, 'atr_stop_top'] = df.ix[self.cur_loc, 'atr_high_bound']
	if np.isnan(log_df.ix[self.cur_loc-1, 'atr_stop_bot']):
	    log_df.ix[self.cur_loc-1, 'atr_stop_bot'] = df.ix[self.cur_loc-1, 'atr_low_bound']
	if np.isnan(log_df.ix[self.cur_loc-1, 'atr_stop_top']):
	    log_df.ix[self.cur_loc-1, 'atr_stop_top'] = df.ix[self.cur_loc-1, 'atr_high_bound']
	
	#print "before: ", log_df.ix[self.cur_loc, 'atr_stop_top'], log_df.ix[self.cur_loc, 'atr_stop_bot']
	#if np.isnan(log_df.ix[self.cur_loc, 'atr_stop_bot']):
	#    print "atr_stop_bot is NAN"
        #    log_df.ix[self.cur_loc, 'atr_stop_bot'] = df.ix[self.cur_loc, 'atr_low_bound']
        #if np.isnan(log_df.ix[self.cur_loc, 'atr_stop_top']):
	#    print "atr_stop_top is NAN"
        #    log_df.ix[self.cur_loc, 'atr_stop_top'] = df.ix[self.cur_loc, 'atr_high_bound']
	
	#if df.ix[self.cur_loc-1, 'microprice'] > log_df.ix[self.cur_loc-1, 'atr_stop_top']:
	#    log_df.ix[self.cur_loc, 'atr_stop_bot'] = max(log_df.ix[self.cur_loc-1, 'atr_stop_bot'], df.ix[self.cur_loc, 'atr_low_bound'])
	#    log_df.ix[self.cur_loc, 'atr_stop_top'] = df.ix[self.cur_loc, 'atr_high_bound']
	#    self.trend2 = MarketTrend.bullish
	#elif df.ix[self.cur_loc-1, 'microprice'] < log_df.ix[self.cur_loc-1, 'atr_stop_bot']:
	#    log_df.ix[self.cur_loc, 'atr_stop_top'] = min(log_df.ix[self.cur_loc-1, 'atr_stop_top'], df.ix[self.cur_loc, 'atr_high_bound'])
	#    log_df.ix[self.cur_loc, 'atr_stop_bot'] = df.ix[self.cur_loc, 'atr_low_bound']
	#    self.trend2 = MarketTrend.bearish
	#else:
	#    log_df.ix[self.cur_loc, 'atr_stop_bot'] = max(log_df.ix[self.cur_loc-1, 'atr_stop_bot'], df.ix[self.cur_loc, 'atr_low_bound'])
	#    log_df.ix[self.cur_loc, 'atr_stop_top'] = min(log_df.ix[self.cur_loc-1, 'atr_stop_top'], df.ix[self.cur_loc, 'atr_high_bound'])
	#    self.trend2 = MarketTrend.neutral

	#print "    trend:", self.trend2, df.ix[self.cur_loc, 'microprice'], log_df.ix[self.cur_loc, 'atr_stop_top'], log_df.ix[self.cur_loc, 'atr_stop_bot']
	if self.trend2 == MarketTrend.neutral and (self.trend == MarketTrend.bullish or self.trend == MarketTrend.hyper_bullish) \
			and df.ix[self.cur_loc-1, 'MACD'] > df.ix[self.cur_loc-1, 'MACDsign'] \
			and df.ix[self.cur_loc-1, 'microprice_ema_200ms'] > log_df.ix[self.cur_loc-1, 'atr_stop_top']:
	    print "    ", df.ix[self.cur_loc, 'time'], "turning", self.trend2, "to", "MarketTrend.bullish"
	    self.trend2 = MarketTrend.bullish
	elif self.trend2 == MarketTrend.neutral and (self.trend == MarketTrend.bearish or self.trend == MarketTrend.hyper_bearish) \
			and df.ix[self.cur_loc-1, 'MACD'] < df.ix[self.cur_loc-1, 'MACDsign'] \
			and df.ix[self.cur_loc-1, 'microprice_ema_200ms'] < log_df.ix[self.cur_loc-1, 'atr_stop_bot']:
	    print "    ", df.ix[self.cur_loc, 'time'], "turning", self.trend2, "to", "MarketTrend.bearish"
	    self.trend2 = MarketTrend.bearish
	elif self.trend2 == MarketTrend.bullish and df.ix[self.cur_loc-1, 'microprice_ema_200ms'] > log_df.ix[self.cur_loc-1, 'atr_stop_top']:
	    self.trend2 = MarketTrend.bullish
	elif self.trend2 == MarketTrend.bearish and df.ix[self.cur_loc-1, 'microprice_ema_200ms'] < log_df.ix[self.cur_loc-1, 'atr_stop_bot']:
	    self.trend2 = MarketTrend.bearish
	elif self.trend2 == MarketTrend.bullish and df.ix[self.cur_loc-1, 'microprice_ema_200ms'] < log_df.ix[self.cur_loc-1, 'atr_stop_bot']:
	    print "    ", df.ix[self.cur_loc, 'time'], "turning", self.trend2, "to", "MarketTrend.neutral"
	    self.trend2 = MarketTrend.neutral
	elif self.trend2 == MarketTrend.bearish and df.ix[self.cur_loc-1, 'microprice'] > log_df.ix[self.cur_loc-1, 'atr_stop_top']:
	    print "    ", df.ix[self.cur_loc, 'time'], "turning", self.trend2, "to", "MarketTrend.neutral"
	    self.trend2 = MarketTrend.neutral
	else:
	    pass

	df.ix[self.cur_loc, 'trend2'] = self.trend2

	#print df.ix[self.cur_loc, 'time'], self.trend2, df.ix[self.cur_loc, 'microprice'], log_df.ix[self.cur_loc, 'atr_stop_top'], log_df.ix[self.cur_loc, 'atr_stop_bot']


	#print "check: ", log_df.ix[self.cur_loc, 'atr_stop_top'], log_df.ix[self.cur_loc, 'atr_stop_bot'], log_df.ix[self.cur_loc-1, 'atr_stop_top'], log_df.ix[self.cur_loc-1, 'atr_stop_bot']
	if self.trend2 == MarketTrend.bullish:
	    log_df.ix[self.cur_loc, 'atr_stop_bot'] = max(log_df.ix[self.cur_loc-1, 'atr_stop_bot'], df.ix[self.cur_loc, 'atr_low_bound'])
            log_df.ix[self.cur_loc, 'atr_stop_top'] = df.ix[self.cur_loc, 'atr_high_bound']
	elif self.trend2 == MarketTrend.bearish:
	    log_df.ix[self.cur_loc, 'atr_stop_top'] = min(log_df.ix[self.cur_loc-1, 'atr_stop_top'], df.ix[self.cur_loc, 'atr_high_bound'])
            log_df.ix[self.cur_loc, 'atr_stop_bot'] = df.ix[self.cur_loc, 'atr_low_bound']
	elif self.trend2 == MarketTrend.neutral:
	    log_df.ix[self.cur_loc, 'atr_stop_bot'] = max(log_df.ix[self.cur_loc-1, 'atr_stop_bot'], df.ix[self.cur_loc, 'atr_low_bound'])
            log_df.ix[self.cur_loc, 'atr_stop_top'] = min(log_df.ix[self.cur_loc-1, 'atr_stop_top'], df.ix[self.cur_loc, 'atr_high_bound'])
	    #print "inside: ", log_df.ix[self.cur_loc, 'atr_stop_top'], log_df.ix[self.cur_loc, 'atr_stop_bot']
	else:
	    print "&#*((*@#&*"

	return df, log_df


    """def computeATRTrailingStop(self, df, log_df, trend, last_trend):
	if trend == MarketTrend.bullish or trend == MarketTrend.hyper_bullish:
	    log_df.ix[self.cur_loc, 'atr_stop'] = df.ix[self.cur_loc, 'atr_low_bound']
	elif trend == MarketTrend.bearish or trend == MarketTrend.hyper_bearish:
	    log_df.ix[self.cur_loc, 'atr_stop'] = df.ix[self.cur_loc, 'atr_high_bound']

	return df, log_df"""	


    """def computeATRTrailingStop(self, df):
	df['atr_stop'] = map(lambda trend, last_trend, atr_high, atr_low, last_atr_stop: \
				(atr_low if (last_trend != MarketTrend.bullish and last_trend != MarketTrend.hyper_bullish) else max(last_atr_stop,atr_low)) \
			if (trend == MarketTrend.bullish or trend == MarketTrend.hyper_bullish) \
			else \
				((atr_high if (last_trend != MarketTrend.bearish and last_trend != MarketTrend.hyper_bearish) else min(last_atr_stop,atr_high)) \
			if (trend == MarketTrend.bearish or trend == MarketTrend.hyper_bearish) else np.nan), \
		df['trend'], df['trend'].shift(1), df['atr_high_bound'], df['atr_low_bound'], df['atr_stop'].shift(1))

	return df"""


    """def computeATRTrailingStop(self, df):
        df['atr_stop'] = map(lambda trend, last_trend, atr_high, atr_low, last_atr_stop: \
				(atr_low  if (last_trend != MarketTrend.bullish and last_trend != MarketTrend.hyper_bullish) else max(last_atr_stop,atr_low)) \
			if (trend == MarketTrend.bullish or trend == MarketTrend.hyper_bullish) \
			else \
				((atr_high if (last_trend != MarketTrend.bearish and last_trend != MarketTrend.hyper_bearish) else min(last_atr_stop,atr_high)) \
			if (trend == MarketTrend.bearish or trend == MarketTrend.hyper_bearish) else np.nan), \
			df['trend'], df['trend'].shift(1), df['atr_high_bound'], df['atr_low_bound'], df['atr_stop'].shift(1))

	return df"""


    def computeSpike(self, df, start_dt, spike_duration_s=5):
        pre_event_snapshot_dt = start_dt - timedelta(seconds=2.5)
        pre_event_snapshot_loc = df.index.get_loc(pre_event_snapshot_dt)
        pre_event_microprice = df[df.time == pre_event_snapshot_dt]['microprice'][0]

	if spike_duration_s is not None:  # if no spike_duration, use cur_loc as end of spike
            spike_end_time = start_dt + timedelta(seconds=spike_duration_s)
            spike_end_loc = self.df.index.get_loc(spike_end_time)
	else:
	    spike_end_loc = self.cur_loc

        high_price = df.ix[pre_event_snapshot_loc : spike_end_loc, 'microprice'].max()
        low_price = df.ix[pre_event_snapshot_loc : spike_end_loc, 'microprice'].min()
 
        high_price_ticks = (high_price - pre_event_microprice) / float(min_price_increment)
        low_price_ticks = (low_price - pre_event_microprice) / float(min_price_increment)

        if abs(high_price_ticks) >= abs(low_price_ticks):
            spike_ticks = high_price_ticks
        else:
            spike_ticks = low_price_ticks

        if spike_ticks >= 4:
	    trade_direction = TradeDirection.bullish
	elif spike_ticks <= -4:
	    trade_direction = TradeDirection.bearish
	else:
	    trade_direction = TradeDirection.neutral

	self.spike_ticks = spike_ticks
    	self.tradeDirection = trade_direction

	return (spike_ticks, trade_direction)
    

    def priceTicks(self, price):
	pre_event_snapshot_dt = self.start_dt - timedelta(seconds=2.5)
	pre_event_microprice = self.df[self.df.time == pre_event_snapshot_dt]['microprice_ema_200ms'][0] 
	price_ticks = (price - pre_event_microprice) / float(min_price_increment)
        
        return price_ticks


    def compute_data(self, df):
        df = self.computeEMAs(df)
	df = self.computeMACD(df)
        df = self.computeTrend(df)
        df = self.computeATR(df, 100)
	df['trend2'] = np.nan
	df['trend3'] = np.nan
	self.log_df['atr_stop'] = np.nan
	self.log_df['atr_stop_bot'] = np.nan
	self.log_df['atr_stop_top'] = np.nan
	#df = self.computeATRTrailingStop(df)
	#self.computeSpike(df, self.start_dt, spike_duration_s=1)
	df['max_disl_ema_200ms_ticks'] = np.nan

        return df


    def on_fill(self, size, price):
	if size != 0:
	    self._pos += size
    	    self._cash -= (size * price * self.dollar_value_per_price_level)
    	    self._vol += abs(size)
    	    self._ltp = price


    def process_fills(self, fills):
        for fill in fills:
	    self.on_fill(fill['size'], fill['price'])


    def computePnL(self):
        if self._pos >= 0:
	    price = self.df.ix[self.cur_loc, 'top_bid_price']
	else:
	    price = self.df.ix[self.cur_loc, 'top_ask_price']

        return self._cash + float(self._pos) * float(price) * self.dollar_value_per_price_level
	    	
    def time_offset_loc(self, offset_secs):
        return self.df.index.get_loc(self.start_dt + timedelta(seconds=offset_secs))
	

    def computeTrend(self):
	if ((self.df.ix[self.cur_loc, 'microprice_ema_1200ms'] > self.df.ix[self.cur_loc, 'microprice_ema_3600ms']) \
			and (self.df.ix[self.cur_loc, 'microprice_ema_3600ms'] > self.df.ix[self.cur_loc, 'microprice_ema_6000ms']) \
			and (self.df.ix[self.cur_loc, 'microprice_ema_700ms'] > self.df.ix[self.cur_loc, 'microprice_ema_1200ms'])):
	    self.trend = MarketTrend.hyper_bullish
        elif ((self.df.ix[self.cur_loc, 'microprice_ema_1200ms'] > self.df.ix[self.cur_loc, 'microprice_ema_3600ms']) \
			and (self.df.ix[self.cur_loc, 'microprice_ema_3600ms'] > self.df.ix[self.cur_loc, 'microprice_ema_6000ms'])):
            self.trend = MarketTrend.bullish
	elif ((self.df.ix[self.cur_loc, 'microprice_ema_1200ms'] < self.df.ix[self.cur_loc, 'microprice_ema_3600ms']) \
			and (self.df.ix[self.cur_loc, 'microprice_ema_3600ms'] < self.df.ix[self.cur_loc, 'microprice_ema_6000ms']) \
                        and (self.df.ix[self.cur_loc, 'microprice_ema_700ms'] < self.df.ix[self.cur_loc, 'microprice_ema_1200ms'])):
            self.trend = MarketTrend.hyper_bearish
	elif ((self.df.ix[self.cur_loc, 'microprice_ema_1200ms'] < self.df.ix[self.cur_loc, 'microprice_ema_3600ms']) \
			and (self.df.ix[self.cur_loc, 'microprice_ema_3600ms'] < self.df.ix[self.cur_loc, 'microprice_ema_6000ms'])):
            self.trend = MarketTrend.bearish
        else:
	    self.trend = MarketTrend.neutral


    def computeTrend(self, df):
	"""df['trend'] = map(lambda ema0, ema1, ema2, ema3: \
				MarketTrend.hyper_bullish if (ema1 > ema2 and ema2 > ema3 and ema0 > ema1) \
			else (MarketTrend.bullish if (ema1 > ema2 and ema2 > ema3) \
			else (MarketTrend.hyper_bearish if (ema1 < ema2 and ema2 < ema3 and ema0 < ema1) \
			else (MarketTrend.bearish if (ema1 < ema2 and ema2 < ema3) \
			else MarketTrend.neutral)))), \
			df['microprice_ema_700ms'], df['microprice_ema_1200ms'], df['microprice_ema_3600ms'], df['microprice_ema_6000ms'])"""

	df['trend'] = map(lambda ema0, ema1, ema2, ema3: MarketTrend.hyper_bullish if (ema1 > ema2 and ema2 > ema3 and ema0 > ema1) \
			else (MarketTrend.bullish if (ema1 > ema2 and ema2 > ema3) \
			else (MarketTrend.hyper_bearish if (ema1 < ema2 and ema2 < ema3 and ema0 < ema1) \
			else (MarketTrend.bearish if (ema1 < ema2 and ema2 < ema3) \
			else MarketTrend.neutral))), \
			df['microprice_ema_700ms'], df['microprice_ema_1200ms'], df['microprice_ema_3600ms'], df['microprice_ema_6000ms'])


	return df


    def computeTrend3(self, df):
	if df.ix[self.cur_loc-1, 'trend3'] == MarketTrend.neutral and df.ix[self.cur_loc, 'microprice_ema_6000ms'] > df.ix[self.cur_loc, 'microprice_ema_26000ms']:
	    df.ix[self.cur_loc, 'trend3'] = MarketTrend.bullish
	elif df.ix[self.cur_loc-1, 'trend3'] == MarketTrend.bullish and df.ix[self.cur_loc, 'microprice_ema_6000ms'] > df.ix[self.cur_loc, 'microprice_ema_26000ms']:
	    df.ix[self.cur_loc, 'trend3'] = MarketTrend.bullish
	elif df.ix[self.cur_loc-1, 'trend3'] == MarketTrend.neutral and df.ix[self.cur_loc, 'microprice_ema_6000ms'] < df.ix[self.cur_loc, 'microprice_ema_26000ms']:
            df.ix[self.cur_loc, 'trend3'] = MarketTrend.bearish
	elif df.ix[self.cur_loc-1, 'trend3'] == MarketTrend.bearish and df.ix[self.cur_loc, 'microprice_ema_6000ms'] < df.ix[self.cur_loc, 'microprice_ema_26000ms']:
            df.ix[self.cur_loc, 'trend3'] = MarketTrend.bearish
	elif df.ix[self.cur_loc-1, 'trend3'] == MarketTrend.bullish and df.ix[self.cur_loc, 'microprice_ema_6000ms'] <= df.ix[self.cur_loc, 'microprice_ema_26000ms']:
            df.ix[self.cur_loc, 'trend3'] = MarketTrend.neutral
	elif df.ix[self.cur_loc-1, 'trend3'] == MarketTrend.bearish and df.ix[self.cur_loc, 'microprice_ema_6000ms'] >= df.ix[self.cur_loc, 'microprice_ema_26000ms']:
            df.ix[self.cur_loc, 'trend3'] = MarketTrend.neutral
	else: 
	    df.ix[self.cur_loc, 'trend3'] = MarketTrend.neutral

	return df


    def computeRelativeDislocation(self, s, p, norm=1.0):
        return (s - p) / norm


    def computeRunningMinMaxSignals(self, df, start_dt, spike_ticks, spike_5s_pred):
#	print start_dt
	pre_event_snapshot_dt = start_dt - timedelta(seconds=2.5)
        pre_event_snapshot_loc = df.index.get_loc(pre_event_snapshot_dt)

#	print df.ix[self.cur_loc, 'time'], len(df.ix[pre_event_snapshot_loc : self.cur_loc+1, 'microprice_ema_200ms'])

        max_to_here = pd.expanding_max(df.ix[pre_event_snapshot_loc : self.cur_loc+1, 'microprice_ema_200ms'])[-1]
        min_to_here = pd.expanding_min(df.ix[pre_event_snapshot_loc : self.cur_loc+1, 'microprice_ema_200ms'])[-1]

        max_to_here_ticks = self.priceTicks(max_to_here)
        min_to_here_ticks = self.priceTicks(min_to_here)

        if (spike_ticks + spike_5s_pred)/2.0 >= 0:
            max_disl = max_to_here_ticks
        else:
            max_disl = min_to_here_ticks

        #df.ix[self.cur_loc, 'from_max_disl_ema_200ms_ticks'] = computeRelativeDislocation(df.ix[self.cur_loc, 'microprice_ema_200ms_ticks'], max_disl)
	df.ix[self.cur_loc, 'max_disl_ema_200ms_ticks'] = max_disl
        #df[self.cur_loc]['max_disl_ema_200ms_ticks'] = max_disl

	return df


    def sendBuyEntry(self):
	new_order = {'type': 'entry', 'direction': 'buy', 'size': 50, 'levels_to_cross': 2, 'order_time_loc': self.cur_loc}
  	
	self.log_df.ix[self.cur_loc, 'buy_entry'] = 1.0
	#self._pos = 100
	print "buy entry at time: ", self.df.ix[self.cur_loc, 'time'], "size = ", new_order['size']

	return new_order


    def sendSellEntry(self):
	new_order = {'type': 'entry', 'direction': 'sell', 'size': 50, 'levels_to_cross': 2, 'order_time_loc': self.cur_loc}

	self.log_df.ix[self.cur_loc, 'sell_entry'] = -1.0
	#self._pos = -100
	print "sell entry at time: ", self.df.ix[self.cur_loc, 'time'], "size = ", new_order['size']

	return new_order


    def sendBuyExit(self):
	new_order = {'type': 'exit', 'direction': 'buy', 'size': abs(self._pos), 'levels_to_cross': 2, 'order_time_loc': self.cur_loc}

	self.log_df.ix[self.cur_loc, 'buy_exit'] = 2.0
	#self._pos = 0
	print "buy exit at time: ", self.df.ix[self.cur_loc, 'time'], "size = ", new_order['size']

	return new_order


    def sendSellExit(self):
	new_order = {'type': 'exit', 'direction': 'sell', 'size': abs(self._pos), 'levels_to_cross': 2, 'order_time_loc': self.cur_loc}	

	self.log_df.ix[self.cur_loc, 'sell_exit'] = -2.0
	#self._pos = 0
	print "sell exit at time: ", self.df.ix[self.cur_loc, 'time'], "size = ", new_order['size']

	return new_order


    def sendBuyExitInvert(self):
	self.sendBuyExit()  # this is incorrect: should really be one order
	self.sendBuyEntry()


    def sendSellExitInvert(self):
	self.sendSellExit()  # this is incorrect: should really be one order
	self.sendSellEntry()

    
    def conditionForBuyEntry(self, early_reaction=False):
	# THIS EXCLUDES EARLY REACTION
        if self._pos == 0 and self.cur_loc < self.max_entry_loc \
                        and self.trend == MarketTrend.hyper_bullish \
                        and self.df.ix[self.cur_loc-1, 'trend3'] == MarketTrend.neutral and self.df.ix[self.cur_loc, 'trend3'] == MarketTrend.bullish:
	    return True
	else:
	    return False

#        if self._pos == 0 \
#			and (True if self.priceTicks(self.df.ix[self.cur_loc, 'microprice']) < 0.95 * self.df.ix[self.cur_loc, 'max_disl_ema_200ms_ticks'] else False) \
#                                        if self.tradeDirection == TradeDirection.bullish \
#                                        else ((True if self.priceTicks(self.df.ix[self.cur_loc, 'microprice']) < 0.75 * self.df.ix[self.cur_loc, 'max_disl_ema_200ms_ticks'] else False) \
#                                                        if self.last_trend != MarketTrend.hyper_bearish else False) \
	# THIS WAS USED PREVIOUSLY
	#if self._pos == 0 and self.cur_loc < self.max_entry_loc \
	#			and self.trend == MarketTrend.hyper_bullish \
        #                and ((True if self.last_trend != MarketTrend.hyper_bullish else False) if not early_reaction else True) \
        #                and ((True if self.tradeDirection == TradeDirection.bullish else False) if early_reaction else True):
        #    return True
        #else:
        #    return False
	
#			#and self.priceTicks(self.df.ix[self.cur_loc, 'microprice']) < 0.95 * self.df.ix[self.cur_loc, 'max_disl_ema_200ms_ticks'] \
#			and (self.priceTicks(self.df.ix[self.cur_loc, 'microprice']) < 0.95 * self.df.ix[self.cur_loc, 'max_disl_ema_200ms_ticks']) \
#					if self.tradeDirection == TradeDirection.bullish \
#					else ((self.priceTicks(self.df.ix[self.cur_loc, 'microprice']) < 0.75 * self.df.ix[self.cur_loc, 'max_disl_ema_200ms_ticks']) \
#							if self.last_trend != MarketTrend.hyper_bearish False)
#			and self.trend == MarketTrend.hyper_bullish \
#			and ((True if self.last_trend != MarketTrend.hyper_bullish else False) if not early_reaction else True) \
#			and ((True if self.tradeDirection == TradeDirection.bullish else False) if early_reaction else True):
#	    return True
#	else:
#	    return False

#			and self.df.ix[self.cur_loc, 'MACDsmooth'] > 0 and self.df.ix[self.cur_loc, 'MACDsmooth'] > self.df.ix[self.cur_loc, 'MACDsign'] \


    def conditionForSellEntry(self, early_reaction=False):
	# THIS EXCLUDES EARLY REACTION
        if self._pos == 0 and self.cur_loc < self.max_entry_loc \
                        and self.trend == MarketTrend.hyper_bearish \
                        and self.df.ix[self.cur_loc-1, 'trend3'] == MarketTrend.neutral and self.df.ix[self.cur_loc, 'trend3'] == MarketTrend.bearish:
	    return True
	else:
	    return False

#	if self._pos == 0 \
#			and (True if self.priceTicks(self.df.ix[self.cur_loc, 'microprice']) > 0.95 * self.df.ix[self.cur_loc, 'max_disl_ema_200ms_ticks'] else False) \
#					if self.last_trend != MarketTrend.hyper_bearish \
#					else ((True if self.priceTicks(self.df.ix[self.cur_loc, 'microprice']) > 0.75 * self.df.ix[self.cur_loc, 'max_disl_ema_200ms_ticks'] else False) \
#							if self.tradeDirection == TradeDirection.bullish else False) \
	# THIS WAS USED PREVIOUSLY
	#if self._pos == 0 and self.cur_loc < self.max_entry_loc \
	#		and self.trend == MarketTrend.hyper_bearish \
	#		and ((True if self.last_trend != MarketTrend.hyper_bearish else False) if not early_reaction else True) \
	#		and ((True if self.tradeDirection == TradeDirection.bearish else False) if early_reaction else True):
	#    return True
	#else:
	#    return False

			#and self.priceTicks(self.df.ix[self.cur_loc, 'microprice']) > 0.95 * self.df.ix[self.cur_loc, 'max_disl_ema_200ms_ticks'] \
#			and self.df.ix[self.cur_loc, 'MACDsmooth'] < 0 and self.df.ix[self.cur_loc, 'MACDsmooth'] < self.df.ix[self.cur_loc, 'MACDsign'] \


    def conditionForBuyExit(self):
	#if self._pos < 0 and ((self.trend != MarketTrend.hyper_bearish and self.trend != MarketTrend.bearish) or (self.df.ix[self.cur_loc, 'microprice'] > self.log_df.ix[self.cur_loc, 'atr_stop'])):
	#if self._pos < 0 and self.df.ix[self.cur_loc, 'microprice'] > self.log_df.ix[self.cur_loc, 'atr_stop']:
	if self._pos < 0 and self.df.ix[self.cur_loc-1, 'trend3'] == MarketTrend.bearish and self.df.ix[self.cur_loc, 'trend3'] == MarketTrend.neutral:
            return True
        else:
            return False


    def conditionForSellExit(self):
        #if self._pos > 0 and ((self.trend != MarketTrend.hyper_bullish and self.trend != MarketTrend.bullish) or (self.df.ix[self.cur_loc, 'microprice'] < self.log_df.ix[self.cur_loc, 'atr_stop'])):
	#if self._pos > 0 and self.df.ix[self.cur_loc, 'microprice'] < self.log_df.ix[self.cur_loc, 'atr_stop']:
	if self._pos > 0 and self.df.ix[self.cur_loc-1, 'trend3'] == MarketTrend.bullish and self.df.ix[self.cur_loc, 'trend3'] == MarketTrend.neutral:
	    return True
	else:
	    return False


    def onTick(self):
        #self.df = self.computeTrend(self.df)
#        print self.cur_loc
#	print self.df.ix[self.cur_loc, 'trend']
        self.trend = self.df.ix[self.cur_loc, 'trend']
        self.last_trend = self.df.ix[self.cur_loc-1, 'trend']
	self.df, self.log_df = self.computeATRTrailingStop(self.df, self.log_df, self.trend, self.last_trend)
	if self.df.ix[self.cur_loc, 'time'] >= self.start_dt + timedelta(seconds=15):
	    self.df, self.log_df = self.computeNewATRTrailingStop(self.df, self.log_df, self.trend, self.last_trend)
	    self.df = self.computeTrend3(self.df)
	#self.atr_stop = self.df.ix[self.cur_loc, 'atr_stop']
#	print "onTick: ", self.df.ix[self.cur_loc, 'time']
        self.df = self.computeRunningMinMaxSignals(self.df, self.start_dt, self.spike_ticks, self.spike_5s_pred)

	#mkt_loc = self.cur_loc#self.stop_loc
	new_order = None
	fills = []
	if self.cur_loc < self.stop_loc:
            if ((self.cur_loc >= self.time_offset_loc(1)) and (self.cur_loc < self.time_offset_loc(5))):
	        self.computeSpike(self.df, self.start_dt, spike_duration_s=None)

	        if self.conditionForBuyEntry(early_reaction=True):
		    self.last_trend = self.trend
                    new_order = self.sendBuyEntry()
	        elif self.conditionForSellEntry(early_reaction=True):
		    self.last_trend = self.trend
                    new_order = self.sendSellEntry()
	    elif self.cur_loc >= self.time_offset_loc(5):
	        if self.conditionForBuyEntry(early_reaction=False):
                    self.last_trend = self.trend
                    new_order = self.sendBuyEntry()
                elif self.conditionForSellEntry(early_reaction=False):
                    self.last_trend = self.trend
                    new_order = self.sendSellEntry()
	
	    if self.conditionForSellExit():
	        new_order = self.sendSellExit()
	    elif self.conditionForBuyExit():
	        new_order = self.sendBuyExit()

	    if new_order:
		mkt_loc, fills_lst = self.mySim.execute([new_order])
		print fills_lst
                fill_report = fills_lst[0]
		print fill_report
        	fill_size = fill_report['filled_size']
        	fill_price = fill_report['avg_filled_price']
        	fills = [{'size': fill_size, 'price': fill_price}]  # just massaging the format a bit
	    else:
		mkt_loc, fills = self.mySim.execute([])  # no new order to send, so just get new market loc from Sim

	# ***** need to handle case of at stop time with leftover position!!!!!

        #if self.cur_loc < self.stop_loc:  # this timing logic should really be more specific in the cases above
        #    mkt_loc, fills = self.mySim.execute(self.cur_loc, [])
#	#print "bottom", self.df.ix[mkt_loc, 'time']

        return mkt_loc, fills


#    def onTick(self):
#        self.computeTrend()
#	self.computeRunningMinMaxSignals(self.df, self.spike_ticks, self.spike_5s_pred)
#
#	if ((self.cur_loc >= self.time_offset_loc(1)) and (self.cur_loc < self.time_offset_loc(5)) and (self._pos == 0)):
#	    self.computeSpike(df, self.start_dt, spike_duration_s=None)
#
#            if self.tradeDirection == TradeDirection.bullish and self.priceTicks(self.df.ix[self.cur_loc, 'microprice']) < self.spike_ticks and self.trend == MarketTrend.hyper_bullish:
#                self.last_trend = self.trend
#	        self.sendBuyEntry()
#	    elif self.tradeDirection == TradeDirection.bearish and self.priceTicks(self.df.ix[self.cur_loc, 'microprice']) > self.spike_ticks and self.trend == MarketTrend.hyper_bearish:
#		self.last_trend = self.trend
#		self.sendSellEntry()
#	elif self.cur_loc >= self.time_offset_loc(5) and self._pos == 0:
#	    if self.last_trend != MarketTrend.hyper_bullish and self.trend == MarketTrend.hyper_bullish:
#		self.sendBuyEntry()
#	    elif self.last_trend != MarketTrend.hyper_bearish and self.trend == MarketTrend.hyper_bearish:
#		self.sendSellEntry()
#	else:  # have a position
#	    if self._pos > 0:  # long position
#		if self.trend == MarketTrend.neutral or self.trend == MarketTrend.bearish:
#		    self.sendSellExit()
#		if self.trend == MarketTrend.hyper_bearish:
#		    self.sendSellExitInvert()
#	    elif self._pos < 0:  #short position
#		if self.trend == MarketTrend.neutral or self.trend == MarketTrend.bullish:
#                    self.sendBuyExit()
#                if self.trend == MarketTrend.hyper_bullish:
#                    self.sendBuyExitInvert()
#
#	if self.cur_loc < self.max_entry_loc:  # this timing logic should really be more specific in the cases above
#	    mkt_loc, fills = self.mySim.execute(self.cur_loc, [])
#
#	return mkt_loc, fills
		    


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

	start_dt = datetime.strptime(event_date + " " + trading_start_time, '%Y-%m-%d %H:%M:%S')
        max_entry_dt = datetime.strptime(event_date + " " + max_entry_time, '%Y-%m-%d %H:%M:%S')
        stop_dt = datetime.strptime(event_date + " " + trading_stop_time, '%Y-%m-%d %H:%M:%S')

	dollar_value_per_price_level = 12.5  # specific to Euro
	Sim = simulator.Simulator(df)
	Strat = Strategy(df, start_dt, max_entry_dt, stop_dt, Sim, dollar_value_per_price_level, spike_5s_pred[event_date])
	Sim.initStrategy(Strat)
	log_df = Strat.start()

	dir_path = '/local/disk1/temp_eg/log_run/'
        filename = 'log_run_' + event_name + '_' + instrument_root + '_' + event_date.replace('-', '')
        store_filename = dir_path + filename + '.h5'
        store = pd.HDFStore(store_filename)
        store['log_df'] = log_df
        store.close()


	print "FINISHED"

if __name__ == "__main__": main()


