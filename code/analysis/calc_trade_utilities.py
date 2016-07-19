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


def printBook(df):
    pass

def printBookCurrentNext(df, index):
    print "------------------------------------------------------------------"
    print str(df.ix[index, 'time']) + "\t" + str(df.ix[index+1, 'time'])
    print str(df.ix[index, 'ask_size_4']) + "\t" + str(df.ix[index, 'top_ask_price']+4*min_price_increment) + "\t\t\t" + str(df.ix[index+1, 'ask_size_4']) + "\t" + str(df.ix[index+1, 'top_ask_price']+4*min_price_increment)
    print str(df.ix[index, 'ask_size_3']) + "\t" + str(df.ix[index, 'top_ask_price']+3*min_price_increment) + "\t\t\t" + str(df.ix[index+1, 'ask_size_3']) + "\t" + str(df.ix[index+1, 'top_ask_price']+3*min_price_increment)
    print str(df.ix[index, 'ask_size_2']) + "\t" + str(df.ix[index, 'top_ask_price']+4*min_price_increment) + "\t\t\t" + str(df.ix[index+1, 'ask_size_2']) + "\t" + str(df.ix[index+1, 'top_ask_price']+2*min_price_increment)
    print str(df.ix[index, 'ask_size_1']) + "\t" + str(df.ix[index, 'top_ask_price']+1*min_price_increment) + "\t\t\t" + str(df.ix[index+1, 'ask_size_1']) + "\t" + str(df.ix[index+1, 'top_ask_price']+1*min_price_increment)
    print str(df.ix[index, 'ask_size_0']) + "\t" + str(df.ix[index, 'top_ask_price']+0*min_price_increment) + "\t\t\t" + str(df.ix[index+1, 'ask_size_0']) + "\t" + str(df.ix[index+1, 'top_ask_price']+0*min_price_increment)
    print "====================" + "\t\t" + "===================="
    print str(df.ix[index, 'bid_size_0']) + "\t" + str(df.ix[index, 'top_bid_price']+0*min_price_increment) + "\t\t\t" + str(df.ix[index+1, 'bid_size_0']) + "\t" + str(df.ix[index+1, 'top_bid_price']+0*min_price_increment)
    print str(df.ix[index, 'bid_size_1']) + "\t" + str(df.ix[index, 'top_bid_price']+1*min_price_increment) + "\t\t\t" + str(df.ix[index+1, 'bid_size_1']) + "\t" + str(df.ix[index+1, 'top_bid_price']+1*min_price_increment)
    print str(df.ix[index, 'bid_size_2']) + "\t" + str(df.ix[index, 'top_bid_price']+4*min_price_increment) + "\t\t\t" + str(df.ix[index+1, 'bid_size_2']) + "\t" + str(df.ix[index+1, 'top_bid_price']+2*min_price_increment)
    print str(df.ix[index, 'bid_size_3']) + "\t" + str(df.ix[index, 'top_bid_price']+3*min_price_increment) + "\t\t\t" + str(df.ix[index+1, 'bid_size_3']) + "\t" + str(df.ix[index+1, 'top_bid_price']+3*min_price_increment)
    print str(df.ix[index, 'bid_size_4']) + "\t" + str(df.ix[index, 'top_bid_price']+4*min_price_increment) + "\t\t\t" + str(df.ix[index+1, 'bid_size_4']) + "\t" + str(df.ix[index+1, 'top_bid_price']+4*min_price_increment)
    print; print

def printBestTradeUtility(direction, entry_size, entry_price, entry_time, exit_size, exit_price, exit_time, utility):
    print "======================================================================================================"
    print direction.upper() + " entry:", str(entry_time) + "\t\t" + "best exit:", str(exit_time) + "\t" + "best utility:", utility
    print "size:" + "\t" + str(entry_size) + "\t\t\t\t\t" + str(exit_size)
    print "price:" + "\t" + str(round(entry_price,1)) + "\t\t\t\t\t" + str(round(exit_price,1))
    print "======================================================================================================"
    print; print; print





def computeAggressiveBuyFill(df, loc, target_size, levels_to_cross, ask_size_npar, bid_size_npar, top_prices_npar):
    t0a = timeit.default_timer()
    #top_ask_price_now = df.ix[loc, 'top_ask_price']
    top_ask_price_now = top_prices_npar[loc,0]
    t0a1 = timeit.default_timer()
    buy_price_now = [top_ask_price_now + min_price_increment * m for m in range(0,book_depth)]
    t0a2 = timeit.default_timer()
    #buy_size_now = [df.ix[loc, 'ask_size_'+str(m)] for m in range(0,book_depth)]
    #buy_size_now = []
    #buy_size_now.append(df.ix[loc, 'ask_size_0'])
    #buy_size_now.append(df.ix[loc, 'ask_size_1'])
    #buy_size_now.append(df.ix[loc, 'ask_size_2'])
    #buy_size_now.append(df.ix[loc, 'ask_size_3'])
    #buy_size_now.append(df.ix[loc, 'ask_size_4'])
    buy_size_now = ask_size_npar[loc].tolist()
    #print buy_size_now
    t0a3 = timeit.default_timer()    

    buy_size_next = [0]*book_depth
    t0a4 = timeit.default_timer()
    
    #top_ask_price_next = df.ix[loc+1, 'top_ask_price']
    top_ask_price_next = top_prices_npar[loc+1,0]
    t0a5 = timeit.default_timer()
    market_move = int((top_ask_price_next - top_ask_price_now) / float(min_price_increment))
    t0b = timeit.default_timer()
    if market_move == 0:
        #buy_size_next = [df.ix[loc+1, 'ask_size_'+str(m)] for m in range(0,book_depth)]
        buy_size_next = ask_size_npar[loc+1].tolist()
    elif (market_move > 0) and (market_move < book_depth):
        buy_size_next[market_move:book_depth] = buy_size_now[0:book_depth-market_move]
    elif market_move < 0:
        buy_size_next[:] = buy_size_now[:]
    t0c = timeit.default_timer()
        
    buy_size = [min(buy_size_now[k], buy_size_next[k]) for k in range(0,book_depth)]
    
    t0d = timeit.default_timer()
    execute_size = target_size
    level = 0
    size_weighted_price = 0
    filled_size = 0
    while execute_size > 0 and level < min(levels_to_cross, book_depth):
        #print buy_size, level, levels_to_cross, book_depth
	#t1 = timeit.default_time()
        if execute_size <= buy_size[level]:
            partial_filled_size = execute_size
        elif execute_size > buy_size[level]:
            partial_filled_size = buy_size[level]
            
        filled_size += partial_filled_size
        size_weighted_price += buy_price_now[level] * partial_filled_size
        execute_size -= partial_filled_size
        level += 1
	#t2 = timeit.default_timer()
	#print t2-t1
        
    t0e = timeit.default_timer()
    avg_filled_price = size_weighted_price / float(filled_size)
    
    #print "    t0a1-t0a=" + "\t" + str(t0a1-t0a)
    #print "    t0a2-t0a1=" + "\t" + str(t0a2-t0a1)
    #print "    t0a3-t0a2=" + "\t" + str(t0a3-t0a2)
    #print "    t0a4-t0a3=" + "\t" + str(t0a4-t0a3)
    #print "    t0a5-t0a4=" + "\t" + str(t0a5-t0a4)
    #print "  t0b-t0a=" + "\t" + str(t0b-t0a)
    #print "  t0c-t0b=" + "\t" + str(t0c-t0b)
    #print "  t0d-t0c=" + "\t" + str(t0d-t0c)
    #print "  t0e-t0d=" + "\t" + str(t0e-t0d)

    return (avg_filled_price, filled_size)


def computeAggressiveSellFill(df, loc, target_size, levels_to_cross, ask_size_npar, bid_size_npar, top_prices_npar):
    #top_bid_price_now = df.ix[loc, 'top_bid_price']
    top_bid_price_now = top_prices_npar[loc, 1]
    sell_price_now = [top_bid_price_now - min_price_increment * m for m in range(0,book_depth)]
    #sell_size_now = [df.ix[loc, 'bid_size_'+str(m)] for m in range(0,book_depth)]
    sell_size_now = bid_size_npar[loc].tolist()    

    sell_size_next = [0]*book_depth
    
    #top_bid_price_next = df.ix[loc+1, 'top_bid_price']
    top_bid_price_next = top_prices_npar[loc+1,1]
    market_move = int((top_bid_price_next - top_bid_price_now) / float(min_price_increment))
    if market_move == 0:
        #sell_size_next = [df.ix[loc+1, 'bid_size_'+str(m)] for m in range(0,book_depth)]
	sell_size_next = bid_size_npar[loc+1].tolist()
    elif (market_move < 0) and (abs(market_move) < book_depth):
        sell_size_next[abs(market_move):book_depth] = sell_size_now[0:book_depth-abs(market_move)]
    elif market_move > 0:
        sell_size_next[:] = sell_size_now[:]
        
    sell_size = [min(sell_size_now[k], sell_size_next[k]) for k in range(0,book_depth)]
    
    execute_size = target_size
    level = 0
    size_weighted_price = 0
    filled_size = 0
    while execute_size > 0 and level < min(levels_to_cross, book_depth):
        if execute_size <= sell_size[level]:
            partial_filled_size = execute_size
        elif execute_size > sell_size[level]:
            partial_filled_size = sell_size[level]
            
        filled_size += partial_filled_size
        size_weighted_price += sell_price_now[level] * partial_filled_size
        execute_size -= partial_filled_size
        level += 1
        
    avg_filled_price = size_weighted_price / float(filled_size)
    
    return (avg_filled_price, filled_size)


def computeRadicalExitFill(df, loc, target_size, direction, top_prices_npar):
    if direction == "buy":
        #top_ask_price_now = df.ix[loc, 'top_ask_price']
	top_ask_price_now = top_prices_npar[loc, 0]
        avg_filled_price = top_ask_price_now + (book_depth + 2) * min_price_increment
    elif direction == "sell":
        #top_bid_price_now = df.ix[loc, 'top_bid_price']
	top_bid_price_now = top_prices_npar[loc, 1]
        avg_filled_price = top_bid_price_now - (book_depth + 2) * min_price_increment
        
    return (avg_filled_price, target_size)


def computeBuyMarkout(df, trades_df, start_dt, stop_dt, max_entry_dt, start_loc, max_entry_loc, stop_loc, ask_size_npar, bid_size_npar, top_prices_npar, time_at_loc_npar):
    for i in range(start_loc, max_entry_loc):
    #for i in range(start_loc, start_loc+4):
	t1 = timeit.default_timer() 
        avg_buy_price, buy_size = computeAggressiveBuyFill(df, i, nominal_trading_size, 2, ask_size_npar, bid_size_npar, top_prices_npar)
	t2 = timeit.default_timer()        

        max_loss = 0
        max_loss_ticks = 0
        is_stopped_out = False
       
	t2A = timeit.default_timer() 
        #entry_time = df.ix[i, 'time']
	entry_time = pd.Timestamp(time_at_loc_npar[i])    
	t2B = timeit.default_timer()
        
        sell_size = 0
        best_avg_exit_price = float('NaN') 
        best_nominal_trade_quality = 0
	best_sized_trade_quality = 0
	best_time_decayed_trade_quality = 0
	best_duration_penalized_trade_quality = 0
        #best_exit_time = df.ix[i + 1, 'time']
	best_exit_time = pd.Timestamp(time_at_loc_npar[i+1])
	t2C = timeit.default_timer()
        
        max_holding_offset = max_hold_period_s * 10  # because looking at 100ms intervals
        max_exit_loc = min(i + max_holding_offset, stop_loc)
        j = i + 1
        while j < max_exit_loc and not is_stopped_out and buy_size > 0:  # compute all exits for this buy entry
	    t3 = timeit.default_timer()
	    #exit_time = df.ix[j, 'time']
            exit_time = pd.Timestamp(time_at_loc_npar[j])
            #printBookCurrentNext(df, j)
 	    t3A = timeit.default_timer()
	    top_bid_price = top_prices_npar[j,1]
	    #if df.ix[j, 'top_bid_price'] <= avg_buy_price - stop_loss_ticks * min_price_increment:
	    if top_bid_price <= avg_buy_price - stop_loss_ticks * min_price_increment:
		t3B = timeit.default_timer()
                is_stopped_out = True
		t3C = timeit.default_timer()
                print "STOPPED OUT:", exit_time
		t3D = timeit.default_timer()
        
            levels_to_cross = 2
            sell_size = 0
	    t4 = timeit.default_timer()
            while sell_size < buy_size and levels_to_cross <= book_depth:
                avg_sell_price, sell_size = computeAggressiveSellFill(df, j, buy_size, levels_to_cross, ask_size_npar, bid_size_npar, top_prices_npar)
                levels_to_cross += 1
	    t5 = timeit.default_timer()
            remaining_size = buy_size - sell_size  # open position
            if remaining_size > 0:
                print "PROBLEM: could not exit entire position.  Resorting to radical fill."
                avg_sell_price, sell_size = computeRadicalExitFill(df, j, remaining_size, "sell", top_prices_npar)
            
            PnL = avg_sell_price - avg_buy_price
            if PnL < max_loss:
                max_loss = PnL
                max_loss_ticks = max_loss / float(min_price_increment)
                
            if PnL > 0:
		nominal_trade_quality = nominal_trading_size * (PnL - 0.5 * max_loss) / float(min_price_increment)
                sized_trade_quality = buy_size * (PnL - 0.5 * max_loss) / float(min_price_increment)
            else:
		nominal_trade_quality = nominal_trading_size * PnL / float(min_price_increment)
                sized_trade_quality = buy_size * PnL / float(min_price_increment)
	    time_decayed_trade_quality = sized_trade_quality * np.power(0.99996, j)
	    
	    trade_duration_s = (exit_time - entry_time).total_seconds()
	    duration_penal_mult = 1 - trade_duration_s/(4 * max_hold_period_s)
	    duration_penalized_trade_quality = sized_trade_quality * duration_penal_mult
                
            #exit_time = df.ix[j, 'time']
	    #trade_duration_s = (exit_time - entry_time).total_seconds()
	    #print "************************trade duration", trade_duration_s
                
            #if sized_trade_quality > best_sized_trade_quality:
	    if duration_penalized_trade_quality > best_duration_penalized_trade_quality:
                best_nominal_trade_quality = nominal_trade_quality
		best_sized_trade_quality = sized_trade_quality
		best_time_decayed_trade_quality = time_decayed_trade_quality
	        best_duration_penalized_trade_quality = duration_penalized_trade_quality
                best_exit_time = exit_time
                best_avg_exit_price = avg_sell_price
                
            j += 1

	    t6 = timeit.default_timer()
            
            #print "buy entry:", str(entry_time) + "\t\t" +  "exit:", str(exit_time) + "\t" +  "utility:", str(trade_quality)
	    #print "size:" + "\t" + str(buy_size) + "\t\t\t\t\t" + str(sell_size)
            #print "price:" + "\t" + str(round(avg_buy_price,1)) + "\t\t\t\t\t" + str(round(avg_sell_price,1))
 	    #print; print

	    t7 = timeit.default_timer()

	t8 = timeit.default_timer()

	#printBestTradeUtility("buy", buy_size, avg_buy_price, entry_time, sell_size, best_avg_exit_price, best_exit_time, best_trade_quality)
	t9 = timeit.default_timer()

	print "Completed buy trade (time = " + str(entry_time) + ")"
	t10 = timeit.default_timer()
	
#	print "num exit iterations=", j
#	print "t2-t1=" + "\t" + str(t2-t1)
#	print "t2A-t2" + "\t" + str(t2A-t2)
#	print "t2B-t2A" + "\t" + str(t2B-t2A)
#	print "t2C-t2B" + "\t" + str(t2C-t2B)
#	print "t4-t3A=" + "\t" + str(t4-t3A)
#	print "t3A-t3=" + "\t" + str(t3A-t3)
#	print "t3B-t3A=" + "\t" + str(t3B-t3A)
#	print "t3C-t3B=" + "\t" + str(t3C-t3B)
#	print "t3D-t3C=" + "\t" + str(t3D-t3C)
#	print "t5-t4=" + "\t" + str(t5-t4)
#	print "t6-t5=" + "\t" + str(t6-t5)
#	print "t7-t6=" + "\t" + str(t7-t6)
#	print "t8-t2=" + "\t" + str(t8-t2)
#	print "t9-t8=" + "\t" + str(t9-t8)
#	print "t10-t9=" + "\t" + str(t10-t9)

	trades_df.ix[i, 'buy_nominal_trade_utility'] = best_nominal_trade_quality
	trades_df.ix[i, 'buy_sized_trade_utility'] = best_sized_trade_quality
	trades_df.ix[i, 'buy_time_decayed_trade_utility'] = best_time_decayed_trade_quality
	trades_df.ix[i, 'buy_duration_penalized_trade_utility'] = best_duration_penalized_trade_quality
	trades_df.ix[i, 'buy_trade_exit_time'] = best_exit_time
	trades_df.ix[i, 'buy_trade_exit_avg_price'] = best_avg_exit_price 
	trades_df.ix[i, 'buy_fill_size'] = buy_size
	#print trades_df.ix[i]
	#print trades_df.loc[trades_df['time'] == entry_time]    
    
    return trades_df


def computeSellMarkout(df, trades_df, start_dt, stop_dt, max_entry_dt, start_loc, max_entry_loc, stop_loc, ask_size_npar, bid_size_npar, top_prices_npar, time_at_loc_npar):
    for i in range(start_loc, max_entry_loc):
        t1 = timeit.default_timer()
        avg_sell_price, sell_size = computeAggressiveSellFill(df, i, nominal_trading_size, 2, ask_size_npar, bid_size_npar, top_prices_npar)
	print avg_sell_price, sell_size
        t2 = timeit.default_timer()

        max_loss = 0
        max_loss_ticks = 0
        is_stopped_out = False

        t2A = timeit.default_timer()
        #entry_time = df.ix[i, 'time']
        entry_time = pd.Timestamp(time_at_loc_npar[i])
        t2B = timeit.default_timer()

	buy_size = 0
        best_avg_exit_price = float('NaN')
        best_nominal_trade_quality = 0
        best_sized_trade_quality = 0
        best_time_decayed_trade_quality = 0
	best_duration_penalized_trade_quality = 0
        #best_exit_time = df.ix[i + 1, 'time']
        best_exit_time = pd.Timestamp(time_at_loc_npar[i+1])
        t2C = timeit.default_timer()

        max_holding_offset = max_hold_period_s * 10  # because looking at 100ms intervals
        max_exit_loc = min(i + max_holding_offset, stop_loc)
        j = i + 1
	#print j, max_exit_loc
	#print is_stopped_out
	#print sell_size
        while j < max_exit_loc and not is_stopped_out and sell_size > 0:  # compute all exits for this sell entry
            t3 = timeit.default_timer()
            #exit_time = df.ix[j, 'time']
            exit_time = pd.Timestamp(time_at_loc_npar[j])
            #printBookCurrentNext(df, j)
            t3A = timeit.default_timer()
            top_ask_price = top_prices_npar[j,0]
            #if df.ix[j, 'top_bid_price'] <= avg_buy_price - stop_loss_ticks * min_price_increment:
            if top_ask_price >= avg_sell_price + stop_loss_ticks * min_price_increment:
                t3B = timeit.default_timer()
                is_stopped_out = True
                t3C = timeit.default_timer()
                print "STOPPED OUT:", exit_time
                t3D = timeit.default_timer()

            levels_to_cross = 2
            buy_size = 0
            t4 = timeit.default_timer()
            while buy_size < sell_size and levels_to_cross <= book_depth:
                avg_buy_price, buy_size = computeAggressiveBuyFill(df, j, sell_size, levels_to_cross, ask_size_npar, bid_size_npar, top_prices_npar)
                levels_to_cross += 1
            t5 = timeit.default_timer()
            remaining_size = sell_size - buy_size  # open position
            if remaining_size > 0:
                print "PROBLEM: could not exit entire position.  Resorting to radical fill."
                avg_buy_price, buy_size = computeRadicalExitFill(df, j, remaining_size, "buy", top_prices_npar)

            PnL = avg_sell_price - avg_buy_price
            if PnL < max_loss:
                max_loss = PnL
                max_loss_ticks = max_loss / float(min_price_increment)

            if PnL > 0:
                nominal_trade_quality = nominal_trading_size * (PnL - 0.5 * max_loss) / float(min_price_increment)
                sized_trade_quality = buy_size * (PnL - 0.5 * max_loss) / float(min_price_increment)
            else:
                nominal_trade_quality = nominal_trading_size * PnL / float(min_price_increment)
                sized_trade_quality = buy_size * PnL / float(min_price_increment)
            time_decayed_trade_quality = sized_trade_quality * np.power(0.99996, j)

	    trade_duration_s = (exit_time - entry_time).total_seconds()
            duration_penal_mult = 1 - trade_duration_s/(4 * max_hold_period_s)
            duration_penalized_trade_quality = sized_trade_quality * duration_penal_mult

            #if sized_trade_quality > best_sized_trade_quality:
	    if duration_penalized_trade_quality > best_duration_penalized_trade_quality:
                best_nominal_trade_quality = nominal_trade_quality
                best_sized_trade_quality = sized_trade_quality
                best_time_decayed_trade_quality = time_decayed_trade_quality
		best_duration_penalized_trade_quality = duration_penalized_trade_quality
                best_exit_time = exit_time
                best_avg_exit_price = avg_sell_price

            j += 1
            t6 = timeit.default_timer()

            #print "buy entry:", str(entry_time) + "\t\t" +  "exit:", str(exit_time) + "\t" +  "utility:", str(trade_quality)
            #print "size:" + "\t" + str(buy_size) + "\t\t\t\t\t" + str(sell_size)
            #print "price:" + "\t" + str(round(avg_buy_price,1)) + "\t\t\t\t\t" + str(round(avg_sell_price,1))
            #print; print

            t7 = timeit.default_timer()

        t8 = timeit.default_timer()

        #printBestTradeUtility("buy", buy_size, avg_buy_price, entry_time, sell_size, best_avg_exit_price, best_exit_time, best_trade_quality)
        t9 = timeit.default_timer()

        print "Completed sell trade (time = " + str(entry_time) + ")"
        t10 = timeit.default_timer()

#        print "num exit iterations=", j
#        print "t2-t1=" + "\t" + str(t2-t1)
#        print "t2A-t2" + "\t" + str(t2A-t2)
#        print "t2B-t2A" + "\t" + str(t2B-t2A)
#        print "t2C-t2B" + "\t" + str(t2C-t2B)
#        print "t4-t3A=" + "\t" + str(t4-t3A)
#        print "t3A-t3=" + "\t" + str(t3A-t3)
#        print "t3B-t3A=" + "\t" + str(t3B-t3A)
#        print "t3C-t3B=" + "\t" + str(t3C-t3B)
#        print "t3D-t3C=" + "\t" + str(t3D-t3C)
#        print "t5-t4=" + "\t" + str(t5-t4)
#        print "t6-t5=" + "\t" + str(t6-t5)
#        print "t7-t6=" + "\t" + str(t7-t6)
#        print "t8-t2=" + "\t" + str(t8-t2)
#        print "t9-t8=" + "\t" + str(t9-t8)
#        print "t10-t9=" + "\t" + str(t10-t9)

        trades_df.ix[i, 'sell_nominal_trade_utility'] = best_nominal_trade_quality
        trades_df.ix[i, 'sell_sized_trade_utility'] = best_sized_trade_quality
        trades_df.ix[i, 'sell_time_decayed_trade_utility'] = best_time_decayed_trade_quality
	trades_df.ix[i, 'sell_duration_penalized_trade_utility'] = best_duration_penalized_trade_quality
        trades_df.ix[i, 'sell_trade_exit_time'] = best_exit_time
        trades_df.ix[i, 'sell_trade_exit_avg_price'] = best_avg_exit_price
        trades_df.ix[i, 'sell_fill_size'] = buy_size

    return trades_df
    


def computeMarkouts(df, trades_df, start_dt, max_entry_dt, stop_dt, ask_size_npar, bid_size_npar, top_prices_npar, time_at_loc_npar):
    
    #print start_dt
    df.ix[(df.time<start_dt)&(df.time>max_entry_dt), 'buy_trade_markout'] = float('NaN')
    df.ix[(df.time<start_dt)&(df.time>max_entry_dt), 'sell_trade_markout'] = float('NaN')
    
    start_loc = df.index.get_loc(start_dt)
    max_entry_loc = df.index.get_loc(max_entry_dt)
    stop_loc = df.index.get_loc(stop_dt)
    #print start_loc, max_entry_loc, stop_loc
    
    trades_df = computeBuyMarkout(df, trades_df, start_dt, max_entry_dt, stop_dt, start_loc, max_entry_loc, stop_loc, ask_size_npar, bid_size_npar, top_prices_npar, time_at_loc_npar)
    trades_df = computeSellMarkout(df, trades_df, start_dt, max_entry_dt, stop_dt, start_loc, max_entry_loc, stop_loc, ask_size_npar, bid_size_npar, top_prices_npar, time_at_loc_npar)
    return trades_df


def main():
    import timeit
    for event_date in event_dates:
    	event_datetime_obj = datetime.strptime(event_date + " " + event_start_time, '%Y-%m-%d %H:%M:%S')
    	symbol = mkdt_utils.getSymbol(instrument_root, event_datetime_obj)
    	df = mkdt_utils.getMarketDataFrameForTradingDate(event_date, instrument_root, symbol, "100ms")
	if isinstance(df.head(1).time[0], basestring):
            df.time = df.time.apply(mkdt_utils.str_to_dt)
    	df.set_index(df['time'], inplace=True)

	t00 = timeit.default_timer()
	ask_size_npar = df[['ask_size_0', 'ask_size_1', 'ask_size_2', 'ask_size_3', 'ask_size_4']].values
	bid_size_npar = df[['bid_size_0', 'bid_size_1', 'bid_size_2', 'bid_size_3', 'bid_size_4']].values
	top_prices_npar = df[['top_ask_price', 'top_bid_price']].values
	time_at_loc_npar = df['time'].values
	print time_at_loc_npar[0], time_at_loc_npar[1]
	t01 = timeit.default_timer()
        print "t01-t00=" + "\t" + str(t01-t00)

	trades_df = pd.DataFrame(index = df['time'])
	trades_df['time'] = df['time']

    	start_dt = datetime.strptime(event_date + " " + trading_start_time, '%Y-%m-%d %H:%M:%S')
    	max_entry_dt = datetime.strptime(event_date + " " + max_entry_time, '%Y-%m-%d %H:%M:%S')
    	stop_dt = datetime.strptime(event_date + " " + trading_stop_time, '%Y-%m-%d %H:%M:%S')
    	
	trades_df = computeMarkouts(df, trades_df, start_dt, max_entry_dt, stop_dt, ask_size_npar, bid_size_npar, top_prices_npar, time_at_loc_npar)
	#print trades_df.loc[trades_df['time'] == df.ix[9013, 'time']]

	dir_path = '/local/disk1/temp_eg/trade_util_labels/'
	trade_name = event_name + '_' + instrument_root + '_' + event_date.replace('-', '')
	store_filename = dir_path + trade_name + '.h5'
	store = pd.HDFStore(store_filename)
	store['trades_df'] = trades_df 
	store.close()
    

if __name__ == "__main__": main()











