import sys
import os
from itertools import izip
sys.path.append('/home/egrois/git/code/pybt')
sys.path.append('/home/egrois/git/code/preprocess')
sys.path.append('/home/egrois/git/code/analysis')
sys.path.append('/home/egrois/git/code/backtest')
import pybt

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime,time, timedelta
import pandas as pd
import utils
reload(utils)
import matplotlib.finance

import csv
import math
import copy

import mkdt_utils
import tech_anal

#import simulator4
#import simulator5
import simulator6


event_time_offset_s = 0  # for example, for Payrolls can be =1.0

symbols_bkp = {'2014-09-24': 'CLX4', '2015-12-30': 'CLG6', '2016-01-06': 'CLG6', '2016-01-13': 'CLG6', '2016-01-21': 'CLH6',
               '2016-01-27': 'CLH6', '2016-02-03': 'CLH6', '2016-02-10': 'CLH6', '2016-02-18': 'CLH6', '2016-02-24': 'CLJ6',
               '2016-03-02': 'CLJ6', '2016-03-09': 'CLJ6', '2016-03-16': 'CLJ6', '2016-03-23': 'CLK6', '2016-03-30': 'CLK6',
               '2016-04-06': 'CLK6', '2016-04-13': 'CLK6'}


"""Parameter settings for valuing an entry/trade.  These are specific to event and instrument (for now)."""
min_price_increment = 1  # need to have a lookup structure for these
dollar_value_per_price_level = 10.0  #6.25
max_size_on_level = 10
max_allowed_pos = 100
max_levels_to_cross = 5
book_depth = 5

pre_trading_stop_secs = 10
  

small_abs_pos = 10

bid_form = "quadratic"
bid_param1 = 0.005
bid_param2 = 2.0
bid_depth = 100 #100
bid_skip = 2
bid_edge = 0 #10 #5

ask_form = "quadratic"
ask_param1 = 0.005
ask_param2 = 2.0
ask_depth = 100 #100
ask_skip = 2
ask_edge = 0 #10 #5



class Stacker:
    def __init__(self):
        self.ask_stacker_orders = []
        self.bid_stacker_orders = []


class TimeDecayTrigger:
    def __init__(self, trigger_period_seconds=1.0):
	self.trigger_period_seconds = trigger_period_seconds
	self.isStartedStatus = False
	self.start_time_dt = None
	#self.start_loc = start_loc
	self.trigger_period_seconds = trigger_period_seconds
	#isTriggered = True
	self.last_triggered_time_dt = None

    def isStarted(self):
	return self.isStartedStatus

    def start(self, cur_time_dt):
	if not self.isStartedStatus:
	    self.start_time_dt = cur_time_dt
	    self.isStartedStatus = True
	    self.last_triggered_time_dt = cur_time_dt
	    return 1
	else:
	    return 0

    def stop(self):
	if self.isStartedStatus:
	    self.isStartedStatus = False
	    return 1
	else:
	    return 0

    def isTriggered(self, cur_time_dt):
	if self.isStarted:
	    if cur_time_dt == self.last_triggered_time_dt:
	        return True
	    elif cur_time_dt > self.last_triggered_time_dt and cur_time_dt < self.last_triggered_time_dt + timedelta(seconds=self.trigger_period_seconds):
	        return False
	    elif cur_time_dt == self.last_triggered_time_dt + timedelta(seconds=self.trigger_period_seconds):
	        self.last_triggered_time_dt = cur_time_dt
	        return True
	    else:
	        return False
	else:
	    return False

    def periodsElapsed(self, cur_time_dt):
	if self.isTriggered:
	    return (cur_time_dt - self.start_time_dt).total_seconds()
	else:
	    return None



class Strategy:

    _pos = 0
    _cash = 0
    _vol = 0
    _ltp = 0
    _pnl = 0

    potential_pnl = 0
    best_potential_pnl = 0
    pnl_per_contract = 0
    worst_potential_pnl = 0


    def __init__(self, df, simlog_df, event_dt, start_dt, stop_dt, opposite_stacker_off_dt, entry_stop_time_dt, time_decay_exit_start_dt, reversal_tracking_start_dt, stop_loss_tracking_start_dt, delayed_trade_start_dt, trade_termination_dt, Sim, dollar_value_per_price_level, cpu_process_id=-1):
        self.df = df
	self.simlog_df = simlog_df
        self.mySim = Sim
        self.dollar_value_per_price_level = dollar_value_per_price_level

	self.run_trade_only = False
	self.trade_running = False

	self.initial_entry_sent = False

        self.myStacker = Stacker()

	self.delayed_trade = True
	self.trading_reaction_delay = 0
	self.delayed_trade_start_dt = delayed_trade_start_dt
	self.delayed_trade_start_loc = df.index.get_loc(self.delayed_trade_start_dt)
	self.delayed_trade_type = "NONE"  # DIRECTIONAL, AGGRESSIVE_REVERSAL, NONE

        self.event_dt = event_dt
        self.start_dt = start_dt
        self.start_loc = df.index.get_loc(start_dt)
        self.stop_dt = stop_dt
        self.stop_loc = df.index.get_loc(stop_dt)
        self.opposite_stacker_off_dt = opposite_stacker_off_dt
        self.opposite_stacker_off_loc = df.index.get_loc(self.opposite_stacker_off_dt)
        self.entry_stop_time_dt = entry_stop_time_dt
        self.entry_stop_time_loc = df.index.get_loc(self.entry_stop_time_dt)
	self.time_decay_exit_start_dt = time_decay_exit_start_dt
	self.time_decay_exit_start_loc = df.index.get_loc(self.time_decay_exit_start_dt)
        self.trade_termination_dt = trade_termination_dt  #self.stop_dt - timedelta(seconds=pre_trading_stop_secs)
        self.trade_termination_loc = df.index.get_loc(self.trade_termination_dt)
	self.reversal_tracking_start_dt = reversal_tracking_start_dt
	self.reversal_tracking_start_loc = df.index.get_loc(self.reversal_tracking_start_dt)
	self.stop_loss_tracking_start_dt = stop_loss_tracking_start_dt
	self.stop_loss_tracking_start_loc = df.index.get_loc(self.stop_loss_tracking_start_dt)

	self.best_potential_pnl_time_dt = start_dt
	self.worst_potential_pnl_time_dt = start_dt
	self.close_position_time_dt = stop_dt

        self.pre_event_price = 0
	self.cur_disl_ticks = 0
        self.max_up_disl_ticks = 0
        self.max_down_disl_ticks = 0
	self.max_disl_ticks = 0
	self.best_percent_reversal = 0

	self.cur_imbalance = 0

        self.max_pos_pos = 0
        self.max_neg_pos = 0
        self.max_abs_pos = 0
	self.previous_pos = 0

        self.out_orders = []
        self.last_order_id = 0

        self.my_live_orders = []  # always comes from Simulator
        self.my_exit_orders = []  # kept internal to the Strategy

	self.initial_target_pnl_per_contract = 0
	self.time_decay_coeff = 0
	self.reversal_percent_target = 0	
	self.stop_loss_pnl_per_contract = -10000

	self.TDT = TimeDecayTrigger(trigger_period_seconds=1.0)
	self.target_exit_price = 0 
	self.previous_target_exit_price = 0	
	self.cpu_process_id = cpu_process_id


    def round_to_nearest_price(self, raw_price):
        return round(float(raw_price) / min_price_increment) * min_price_increment


    def start(self):
        self.cur_loc  = self.mySim.start_sim(self.start_dt)
	
	if not self.delayed_trade:
	    print "***************************************** DELAYED TRADE"
            self.cur_loc, fills = self.initOrderStack()

        self.pre_event_price = self.round_to_nearest_price(self.df.ix[self.cur_loc, 'microprice'])

        #print "      ", self.df.ix[self.cur_loc, 'time']
        #print "        pos: ", self._pos
        #print "        PnL: ", self._pnl
        #print

	self.trade_running = True

        while (self.cur_loc < self.stop_loc):
	    #print self.run_trade_only, self.trade_running
	    if self.run_trade_only:
		if not self.trade_running:
		    print "TERMINATING EARLY: Trade is done.", self.df.ix[self.cur_loc, 'time']
		    return 0

            self.out_orders = []
            #print self.df.ix[self.cur_loc, 'time'], "mid market:", str((self.df.ix[self.cur_loc, 'top_bid_price'] + self.df.ix[self.cur_loc, 'top_ask_price']) / 2.0)
#TEMPOFF            print self.df.ix[self.cur_loc, 'time'], "ask:", self.df.ix[self.cur_loc, 'top_ask_price'], "bid:", self.df.ix[self.cur_loc, 'top_bid_price']
            self.cur_loc, fills = self.onTick()
            self.update(self.cur_loc, fills)

            #print "fills:", fills

#            if fills:
#                print "      ", self.df.ix[self.cur_loc, 'time']
#                print "        pos: ", self._pos
#                print "        PnL: ", self._pnl
#		print "        vol: ", self._vol
#		print "        pnl per contract", self.pnl_per_contract
#
#                print
        return 1


    def update(self, mkt_data_loc, fills):
        self.process_fills(fills)
        self._pnl = self.computePnL()
        self.computeEstimatePnL()
	self.computePnLperContract()


    def on_fill(self, size, price):
        if size != 0:
            self._pos += size
            self._cash -= (size * price * self.dollar_value_per_price_level)
            self._vol += abs(size)
            self._ltp = price

            if self.best_potential_pnl == 0 and self._pnl < 0:
                self.best_potential_pnl = self._pnl
            elif self._pnl > self.best_potential_pnl:
                self.best_potential_pnl = self._pnl

            #print self.df.ix[self.cur_loc, 'time'], self._pos, self.max_pos_pos, self.max_neg_pos, self._pnl, self.best_potential_pnl

            if self._pos > 0:
                if self._pos > self.max_pos_pos:
                    self.max_pos_pos = self._pos
            elif self._pos < 0:
                if abs(self._pos) > self.max_neg_pos:
                    self.max_neg_pos = abs(self._pos)
            self.max_abs_pos = max(self.max_pos_pos, self.max_neg_pos)


    def process_fills(self, fills):
        for fill in fills:
            #print fill
            for partial_fill in fill['partial_fills']:
                self.on_fill(partial_fill['size'], partial_fill['price'])


    def computePnL(self):
        if self._pos >= 0:
            price = self.df.ix[self.cur_loc, 'top_bid_price']
        else:
            price = self.df.ix[self.cur_loc, 'top_ask_price']

        return self._cash + float(self._pos) * float(price) * self.dollar_value_per_price_level


    def computeEstimatePnL(self):
        self.potential_pnl = self.computePnL()

        if self.best_potential_pnl == 0 and self.potential_pnl < 0:
            self.best_potential_pnl = self.potential_pnl
	    self.best_potential_pnl_time_dt = self.df.ix[self.cur_loc, 'time']
        elif self.potential_pnl > self.best_potential_pnl:
            self.best_potential_pnl = self.potential_pnl
	    self.best_potential_pnl_time_dt = self.df.ix[self.cur_loc, 'time']

	if self.worst_potential_pnl == 0 and self.potential_pnl > 0:
	    self.worst_potential_pnl = self.potential_pnl
	    self.worst_potential_pnl_time_dt = self.df.ix[self.cur_loc, 'time']	    
	elif self.potential_pnl < self.worst_potential_pnl:
	    self.worst_potential_pnl = self.potential_pnl
            self.worst_potential_pnl_time_dt = self.df.ix[self.cur_loc, 'time']


    def computePnLperContract(self):
	estimated_volume = self._vol + abs(self._pos)
	if estimated_volume > 0:
	    self.pnl_per_contract = self.potential_pnl / float(estimated_volume)
	else:
	    self.pnl_per_contract = 0


    def computeImbalance(self):
	bid_size_0 = self.df.ix[self.cur_loc, 'bid_size_0']
	bid_size_1 = self.df.ix[self.cur_loc, 'bid_size_1']
	bid_size_2 = self.df.ix[self.cur_loc, 'bid_size_2']
	bid_size_3 = self.df.ix[self.cur_loc, 'bid_size_3']
	bid_size_4 = self.df.ix[self.cur_loc, 'bid_size_4']
	ask_size_0 = self.df.ix[self.cur_loc, 'ask_size_0']
        ask_size_1 = self.df.ix[self.cur_loc, 'ask_size_1']
        ask_size_2 = self.df.ix[self.cur_loc, 'ask_size_2']
        ask_size_3 = self.df.ix[self.cur_loc, 'ask_size_3']
        ask_size_4 = self.df.ix[self.cur_loc, 'ask_size_4']
	bid_size = bid_size_0 + bid_size_1 + bid_size_2 + bid_size_3 + bid_size_4
	ask_size = ask_size_0 + ask_size_1 + ask_size_2 + ask_size_3 + ask_size_4

	#print bid_size_0, bid_size_1, bid_size_2, bid_size_3, bid_size_4
	#print ask_size_0, ask_size_1, ask_size_2, ask_size_3, ask_size_4
	#print bid_size, ask_size
	
	if bid_size > 0 and ask_size > 0:
	    self.cur_imbalance = float(bid_size - ask_size) / float(bid_size + ask_size)
	else:
	    self.cur_imbalance = 0


    def updateExitOrders(self):
        valid_exit_orders = []

        exit_order_ids = [exit_order['id'] for exit_order in self.my_exit_orders]
	valid_exit_orders = [live_order for live_order in self.my_live_orders if live_order['id'] in exit_order_ids]
        self.my_exit_orders = valid_exit_orders


    def time_offset_loc(self, offset_secs):
        return self.df.index.get_loc(self.start_dt + timedelta(seconds=offset_secs))


    def getNewOrderID(self):
        new_order_id = self.last_order_id + 1
        self.last_order_id = new_order_id
        return new_order_id


    def createNewOrder(self, type, direction, price, size, order_time_loc):
        new_order = {'id': self.getNewOrderID(), 'type': type, 'direction': direction, 'price': price, 'size': size, 'order_time_loc': order_time_loc}

        return new_order


    def createBuyOrder(self, price, size, order_time_loc):
        new_order = self.createNewOrder("limit", "buy", price, size, order_time_loc)

        return new_order


    def createSellOrder(self, price, size, order_time_loc):
        new_order = self.createNewOrder("limit", "sell", price, size, order_time_loc)

        return new_order


    def createFAKBuyOrder(self, price, size, order_time_loc):
        new_order = self.createNewOrder("fak", "buy", price, size, order_time_loc)

        return new_order


    def createFAKSellOrder(self, price, size, order_time_loc):
        new_order = self.createNewOrder("fak", "sell", price, size, order_time_loc)

        return new_order


    def createCancelOrder(self, id, order_time_loc):
        cancel_order = {'id': id, 'type': "cancel", 'direction': "", 'price': None, 'size': None, 'order_time_loc': order_time_loc}

        return cancel_order

    
    def sizeCurveFunction(self, form, param1, param2, edge, X):
        if form == "linear":
            Y = param1 * (X - edge) + param2
        elif form == "quadratic":
            Y = param1 * (X - edge)**2 + param2

        Yrounded = int(round(Y,0))

        return Yrounded


    def buildBidOrderCurve(self, form, param1, param2, depth, skip, edge, top_bid_price):
        bid_curve_orders = []

        start_price = top_bid_price - edge * min_price_increment

        first_price_level = edge
        last_price_level = first_price_level + depth - 1

        for lev in xrange(first_price_level, first_price_level + depth, skip+1):
            size = min(self.sizeCurveFunction(form, param1, param2, edge, lev), max_size_on_level)
            price = top_bid_price - lev * min_price_increment

            new_order = self.createNewOrder("limit", "buy", price, size, self.cur_loc)
            bid_curve_orders.append(new_order)

        return bid_curve_orders


    def buildAskOrderCurve(self, form, param1, param2, depth, skip, edge, top_ask_price):
        ask_curve_orders = []

        start_price = top_ask_price + edge * min_price_increment

        first_price_level = edge
        last_price_level = first_price_level + depth - 1

        for lev in xrange(first_price_level, first_price_level + depth, skip+1):
            size = min(self.sizeCurveFunction(form, param1, param2, edge, lev), max_size_on_level)
            price = top_ask_price + lev * min_price_increment

            new_order = self.createNewOrder("limit", "sell", price, size, self.cur_loc)
            ask_curve_orders.append(new_order)

        return ask_curve_orders


    def buildFullOrderStack(self, bid_form, bid_param1, bid_param2, bid_depth, bid_skip, bid_edge,
                                ask_form, ask_param1, ask_param2, ask_depth, ask_skip, ask_edge,
                                top_bid_price, top_ask_price):
        order_stack = []
        bid_order_stack = []
        ask_order_stack = []
        if bid_form != "none":
            bid_order_stack = self.buildBidOrderCurve(bid_form, bid_param1, bid_param2, bid_depth, bid_skip, bid_edge, top_bid_price)
            self.myStacker.bid_stacker_orders = bid_order_stack
            order_stack.extend(bid_order_stack)
        if ask_form != "none":
            ask_order_stack = self.buildAskOrderCurve(ask_form, ask_param1, ask_param2, ask_depth, ask_skip, ask_edge, top_ask_price)
            self.myStacker.ask_stacker_orders = reversed(ask_order_stack)
            order_stack.extend(ask_order_stack)

        #self.printOrderStack(ask_order_stack, bid_order_stack)

        return order_stack


    def initOrderStack(self):
        top_bid_price = self.df.ix[self.cur_loc, 'top_bid_price']
        top_ask_price = self.df.ix[self.cur_loc, 'top_ask_price']

        print "sending stack to market:", self.df.ix[self.cur_loc, 'time']
        stack_orders = self.buildFullOrderStack(bid_form, bid_param1, bid_param2, bid_depth, bid_skip, bid_edge,
                                                ask_form, ask_param1, ask_param2, ask_depth, ask_skip, ask_edge,
                                                top_bid_price, top_ask_price)

        self.out_orders.extend(stack_orders)


        mkt_loc, fills, self.my_live_orders = self.mySim.execute(stack_orders)

        return mkt_loc, fills


    def printOrderStack(self, ask_order_stack, bid_order_stack):
        print "ORDER STACK:"
        for ao in reversed(ask_order_stack):
            print(str(ao['id']) + "\t" + "\t" + str(ao['price']) + "\t" + str(ao['size'])).expandtabs(15)
        print "-----------------------------------------------"
        for bo in bid_order_stack:
            print(str(bo['id']) + "\t" + str(bo['size']) + "\t" + str(bo['price']) + "\t" + "  ").expandtabs(15)
        print


    def cancelAskOrderStack(self):
        for order in self.myStacker.ask_stacker_orders:
            cancel_order = self.createCancelOrder(order['id'], self.cur_loc)
            self.out_orders.append(cancel_order)
        self.myStacker.ask_stacker_orders = []
	print "^^^^^^^^^^^^^^^^ASK stack got cancelled", self.df.ix[self.cur_loc, 'time']


    def cancelBidOrderStack(self):
        for order in self.myStacker.bid_stacker_orders:
            cancel_order = self.createCancelOrder(order['id'], self.cur_loc)
            self.out_orders.append(cancel_order)
        self.myStacker.bid_stacker_orders = []
	print "^^^^^^^^^^^^^^^^BID stack got cancelled", self.df.ix[self.cur_loc, 'time']


    def cancelOrderStack(self):
        self.cancelAskOrderStack()
        self.cancelBidOrderStack()


    def isLiveStacker(self):
        if self.myStacker.ask_stacker_orders or self.myStacker.bid_stacker_orders:
            return True
        else:
            return False


    def generateCrossBookExitOrder(self, top_bid_price, top_ask_price, ticks_to_cross, size):
        if self._pos > 0:
            exit_price = top_bid_price - ticks_to_cross * min_price_increment
            exit_size = size
            exit_sell_order = self.createSellOrder(exit_price, exit_size, self.cur_loc)
            self.out_orders.append(exit_sell_order)
        elif self._pos < 0:
            exit_price = top_ask_price + ticks_to_cross * min_price_increment
            exit_size = size
            exit_buy_order = self.createBuyOrder(exit_price, exit_size, self.cur_loc)
            self.out_orders.append(exit_buy_order)


    def cancelExitOrder(self, exit_order):
        cancel_exit_order = self.createCancelOrder(exit_order['id'], self.cur_loc)
        self.out_orders.append(cancel_exit_order)
        if exit_order in self.my_exit_orders:
            self.my_exit_orders.remove(next((x for x in self.my_exit_orders if x['id'] == exit_order['id']), None))

            #print "  ********"
            #print "  exit_order: ", exit_order
            #print "  ", self.df.ix[self.cur_loc, 'time'], "self.my_exit_orders: ", self.my_exit_orders
            #print "  ********"


    def cancelAllExitOrders(self, total_exit_size_at_risk):
        my_exit_orders_cpy = copy.deepcopy(self.my_exit_orders)
        for exit_order in my_exit_orders_cpy:
            self.cancelExitOrder(exit_order)
            total_exit_size_at_risk -= exit_order['size']

        return total_exit_size_at_risk


    def cancelExitOrders(self, exit_orders_to_cancel, total_exit_size_at_risk):
        for exit_order in exit_orders_to_cancel:
            self.cancelExitOrder(exit_order)
            total_exit_size_at_risk -= exit_order['size']

        return total_exit_size_at_risk


    def reEvaluateExitOrders(self, target_exit_price, total_exit_size_at_risk, max_size_per_level=10, max_num_exit_levels=2):
        print self.df.ix[self.cur_loc, 'time'], " self._pos:", self._pos, "self._pnl:", self._pnl
        #print "total_exit_size_at_risk: ", total_exit_size_at_risk
        #print "self.my_live_orders:", self.my_live_orders
        #print self.df.ix[self.cur_loc, 'time'], "self.my_exit_orders: ", self.my_exit_orders
	print "self.isLiveStacker(): ", self.isLiveStacker(), self.df.ix[self.cur_loc, 'time']
        if self._pos != 0 and abs(self._pos) < total_exit_size_at_risk:
            self.cancelAllExitOrders(total_exit_size_at_risk)
        elif abs(self._pos) < self.max_abs_pos and self.cur_loc >= self.entry_stop_time_loc and abs(self._pos) < small_abs_pos and self.isLiveStacker():
            self.cancelOrderStack()
            print "cancelled Curve |#1|", self.df.ix[self.cur_loc, 'time']
        elif self.cur_loc >= self.entry_stop_time_loc and abs(self._pos) == 0 and self.isLiveStacker():
            self.cancelOrderStack()
	    self.trade_running = False
            print "cancelled Curve |#2|", self.df.ix[self.cur_loc, 'time']
	elif self.cur_loc >= self.entry_stop_time_loc and abs(self._pos) == 0:  # no pos, past entry time --> end trade
	    self.trade_running = False
	elif self.cur_loc >= self.entry_stop_time_loc and self.isLiveStacker():
	    self.cancelOrderStack()
	    print "cancelled Curve |#1.u5|", self.df.ix[self.cur_loc, 'time']
	elif abs(self._pos) >= max_allowed_pos and self.isLiveStacker():
	    self.cancelOrderStack()
            print "cancelled Curve |#3|", self.df.ix[self.cur_loc, 'time']
	    print "  ---> current market: ", self.df.ix[self.cur_loc, 'top_ask_price'], "x", self.df.ix[self.cur_loc, 'top_bid_price']
        elif self._pos != 0: 
            if self._pos > 0:
                exit_direction = "sell"
            else:
                exit_direction = "buy"

            wrong_direction_exit_orders = [o for o in self.my_exit_orders if o['direction'] != exit_direction]  # for case where position flips
            total_exit_size_at_risk = self.cancelExitOrders(wrong_direction_exit_orders, total_exit_size_at_risk)

            if self.cur_loc >= self.trade_termination_loc \
			or (self.best_percent_reversal >= self.reversal_percent_target and self.cur_loc >= self.reversal_tracking_start_loc) \
			or (self.pnl_per_contract <= self.stop_loss_pnl_per_contract and self.cur_loc >= self.stop_loss_tracking_start_loc):
                ##NOTE: removed this because passive order can be cancelled without aggressive order filled
		##if self.my_exit_orders:
                ##    total_exit_size_at_risk = self.cancelAllExitOrders(total_exit_size_at_risk)

                exit_size = abs(self._pos)
                if self._pos > 0:
                    exit_price = self.df.ix[self.cur_loc, 'top_bid_price'] - min_price_increment * max_levels_to_cross
                    exit_order = self.createSellOrder(exit_price, exit_size, self.cur_loc)
                    self.out_orders.append(exit_order)
                    self.my_exit_orders.append(exit_order)
		    print "self.best_percent_reversal: ", self.best_percent_reversal, "self.reversal_percent_target: ", self.reversal_percent_target
		    if self.best_percent_reversal >= self.reversal_percent_target:
			print "reached reversal target - sent cross-book sell order: ", exit_size, exit_price, self.df.ix[self.cur_loc, 'time']
		    elif self.pnl_per_contract <= self.stop_loss_pnl_per_contract:
			print "reached stop loss pnl per contract - sent cross-book sell order: ", exit_size, exit_price, self.df.ix[self.cur_loc, 'time']
		    else:
                        print "reached end-time - sent emergency sell order: ", exit_size, exit_price, self.df.ix[self.cur_loc, 'time']
                elif self._pos < 0:
                    exit_price = self.df.ix[self.cur_loc, 'top_ask_price'] + min_price_increment * max_levels_to_cross
                    exit_order = self.createBuyOrder(exit_price, exit_size, self.cur_loc)
                    self.out_orders.append(exit_order)
                    self.my_exit_orders.append(exit_order)
		    print "self.best_percent_reversal: ", self.best_percent_reversal, "self.reversal_percent_target: ", self.reversal_percent_target
		    if self.best_percent_reversal >= self.reversal_percent_target:
			print "reached reversal target - sent cross-book buy order: ", exit_size, exit_price, self.df.ix[self.cur_loc, 'time']
		    elif self.pnl_per_contract <= self.stop_loss_pnl_per_contract:
			print "reached stop loss pnl per contract - sent cross-book buy order: ", exit_size, exit_price, self.df.ix[self.cur_loc, 'time']
		    else:
                        print "reached end-time - sent emergency buy order: ", exit_size, exit_price, self.df.ix[self.cur_loc, 'time']

                return

	    if target_exit_price != self.previous_target_exit_price:
		target_exit_price_changed = True
	    else:
		target_exit_price_changed = False

	    
	    if self.TDT.isStarted():
		exit_orders_authority = self.my_live_orders
                if self.TDT.isTriggered(self.df.ix[self.cur_loc, 'time']) and target_exit_price_changed:
                    total_exit_size_at_risk = self.cancelAllExitOrders(total_exit_size_at_risk)  # needed for cancel-replace effect <==== the correct way to handle this is probably to (1) figure out new front, (2) figure out which are new levels, (3) cancel orders on abandoned levels only, (4) add allowed number of additional size starting from new front.
		    #print "self.TDT.isTriggered:", self.TDT.isTriggered(self.df.ix[self.cur_loc, 'time']), self.df.ix[self.cur_loc, 'time']
		    exit_orders_authority = self.my_exit_orders		    

		base_exit_price = target_exit_price

                available_exit_size = abs(self._pos) - total_exit_size_at_risk
                #print "available_exit_size: ", available_exit_size, "  total_exit_size_at_risk :", total_exit_size_at_risk, self.df.ix[self.cur_loc, 'time']
                #print "base_exit_price: ", base_exit_price, "   exit_direction: ", exit_direction

		if available_exit_size:
                    for x in xrange(0, max_num_exit_levels):
                        if self._pos > 0:
                            exit_price = base_exit_price + x
                        elif self._pos < 0:
                            exit_price = base_exit_price - x
		    
                        total_size_on_exit_level = 0
                        for live_order in exit_orders_authority:  
                            if live_order['price'] == exit_price and live_order['direction'] == exit_direction:
                                total_size_on_exit_level += live_order['size']

                        exit_size = 0
                        if total_size_on_exit_level < max_size_per_level and available_exit_size:
                            exit_size = min(max_size_per_level - total_size_on_exit_level, available_exit_size)
                            available_exit_size -= exit_size

                        if self._pos > 0 and exit_size > 0:
                            exit_order = self.createSellOrder(exit_price, exit_size, self.cur_loc)
                            self.out_orders.append(exit_order)
                            self.my_exit_orders.append(exit_order)
                        elif self._pos < 0 and exit_size > 0:
                            exit_order = self.createBuyOrder(exit_price, exit_size, self.cur_loc)
                            self.out_orders.append(exit_order)
                            self.my_exit_orders.append(exit_order)
#                        print "self.my_exit_orders: ", self.my_exit_orders

	elif self._pos == 0 and self.previous_pos != 0:
	    print "!!! just closed position !!!", self.df.ix[self.cur_loc, 'time']
	    if self.isLiveStacker():
                self.cancelOrderStack()
                print "cancelled Curve |#4|", self.df.ix[self.cur_loc, 'time']
	    if total_exit_size_at_risk > 0:
		self.cancelAllExitOrders(total_exit_size_at_risk)
            	print "cancelled passive exit orders", self.df.ix[self.cur_loc, 'time']
	    self.cancelAllExitOrders(total_exit_size_at_risk) 
	    self.close_position_time_dt = self.df.ix[self.cur_loc, 'time']
	    self.trade_running = False

	# just in case...
	if self._pos == 0 and total_exit_size_at_risk > 0:
            self.cancelAllExitOrders(total_exit_size_at_risk)
            print "cancelled passive exit orders", self.df.ix[self.cur_loc, 'time']

	self.previous_target_exit_price = target_exit_price
	self.previous_pos = self._pos

        #print self.df.ix[self.cur_loc, 'time'], "self.my_exit_orders: ", self.my_exit_orders
        #print "-----"



    def exitLogic3(self, max_disl_ticks, top_bid_price, top_ask_price):   
	initial_target_reversal_coeff = 0.45  #1.0
	time_decay_coeff = -0.0005   #-0.001

	#print "self.pre_event_price:", self.pre_event_price, "self.max_up_disl_ticks:", self.max_up_disl_ticks, "self.max_down_disl_ticks:", self.max_down_disl_ticks, "   ", self.df.ix[self.cur_loc, 'time']
	if self.TDT.isStarted():
            if self.TDT.isTriggered(self.df.ix[self.cur_loc, 'time']):  # <----- may want to add condition: max_disl_ticks != 0
		#print "self.pre_event_price:", self.pre_event_price, "self.max_up_disl_ticks:", self.max_up_disl_ticks, "self.max_down_disl_ticks:", self.max_down_disl_ticks, "   ", self.df.ix[self.cur_loc, 'time']
		#print "---->self.max_disl_ticks:", max_disl_ticks, "   ", self.df.ix[self.cur_loc, 'time']
	        initial_target_reversal_ticks = initial_target_reversal_coeff * abs(max_disl_ticks)
	        decay_adjustment_ticks = time_decay_coeff * self.TDT.periodsElapsed(self.df.ix[self.cur_loc, 'time']) * abs(max_disl_ticks)
	        adjusted_target_reversal_ticks = initial_target_reversal_ticks + decay_adjustment_ticks

	        target_exit_ticks = max_disl_ticks * (1 - adjusted_target_reversal_ticks/float(abs(max_disl_ticks)))
	        target_exit_price = self.round_to_nearest_price(self.pre_event_price + target_exit_ticks * min_price_increment)
	        self.target_exit_price = target_exit_price
		#print "self.max_disl_ticks:", max_disl_ticks, "initial_target_reversal_ticks:", initial_target_reversal_ticks, "decay_adjustment_ticks:", decay_adjustment_ticks, "target_exit_ticks:", target_exit_ticks, "   ", self.df.ix[self.cur_loc, 'time'] 
	else:
	    #self.target_exit_price = self.df.ix[self.cur_loc, 'midquote']
	    self.target_exit_price = float('NaN')

	self.simlog_df.ix[self.cur_loc, 'target_exit_price'] = self.target_exit_price


	max_size_per_level = 20   #<============== change to 10 for CL, better yet, have lookup dict by instrument root
        max_num_exit_levels = 5	

	live_order_ids = [o['id'] for o in self.my_live_orders]
        total_exit_size_at_risk = sum(o['size'] for o in self.my_exit_orders if o['id'] in live_order_ids)

	self.reEvaluateExitOrders(self.target_exit_price, total_exit_size_at_risk, max_size_per_level, max_num_exit_levels)


    def estimatePriceLevelforPnL(self, target_pnl_per_contract, max_size_per_level, max_num_exit_levels):
	#print "##############################################"

	target_pnl = target_pnl_per_contract * (self._vol + abs(self._pos))
	#print abs(self._pos), float(max_size_per_level * max_num_exit_levels), abs(self._pos) / float(max_size_per_level * max_num_exit_levels), math.log(abs(self._pos) / float(max_size_per_level * max_num_exit_levels)), round(1 + math.log(abs(self._pos) / float(max_size_per_level * max_num_exit_levels)))
	abs_level_offset =round(math.log(1 + abs(self._pos) / float(max_size_per_level * max_num_exit_levels)))
	
	#backout_price = self.round_to_nearest_price((target_pnl - self._cash) / float(self._pos * self.dollar_value_per_price_level))

	#print "target_pnl_per_contract: ", target_pnl_per_contract, self.df.ix[self.cur_loc, 'time']
	#print "target_pnl: ", target_pnl, "self._vol + abs(self._pos): ", self._vol + abs(self._pos)
	#print " self._cash:", self._cash
	#print " avg price:", self._cash / (self._pos * self.dollar_value_per_price_level)
	#print "backout_price: ", backout_price, self.df.ix[self.cur_loc, 'time']
	#print "abs_level_offset: ", abs_level_offset
	#print "current mid market: ", (self.df.ix[self.cur_loc, 'top_bid_price'] + self.df.ix[self.cur_loc, 'top_bid_price']) / 2.0 

	avg_entry_price = abs(self._cash / (self._pos * self.dollar_value_per_price_level))
	num_price_increments = float(target_pnl_per_contract * min_price_increment) / float(self.dollar_value_per_price_level)

	global min_price_increment

	if self._pos >= 0:
	    backout_price = self.round_to_nearest_price(avg_entry_price + num_price_increments)
	    target_price = backout_price - abs_level_offset * min_price_increment
	else:
	    backout_price = self.round_to_nearest_price(avg_entry_price - num_price_increments)
	    target_price = backout_price + abs_level_offset * min_price_increment

	
	##print "target_exit_price: ", target_price, "top_ask_price: ", self.df.ix[self.cur_loc, 'top_ask_price']
	#sys.exit(0)
	return target_price	


    def exitLogic4(self):
	max_size_per_level = 20   #<============== change to 10 for CL, better yet, have lookup dict by instrument root
        max_num_exit_levels = 5

	if self.TDT.isStarted():
            if self.TDT.isTriggered(self.df.ix[self.cur_loc, 'time']):
		if self._pos != 0:
		    target_pnl_per_contract = self.initial_target_pnl_per_contract + self.time_decay_coeff * self.TDT.periodsElapsed(self.df.ix[self.cur_loc, 'time'])
		    self.target_exit_price = self.estimatePriceLevelforPnL(target_pnl_per_contract, max_size_per_level, max_num_exit_levels)
		else:
		    self.target_exit_price = float('NaN')
	else:
	    self.target_exit_price = float('NaN')
	print "target_exit_price: ", self.target_exit_price, "top_ask_price: ", self.df.ix[self.cur_loc, 'top_ask_price'] 

        self.simlog_df.ix[self.cur_loc, 'target_exit_price'] = self.target_exit_price

        live_order_ids = [o['id'] for o in self.my_live_orders]
        total_exit_size_at_risk = sum(o['size'] for o in self.my_exit_orders if o['id'] in live_order_ids)

        self.reEvaluateExitOrders(self.target_exit_price, total_exit_size_at_risk, max_size_per_level, max_num_exit_levels) 


    def delayedDirectionalEntryLogic(self):
	if self.cur_loc >= self.delayed_trade_start_loc and self.cur_loc < self.entry_stop_time_loc:
	    entry_size = max_allowed_pos
	    print "imbalance: ", self.cur_imbalance
	    print self.cur_disl_ticks, self.max_disl_ticks
	    if self.cur_disl_ticks < 0 and self.max_disl_ticks < 0 and self.cur_imbalance < 0:  # down move
		print "sending directional order DOWN"
		entry_price = self.df.ix[self.cur_loc, 'top_bid_price'] - min_price_increment * max_levels_to_cross
                entry_order = self.createFAKSellOrder(entry_price, entry_size, self.cur_loc)
                self.out_orders.append(entry_order)
		self.initial_entry_sent = True
	    elif self.cur_disl_ticks > 0 and self.max_disl_ticks > 0 and self.cur_imbalance > 0:  # up move
		print "sending directional order UP"
		entry_price = self.df.ix[self.cur_loc, 'top_ask_price'] + min_price_increment * max_levels_to_cross
                entry_order = self.createFAKBuyOrder(entry_price, entry_size, self.cur_loc)
                self.out_orders.append(entry_order)
		self.initial_entry_sent = True		


    def delayedReversalAggressiveEntryLogic(self):
	pass


    def delayedOneSidedCurveEntryLogic(self):
	top_bid_price = self.df.ix[self.cur_loc, 'top_bid_price']
        top_ask_price = self.df.ix[self.cur_loc, 'top_ask_price']
	order_stack = []
        bid_order_stack = []
        ask_order_stack = []
	one_sided_edge = 2

	if self.cur_loc >= self.delayed_trade_start_loc and self.cur_loc < self.entry_stop_time_loc and not self.isLiveStacker():
	    #print self.cur_disl_ticks, self.max_disl_ticks
	    #sys.exit(0)
            if self.cur_disl_ticks < 0 and self.max_disl_ticks < 0 and bid_form != "none":
                bid_order_stack = self.buildBidOrderCurve(bid_form, bid_param1, bid_param2, bid_depth, bid_skip, one_sided_edge, top_bid_price)
                self.myStacker.bid_stacker_orders = bid_order_stack
                order_stack.extend(bid_order_stack)
		self.out_orders.extend(order_stack)
		self.initial_entry_sent = True
            elif self.cur_disl_ticks > 0 and self.max_disl_ticks > 0 and ask_form != "none":
                ask_order_stack = self.buildAskOrderCurve(ask_form, ask_param1, ask_param2, ask_depth, ask_skip, one_sided_edge, top_ask_price)
                self.myStacker.ask_stacker_orders = reversed(ask_order_stack)
                order_stack.extend(ask_order_stack)
		self.out_orders.extend(order_stack)
		self.initial_entry_sent = True    
		

    def delayedEntryLogic(self):
	if not self.initial_entry_sent \
		and self.cur_loc >= self.delayed_trade_start_loc and self.cur_loc < self.entry_stop_time_loc and not self.isLiveStacker() and self._pos == 0:
	    if self.delayed_trade_type == "ONE-SIDED-CURVE":
		print "Generating entry as DELAYED ONE-SIDED-CURVE"
		self.delayedOneSidedCurveEntryLogic()
		#self.initial_entry_sent = True
	    elif self.delayed_trade_type == "DIRECTIONAL":
   		print "Generating entry as DELAYED DIRECTIONAL: ", self.cur_disl_ticks
		self.delayedDirectionalEntryLogic()
		#self.initial_entry_sent = True


    def onTick(self):
        top_bid_price = self.df.ix[self.cur_loc, 'top_bid_price']
        top_ask_price = self.df.ix[self.cur_loc, 'top_ask_price']

        ask_size = self.df.ix[self.cur_loc, ['ask_size_0', 'ask_size_1', 'ask_size_2', 'ask_size_3', 'ask_size_4']]
        bid_size = self.df.ix[self.cur_loc, ['bid_size_0', 'bid_size_1', 'bid_size_2', 'bid_size_3', 'bid_size_4']]

	self.computeImbalance()

        if self.cur_loc >= self.start_loc:# and self.cur_loc < self.entry_stop_time_loc:  # was previously entry_stop_time  
            disl_ticks = (self.df.ix[self.cur_loc, 'microprice'] - self.pre_event_price) / float(min_price_increment)
	    self.cur_disl_ticks = disl_ticks
            if disl_ticks > self.max_up_disl_ticks:
                self.max_up_disl_ticks = disl_ticks
            elif disl_ticks < self.max_down_disl_ticks:
                self.max_down_disl_ticks = disl_ticks

            #print "self.max_up_disl_ticks: ", self.max_up_disl_ticks, "  self.max_down_disl_ticks: ", self.max_down_disl_ticks

        if self.cur_loc == self.opposite_stacker_off_loc:
            #if self.max_up_disl_ticks > abs(self.max_down_disl_ticks):  # cancel bid side
            if self.max_up_disl_ticks > abs(self.max_down_disl_ticks) or self._pos < 0:  # cancel bid side
                for order in self.myStacker.bid_stacker_orders:
                    cancel_order = self.createCancelOrder(order['id'], self.cur_loc)
                    self.out_orders.append(cancel_order)
                self.myStacker.bid_stacker_orders = []
            elif abs(self.max_down_disl_ticks) > self.max_up_disl_ticks or self._pos > 0:  # cancel ask side
                for order in self.myStacker.ask_stacker_orders:
                    cancel_order = self.createCancelOrder(order['id'], self.cur_loc)
                    self.out_orders.append(cancel_order)
                self.myStacker.ask_stacker_orders = []
	
	if self.cur_loc < self.entry_stop_time_loc:
            if (self.max_up_disl_ticks > abs(self.max_down_disl_ticks) \
		or ((self.delayed_trade_type == "ONE-SIDED-CURVE" or self.delayed_trade_type == "REVERSAL-AGGRESSIVE") and self._pos < 0) \
		or (self.delayed_trade_type == "DIRECTIONAL" and self._pos > 0) ):  #or self._pos < 0 
                    self.max_disl_ticks = self.max_up_disl_ticks
            elif (abs(self.max_down_disl_ticks) > self.max_up_disl_ticks \
		or ((self.delayed_trade_type == "ONE-SIDED-CURVE" or self.delayed_trade_type == "REVERSAL-AGGRESSIVE") and self._pos > 0) \
                or (self.delayed_trade_type == "DIRECTIONAL" and self._pos < 0) ):  #or self._pos > 0
                    self.max_disl_ticks = self.max_down_disl_ticks
	if self.max_disl_ticks != 0:
	    cur_reversal = -1.0 * (self.max_disl_ticks - self.cur_disl_ticks)
	    cur_percent_reversal = abs(float(cur_reversal) / float(self.max_disl_ticks))
	    self.best_percent_reversal = cur_percent_reversal

	    print "self.max_disl_ticks: ", self.max_disl_ticks, "disl_ticks: ", self.cur_disl_ticks, "cur_reversal: ", cur_reversal, "cur_percent_reversal: ", self.best_percent_reversal, self.df.ix[self.cur_loc, 'time']

	if self.cur_loc == self.time_decay_exit_start_loc:  # <----- can add condition:  self._pos != 0
	    self.TDT.start(self.df.ix[self.cur_loc, 'time'])	

        #self.exitLogic3(self.max_disl_ticks, top_bid_price, top_ask_price)
	self.exitLogic4()
	
	if self.delayed_trade:
	    self.delayedEntryLogic()

        fills = []

        mkt_loc, fills, self.my_live_orders = self.mySim.execute(self.out_orders)

        self.updateExitOrders()

        return mkt_loc, fills



def readEventDatesTimes(event_name, instrument_root, directory="./"):
    dates_times_dict = {}
    in_filename = directory + event_name + "_" + instrument_root + "_dates_times.csv"
    for key, val in csv.reader(open(in_filename)):
        dates_times_dict[key] = val

    return dates_times_dict


def readBestPnLperContract(event_name, instrument_root, directory="./"):
    best_pnl_per_contract_dict = {}
    in_filename = directory + event_name + "_" + instrument_root + "_best_pnl_per_contract.csv"
    for key, val in csv.reader(open(in_filename)):
        best_pnl_per_contract_dict[key] = float(val)

    return best_pnl_per_contract_dict


def serializePnLResults(pnl_results, incl_timestamp=True, info_str="test_out"):
    cur_time_dt = datetime.now()
    cur_time_str = cur_time_dt.strftime("%Y%m%d_%H%M%S")
    out_filename = "pnl_results_" + info_str
    if incl_timestamp:
	out_filename = out_filename + "__" + cur_time_str 	

    csvWriter = csv.writer(open(out_filename, 'w'), delimiter=',')

    for k, v in pnl_results.items():
	csvWriter.writerow([k, v])	


def serializeSimLog(simlog_df, run_id="test_run", info_str="test_out"):	
    dir_path = '/local/disk1/temp_eg/backtest/simlogs/' + run_id + '/'

    if not os.path.isdir(dir_path):
   	os.makedirs(dir_path)


    store_filename = 'simlog_' + info_str

    store_filename = dir_path + store_filename + '.h5'
    store = pd.HDFStore(store_filename)
    store['simlog_df'] = simlog_df
    store.close()

    print "serialized simlog: " + store_filename 


def readMarketDataForDate(instrument_root, event_date, event_datetime_obj):
    try:
        symbol = mkdt_utils.getSymbol(instrument_root, event_datetime_obj)
    except:
        symbol = symbols_bkp[event_date]
    df = mkdt_utils.getMarketDataFrameForTradingDate(event_date, instrument_root, symbol, "100ms")
    if isinstance(df.head(1).time[0], basestring):
        df.time = df.time.apply(mkdt_utils.str_to_dt)
    df.set_index(df['time'], inplace=True)
    
    return symbol, df


def loadEventDates(event_name, instrument_root, date_range_start=None, date_range_end=None, target_dates_lst=None, directory="./"):
    event_dates_times_dict = readEventDatesTimes(event_name, instrument_root, directory)
    event_dates_times_dict['2016-05-11'] = "10:30:00"
    dates_lst = [datetime.strptime(d, '%Y-%m-%d') for d in event_dates_times_dict]
    dates_lst.sort()
    sorted_dates_str_lst = [t.strftime('%Y-%m-%d') for t in dates_lst]  # all dates sorted

    if date_range_start and date_range_end and date_range_start < date_range_end:
        date_range_start_dt = datetime.strptime(date_range_start, '%Y-%m-%d')
        date_range_end_dt = datetime.strptime(date_range_end, '%Y-%m-%d')
        event_dates = [d.strftime('%Y-%m-%d') for d in dates_lst if d >= date_range_start_dt and d < date_range_end_dt]  # select dates sorted
    elif target_dates_lst:
        target_dates_dt_lst = [datetime.strptime(d, '%Y-%m-%d') for d in target_dates_lst]
        event_dates = [d.strftime('%Y-%m-%d') for d in dates_lst if d in target_dates_dt_lst]
    else:
	event_dates = None

    return event_dates, event_dates_times_dict


def loadMarketDataForDatesList(instrument_root, dates_lst, event_dates_times_dict):
    md_dict = {}

    for d in dates_lst:
	event_start_time = event_dates_times_dict[d]
	event_datetime_obj = datetime.strptime(d + " " + event_start_time, '%Y-%m-%d %H:%M:%S')
	md_dict[d] = readMarketDataForDate(instrument_root, d, event_datetime_obj)
	print "loaded market data for date: ", d	

    return md_dict    


def run(initial_target_pnl_per_contract, time_decay_coeff, reversal_percent_target, stop_loss_pnl_per_contract,
	event_name, instrument_root, event_dates, event_dates_times_dict, md_dict, cpu_process_id=-1):
#	event_name, instrument_root, date_range_start=None, date_range_end=None, target_dates_lst=None):
    print "starting.......cpu_process:", cpu_process_id
    import timeit
    results = []

    cur_time_dt = datetime.now()
    cur_time_str = cur_time_dt.strftime("%Y%m%d_%H%M%S")
    this_run_id = cur_time_str

    print event_dates

    for event_date in event_dates:
        print event_date

        event_start_time = event_dates_times_dict[event_date]

        event_datetime_obj = datetime.strptime(event_date + " " + event_start_time, '%Y-%m-%d %H:%M:%S')

	symbol, df = md_dict[event_date]

        start_dt = event_datetime_obj - timedelta(seconds=30)
        stop_dt = event_datetime_obj + timedelta(minutes=62)  #32   #22  #12	

        opposite_stacker_off_dt = event_datetime_obj + timedelta(seconds=30)  #<======= adjust down to 10s?
        entry_stop_time_dt = stop_dt - timedelta(seconds=60) #event_datetime_obj + timedelta(seconds=30)  #stop_dt - timedelta(seconds=60)  #event_datetime_obj + timedelta(seconds=30)
	time_decay_exit_start_dt = opposite_stacker_off_dt
	reversal_tracking_start_dt = event_datetime_obj + timedelta(seconds=5)
	stop_loss_tracking_start_dt = event_datetime_obj + timedelta(seconds=5)
	delayed_trade_start_dt = event_datetime_obj + timedelta(seconds=1)  # should be 1 s for directional
	trade_termination_dt = event_datetime_obj + timedelta(minutes=30)  #self.stop_dt - timedelta(seconds=pre_trading_stop_secs)

	simlog_df = pd.DataFrame(index = df['time'])
	simlog_df['time'] = df['time']
	simlog_df['target_exit_price'] = float('NaN')


        Sim = simulator6.Simulator(df, symbol, min_price_increment)
        Strat = Strategy(df, simlog_df, event_datetime_obj, start_dt, stop_dt, opposite_stacker_off_dt, entry_stop_time_dt, time_decay_exit_start_dt, reversal_tracking_start_dt, stop_loss_tracking_start_dt, delayed_trade_start_dt, trade_termination_dt, Sim, dollar_value_per_price_level, cpu_process_id)
        Sim.initStrategy(Strat)
	Strat.initial_target_pnl_per_contract = initial_target_pnl_per_contract
	Strat.time_decay_coeff = time_decay_coeff
	Strat.reversal_percent_target = reversal_percent_target
	Strat.stop_loss_pnl_per_contract = stop_loss_pnl_per_contract
	Strat.run_trade_only = True#False#True  # run will terminate after conclusion of trade (pos=0)
	Strat.delayed_trade = True
	Strat.delayed_trade_type = "ONE-SIDED-CURVE"#"DIRECTIONAL"#"ONE-SIDED-CURVE"
        Strat.start()

	
	trade_duration_s = (Strat.close_position_time_dt - Strat.start_dt).seconds

	simlog_serialize_info = event_name + '_' + instrument_root + '_' + event_date.replace('-', '')
#TEMP -- rewrite for parallelization	serializeSimLog(Strat.simlog_df, this_run_id, info_str=simlog_serialize_info)


        event_results = {'pnl': Strat._pnl, 'pos': Strat._pos, 'vol': Strat._vol}
        results.append({'event_date': event_date, 'event_results': event_results, 'best_potential_pnl': Strat.best_potential_pnl, 'max_abs_pos': Strat.max_abs_pos, 'best_potential_pnl_time': Strat.best_potential_pnl_time_dt.strftime("%Y-%m-%d %H:%M:%S"), 'worst_potential_pnl': Strat.worst_potential_pnl, 'worst_potential_pnl_time': Strat.worst_potential_pnl_time_dt.strftime("%Y-%m-%d %H:%M:%S"), 'close_position_time': Strat.close_position_time_dt.strftime("%Y-%m-%d %H:%M:%S"), 'trade_duration': trade_duration_s})


        print "FINISHED:", event_date
        print "  pos: ", Strat._pos
        print "  PnL: ", Strat._pnl
        print "  vol: ", Strat._vol
	print "  initial_target_pnl_per_contract: ", Strat.initial_target_pnl_per_contract
	print "  time_decay_coeff: ", Strat.time_decay_coeff
	print "  reversal_percent_target: ", Strat.reversal_percent_target
	print "  cpu_process_id: ", Strat.cpu_process_id
	print "===================="
	print


    pnl_results = {}
    per_contract_pnl_results = {}
    trade_durations = {}
    modified_pnl_results = {}

    print
    print "--------"
    print "RESULTS:"
    print "--------"

    print "event_date".ljust(16) + "PnL".ljust(20) + "pos".ljust(12) + "vol".ljust(12) + "max_abs_pos".ljust(16) + "best_potential_pnl".ljust(20) \
		+ "best_potential_pnl_time".ljust(30) + "worst_potential_pnl".ljust(20) + "worst_potential_pnl_time".ljust(30) + "close_position_time".ljust(30)
    for result in results:
        print result['event_date'].ljust(16) + str(result['event_results']['pnl']).ljust(20) + str(result['event_results']['pos']).ljust(12) \
                + str(result['event_results']['vol']).ljust(12) + str(result['max_abs_pos']).ljust(16) + str(result['best_potential_pnl']).ljust(20) \
		+ str(result['best_potential_pnl_time']).ljust(30) + str(result['worst_potential_pnl']).ljust(20) + str(result['worst_potential_pnl_time']).ljust(30) \
		+ str(result['close_position_time']).ljust(30)

        pnl_results[result['event_date']] = result['event_results']['pnl']
	if float(result['event_results']['vol']) != 0:
	    per_contract_pnl_results[result['event_date']] = result['event_results']['pnl'] / float(result['event_results']['vol'])
	else:
	    per_contract_pnl_results[result['event_date']] = 0
	trade_durations[result['event_date']] = result['trade_duration']
	modified_pnl_results[result['event_date']] = (pnl_results[result['event_date']] + result['best_potential_pnl']) / 2.0

    print pnl_results
    print
    print modified_pnl_results

    serialize_info = event_name + "_" + instrument_root
#TEMP -- rewrite for paralellization    serializePnLResults(pnl_results, incl_timestamp=True, info_str=serialize_info)


    return (pnl_results, per_contract_pnl_results, trade_durations, modified_pnl_results)



def main():
    #run("Crude", "CL", '2014-12-17', '2014-12-24')#'2015-11-25')
    #run("Crude", "CL", '2014-04-09', '2014-05-29')   #<++++++
    #run("Crude", "CL", '2014-04-30', '2014-05-07')
    #run("Crude", "CL", '2014-04-09', '2015-11-25')    #<------
    #run("Crude", "CL", '2014-05-29', '2014-06-04')
    #run("Crude", "CL", '2014-05-07', '2014-05-14')

    #run(20.0, -0.01, "Crude", "CL", target_dates_lst=['2014-05-07', '2014-05-14'])
    #run(1000.0, -0.00000001, "Crude", "CL", '2014-04-09', '2015-11-25')

    event_name = "Crude"
    instrument_root = "CL"

    #event_dates, event_dates_times_dict = loadEventDates(event_name, instrument_root, target_dates_lst=['2014-05-07', '2014-05-14'])
    #event_dates, event_dates_times_dict = loadEventDates(event_name, instrument_root, date_range_start='2014-04-09', date_range_end='2015-11-25')
    #event_dates, event_dates_times_dict = loadEventDates(event_name, instrument_root, target_dates_lst=['2014-04-09'])  
    #event_dates, event_dates_times_dict = loadEventDates(event_name, instrument_root, target_dates_lst=['2016-05-11'])
    #event_dates, event_dates_times_dict = loadEventDates(event_name, instrument_root, target_dates_lst=['2015-02-04'])
    event_dates, event_dates_times_dict = loadEventDates(event_name, instrument_root, date_range_start='2014-04-09', date_range_end='2016-04-20')
    #event_dates, event_dates_times_dict = loadEventDates(event_name, instrument_root, target_dates_lst=['2014-10-01'])
    #event_dates, event_dates_times_dict = loadEventDates(event_name, instrument_root, target_dates_lst=['2014-08-20'])
    #event_dates, event_dates_times_dict = loadEventDates(event_name, instrument_root, date_range_start='2016-04-06', date_range_end='2016-04-13')
    if not event_dates:
        print "ERROR with date range.  Simulation cannot run."
        return 0

    md_dict = loadMarketDataForDatesList(instrument_root, event_dates, event_dates_times_dict)
        
    best_pnl_per_contract_dict = readBestPnLperContract(event_name, instrument_root)

    

    #run_results_pnl, run_results_per_contract_pnl, trade_durations = run(100.0, -0.05, 0.3, "Crude", "CL", event_dates, event_dates_times_dict, md_dict)

    #run_results_pnl, run_results_per_contract_pnl, trade_durations = run(1000.0, -0.00001, 2.0, -1000, event_name, instrument_root, event_dates, event_dates_times_dict, md_dict)
    run_results_pnl, run_results_per_contract_pnl, trade_durations, modified_pnl_results = run(120.0, -0.01, 2.0, -1000, event_name, instrument_root, event_dates, event_dates_times_dict, md_dict)

    #run_results_pnl, run_results_per_contract_pnl, trade_durations, modified_pnl_results = run(250.0, -0.01, 0.3, -1000, event_name, instrument_root, event_dates, event_dates_times_dict, md_dict)

#    errors = [best_pnl_per_contract_dict[d] - run_results_per_contract_pnl[d] for d in event_dates]
#    squared_errors = [e * e for e in errors]
#    rmse = math.sqrt(sum(squared_errors) / float(len(squared_errors)))
#    print "rmse = ", rmse

    



if __name__ == "__main__": main()


