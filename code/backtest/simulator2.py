



import numpy as np
from datetime import datetime,time, timedelta
import timeit
import pandas as pd











class Simulator:
    min_price_increment = 1  # need to have a lookup structure for these
    book_depth = 5

    df = None
    start_dt = None
    start_loc = 0
    max_entry_dt = None
    max_entry_loc = 0
    stop_dt = None
    stop_loc = 0
    cur_loc = 0
    last_loc = 0
    strategy = None

    ask_size_npar = None
    bid_size_npar = None
    top_prices_npar = None
    time_at_loc_npar = None

    def __init__(self, df):
        self.df = df

	self.ask_size_npar = df[['ask_size_0', 'ask_size_1', 'ask_size_2', 'ask_size_3', 'ask_size_4']].values
        self.bid_size_npar = df[['bid_size_0', 'bid_size_1', 'bid_size_2', 'bid_size_3', 'bid_size_4']].values
        self.top_prices_npar = df[['top_ask_price', 'top_bid_price']].values
        self.time_at_loc_npar = df['time'].values
        print self.time_at_loc_npar[0], self.time_at_loc_npar[1]


    def initStrategy(self, strategy):
        self.strategy = strategy


    def start_sim(self, start_dt):
        self.start_dt = start_dt
        self.start_loc = self.df.index.get_loc(start_dt)
	self.cur_loc = self.start_loc
	self.last_loc = self.cur_loc
	#self.max_entry_dt = max_entry_dt
	#self.max_entry_loc = df.index.get_loc(max_entry_dt)
	#self.stop_dt = stop_dt
	#self.stop_loc = df.index.get_loc(stop_dt)
	
        #self.strategy.update(self.start_loc, [])  # send strategy first market data loc and empy list of fills
#	print "StartSim:", self.cur_loc, self.last_loc
	return self.start_loc
    
    
    def execute(self, order_lst):
        fills_lst = []        

	for order in order_lst:
	    type = order['type']
	    direction = order['direction']
	    size = order['size']
	    ltc = order['levels_to_cross']
	    loc = order['order_time_loc']

	    if type == "entry":
		partial_fills, avg_filled_price, filled_size = self.executeEntry(direction, ltc, size, self.cur_loc)	    
	    elif type == "exit":
		partial_fills, avg_filled_price, filled_size = self.executeExit(direction, ltc, size, self.cur_loc)

	    #print fills_lst 
	    #print avg_filled_price
	    #print filled_size
	    #print

	    fill_report = {'partial_fills': partial_fills, 'avg_filled_price': avg_filled_price, 'filled_size': filled_size}
            fills_lst.append(fill_report)

	self.cur_loc = self.last_loc + 1
	self.last_loc = self.cur_loc

        # execute orders

	#self.strategy.update(self.cur_loc + 1, fills_lst)
	return (self.cur_loc + 1, fills_lst)


    def computeAggressiveBuyFill(self, df, loc, target_size, levels_to_cross, ask_size_npar, bid_size_npar, top_prices_npar):
	top_ask_price_now = top_prices_npar[loc,0]
	buy_price_now = [top_ask_price_now + self.min_price_increment * m for m in range(0,self.book_depth)]
	buy_size_now = ask_size_npar[loc].tolist()

	buy_size_next = [0]*self.book_depth
	top_ask_price_next = top_prices_npar[loc+1,0]

	market_move = int((top_ask_price_next - top_ask_price_now) / float(self.min_price_increment))
	if market_move == 0:
	    buy_size_next = ask_size_npar[loc+1].tolist()
	elif (market_move > 0) and (market_move < self.book_depth):
            buy_size_next[market_move:self.book_depth] = buy_size_now[0:self.book_depth-market_move]
        elif market_move < 0:
            buy_size_next[:] = buy_size_now[:]

	buy_size = [min(buy_size_now[k], buy_size_next[k]) for k in range(0,self.book_depth)]

	fills_lst = []

	execute_size = target_size
        level = 0
        size_weighted_price = 0
        filled_size = 0
        while execute_size > 0 and level < min(levels_to_cross, self.book_depth):
	    if execute_size <= buy_size[level]:
                partial_filled_size = execute_size
            elif execute_size > buy_size[level]:
                partial_filled_size = buy_size[level]

            filled_size += partial_filled_size
            size_weighted_price += buy_price_now[level] * partial_filled_size
            execute_size -= partial_filled_size

	    fills_lst.append({'size': partial_filled_size, 'price': buy_price_now[level], 'loc': loc})

            level += 1

	avg_filled_price = size_weighted_price / float(filled_size)

	return (fills_lst, avg_filled_price, filled_size)


    def computeAggressiveSellFill(self, df, loc, target_size, levels_to_cross, ask_size_npar, bid_size_npar, top_prices_npar):
	top_bid_price_now = top_prices_npar[loc, 1]
	sell_price_now = [top_bid_price_now - self.min_price_increment * m for m in range(0,self.book_depth)]
	sell_size_now = bid_size_npar[loc].tolist()

	sell_size_next = [0]*self.book_depth
	top_bid_price_next = top_prices_npar[loc+1,1]

	market_move = int((top_bid_price_next - top_bid_price_now) / float(self.min_price_increment))
    	if market_move == 0:	
	    sell_size_next = bid_size_npar[loc+1].tolist()
	elif (market_move < 0) and (abs(market_move) < self.book_depth):
            sell_size_next[abs(market_move):self.book_depth] = sell_size_now[0:self.book_depth-abs(market_move)]
    	elif market_move > 0:
            sell_size_next[:] = sell_size_now[:]

	sell_size = [min(sell_size_now[k], sell_size_next[k]) for k in range(0,self.book_depth)]	

	fills_lst = []

	execute_size = target_size
        level = 0
        size_weighted_price = 0
        filled_size = 0
	while execute_size > 0 and level < min(levels_to_cross, self.book_depth):
            if execute_size <= sell_size[level]:
                partial_filled_size = execute_size
            elif execute_size > sell_size[level]:
                partial_filled_size = sell_size[level]

	    filled_size += partial_filled_size
            size_weighted_price += sell_price_now[level] * partial_filled_size
            execute_size -= partial_filled_size		

	    fills_lst.append({'size': -1*partial_filled_size, 'price': sell_price_now[level], 'loc': loc})

            level += 1

	avg_filled_price = size_weighted_price / float(filled_size)

    	return (fills_lst, avg_filled_price, -1*filled_size)


    def computeRadicalExitFill(self, df, loc, target_size, direction, top_prices_npar):
	if direction == "buy":
	    top_ask_price_now = top_prices_npar[loc, 0]
            avg_filled_price = top_ask_price_now + (self.book_depth + 2) * self.min_price_increment
	    filled_size = target_size
	elif direction == "sell":
	    top_bid_price_now = top_prices_npar[loc, 1]
            avg_filled_price = top_bid_price_now - (self.book_depth + 2) * self.min_price_increment
	    filled_size = -1 * target_size

	fills_lst = []
	fills_lst.append({'size': target_size, 'price': avg_filled_price, 'loc': loc})

	return (fills_lst, avg_filled_price, filled_size)


    def executeEntry(self, direction, levels_to_cross, size, loc):
	fills_lst = []

	if direction == "buy":
	    #top_ask_price_now = self.top_prices_npar[loc,0]
	    #levels_to_cross = 1 + (price - top_ask_price_now) / float(self.min_price_increment)
	    fills_lst, avg_filled_price, filled_size = self.computeAggressiveBuyFill(self.df, loc, size, levels_to_cross, self.ask_size_npar, self.bid_size_npar, self.top_prices_npar)
	elif direction == "sell":
	    #top_bid_price_now = top_prices_npar[loc, 1]
	    #levels_to_cross = 1 + (price - top_bid_price_now) / float(self.min_price_increment)
	    fills_lst, avg_filled_price, filled_size = self.computeAggressiveSellFill(self.df, loc, size, levels_to_cross, self.ask_size_npar, self.bid_size_npar, self.top_prices_npar)

	return (fills_lst, avg_filled_price, filled_size)


    def executeExit(self, direction, levels_to_cross, target_size, loc):
	fills_lst = []
	avg_filled_price = 0

	filled_size = 0
	if direction == "sell":
	    #print "trying to sell exit size:", target_size
	    while filled_size < target_size and levels_to_cross <= self.book_depth:
		fills_lst, avg_filled_price, filled_size = self.computeAggressiveSellFill(self.df, loc, target_size, levels_to_cross, self.ask_size_npar, self.bid_size_npar, self.top_prices_npar)
		levels_to_cross += 1
		#print "  filled_size:", filled_size
	    remaining_size = target_size - abs(filled_size)  # open position
	    if remaining_size > 0:
                print "PROBLEM: could not exit entire position.  Resorting to radical fill."
                fills_lst, avg_filled_price, filled_size = self.computeRadicalExitFill(self.df, loc, remaining_size, "sell", self.top_prices_npar)
	elif direction == "buy":
	    while filled_size < target_size and levels_to_cross <= self.book_depth:
		fills_lst, avg_filled_price, filled_size = self.computeAggressiveBuyFill(self.df, loc, target_size, levels_to_cross, self.ask_size_npar, self.bid_size_npar, self.top_prices_npar)
		levels_to_cross += 1
	    remaining_size = target_size - filled_size  # open position
	    if remaining_size > 0:
                print "PROBLEM: could not exit entire position.  Resorting to radical fill."
                fills_lst, avg_filled_price, filled_size = self.computeRadicalExitFill(self.df, loc, remaining_size, "buy", self.top_prices_npar)

	return (fills_lst, avg_filled_price, filled_size)







