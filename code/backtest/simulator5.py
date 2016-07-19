



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

    resting_orders = {}

    def __init__(self, df, symbol, min_price_increment):
        self.df = df
	self.symbol = symbol
	self.min_price_increment = min_price_increment

        self.live_orders = []

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

        return self.start_loc


    def processLiveOrders(self):
	cur_top_ask_price = self.top_prices_npar[self.cur_loc,0]
	cur_top_bid_price = self.top_prices_npar[self.cur_loc,1]
	fills_lst = []

        remaining_resting_orders = []

        status = None

#	for order in self.live_orders:
#            id = order['id']
#            direction = order['direction']
#            price = order['price']
#            size = order['size']
#            loc = order['order_time_loc']

#            if direction == "buy":
#                partial_fills, avg_filled_price, filled_size = self.computePassiveBuyFill(self.df, loc, size, price, self.ask_size_npar, self.bid_size_npar, self.top_prices_npar)
#                #fills_lst

#            elif direction == "sell":
#                partial_fills, avg_filled_price, filled_size = self.computePassiveSellFill(self.df, loc, size, price, self.ask_size_npar, self.bid_size_npar, self.top_prices_npar)
                #fills_lst

#            if abs(filled_size) < size:
#                remaining_resting_orders.append(order)
#                status = "open"
#            else:
#                status = "completed"

#            fill_report = {'id': id, 'partial_fills': partial_fills, 'avg_filled_price': avg_filled_price, 'filled_size': filled_size, 'status': status}
#            if abs(filled_size) > 0:  # send report only if something changed with this resting order
#                fills_lst.append(fill_report)

#	self.live_orders = remaining_resting_orders



	active_buy_orders = [o for o in self.live_orders if o['direction'] == "buy" and o['price'] > cur_top_ask_price]
	active_sell_orders = [o for o in self.live_orders if o['direction'] == "sell" and o['price'] < cur_top_bid_price]
	nonactive_orders = [o for o in self.live_orders if o['direction'] == "buy" and o['price'] <= cur_top_ask_price \
							    or o['direction'] == "sell" and o['price'] >= cur_top_bid_price]


	remaining_resting_orders = nonactive_orders 


	fill_info_lst = []

	for order in active_buy_orders:
	    id = order['id']
            direction = order['direction']
            price = order['price']
            size = order['size']
            loc = order['order_time_loc']	

	    partial_fills, avg_filled_price, filled_size = self.computePassiveBuyFill(self.df, loc, size, price, self.ask_size_npar, self.bid_size_npar, self.top_prices_npar)
	    fill_info_lst.append((partial_fills, avg_filled_price, filled_size))

	for order in active_sell_orders:
	    id = order['id']
            direction = order['direction']
            price = order['price']
            size = order['size']
            loc = order['order_time_loc']

	    partial_fills, avg_filled_price, filled_size = self.computePassiveSellFill(self.df, loc, size, price, self.ask_size_npar, self.bid_size_npar, self.top_prices_npar)
	    fill_info_lst.append((partial_fills, avg_filled_price, filled_size))

	for item in fill_info_lst:
	    (partial_fills, avg_filled_price, filled_size) = item
	    
	    if abs(filled_size) < size:
                remaining_resting_orders.append(order)
                status = "open"
            else:
                status = "completed"

            fill_report = {'id': id, 'partial_fills': partial_fills, 'avg_filled_price': avg_filled_price, 'filled_size': filled_size, 'status': status}
            if abs(filled_size) > 0:  # send report only if something changed with this resting order
                fills_lst.append(fill_report)

	self.live_orders = remaining_resting_orders


	return fills_lst


    def processNewOrders(self, new_order_lst):
	fills_lst = []

	status = None

	for order in new_order_lst:
            id = order['id']
            #type = order['type']
            #intent = order['intent']
            direction = order['direction']
            price = order['price']
            size = order['size']
            #ltc = order['levels_to_cross']
            loc = order['order_time_loc']

            top_bid_price_now = self.top_prices_npar[self.cur_loc,0]
            top_ask_price_now = self.top_prices_npar[self.cur_loc,1]

            partial_fills, avg_filled_price, filled_size = ([], 0, 0)  #just a dummy

            if ((direction == "buy" and price < top_ask_price_now) or (direction == "sell" and price > top_bid_price_now)):
                self.live_orders.append(order)
                status = "open"
            elif (direction == "buy" and price >= top_ask_price_now):
                levels_to_cross = (price - top_ask_price_now) / float(self.min_price_increment) + 1
                partial_fills, avg_filled_price, filled_size = self.computeAggressiveBuyFill(self.df, loc, size, levels_to_cross, self.ask_size_npar, self.bid_size_npar, self.top_prices_npar)
                #fills_lst

                unfilled_size = size - abs(filled_size)
                if unfilled_size > 0:
                    self.live_orders.append({'id': id, 'direction': direction, 'price': price, 'size': unfilled_size, 'order_time_loc': loc})
                    status = "open"
                else:
                    status = "completed"
            elif (direction == "sell" and price <= top_bid_price_now):
                levels_to_cross = (top_bid_price_now - price) / float(self.min_price_increment) + 1
                partial_fills, avg_filled_price, filled_size = self.computeAggressiveSellFill(self.df, loc, size, levels_to_cross, self.ask_size_npar, self.bid_size_npar, self.top_prices_npar)
                #fills_lst

                unfilled_size = size - abs(filled_size)
                if unfilled_size > 0:
                    self.live_orders.append({'id': id, 'direction': direction, 'price': price, 'size': unfilled_size, 'order_time_loc': loc})
                    status = "open"
                else:
                    status = "completed"
            elif direction == "cancel":
                #print "LIVE ORDERS:"
                #print self.live_orders
                #ord_ids = (o.id for o in self.live_orders)
                #print "ORD IDs:"
                #print ord_ids
                #print "TO CANCEL id:", id
                order_id, ok_cancel = self.cancelOrder(self.df, loc, id)
                if ok_cancel:
                    status = "cancelled"
                else:
                    status = "not_cancelled"
	

            fill_report = {'id': id, 'partial_fills': partial_fills, 'avg_filled_price': avg_filled_price, 'filled_size': filled_size, 'status': status}
            fills_lst.append(fill_report)

	#if new_order_lst:
	#    self.live_orders.sort(key=lambda x: x['price'], reverse=True)
	
	return fills_lst

    
    def execute(self, new_order_lst):
	fills_lst = []

	fills_lst.extend(self.processNewOrders(new_order_lst))
	fills_lst.extend(self.processLiveOrders())
	#fills_lst.extend(self.processNewOrders(new_order_lst))

        self.cur_loc = self.last_loc + 1
        self.last_loc = self.cur_loc

	#print self.df.ix[self.cur_loc, 'time'], "Sim::self.live_orders: ", self.live_orders

        return (self.cur_loc + 1, fills_lst, self.live_orders)

    
    def cancelOrder(self, df, loc, order_id):
	live_order_ids = [o['id'] for o in self.live_orders]
        if order_id in live_order_ids:
	    remove_order = next((x for x in self.live_orders if x['id'] == order_id), None)
	    self.live_orders.remove(remove_order)
            #del self.live_orders[order_id]
            return (order_id, True)
        return (order_id, False)


    def computePassiveBuyFill(self, df, loc, target_size, price, ask_size_npar, bid_size_npar, top_prices_npar):
	top_ask_price_now = top_prices_npar[self.cur_loc,0]

	fills_lst = []

	filled_size = 0
	avg_filled_price = 0

	if top_ask_price_now < price:
	    filled_size = target_size
	    fill_price = price
	    fills_lst.append({'size': filled_size, 'price': price, 'loc': loc})
	    avg_filled_price = price

	return (fills_lst, avg_filled_price, filled_size)


    def computePassiveSellFill(self, df, loc, target_size, price, ask_size_npar, bid_size_npar, top_prices_npar): 
	top_bid_price_now = top_prices_npar[self.cur_loc,1]

 	fills_lst = []

        filled_size = 0
        avg_filled_price = 0

        if top_bid_price_now > price:
	    filled_size = target_size
	    fill_price = price
            fills_lst.append({'size': -1*filled_size, 'price': price, 'loc': loc})
            avg_filled_price = price

        return (fills_lst, avg_filled_price, -1*filled_size)


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

	if filled_size == 0:
	    avg_filled_price = 0
	else:
            avg_filled_price = size_weighted_price / float(filled_size)  # need check for zero denominator

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

	if filled_size == 0:
	    avg_filled_price = 0
	else:
            avg_filled_price = size_weighted_price / float(filled_size)  # need check for zero denominator

        return (fills_lst, avg_filled_price, -1*filled_size)
