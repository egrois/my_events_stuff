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








def main():
    import timeit

    for event_date in event_dates:
        print event_date
        



if __name__ == "__main__": main()
