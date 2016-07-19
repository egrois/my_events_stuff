import sys
import os
import getopt
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
import itertools
import multiprocessing
import timeit

import mkdt_utils

import strat_directional
reload(strat_directional)



backtest_config = [120.0, -0.01, 2.0, -1000]

backtest_df = None
backtest_params = None

global_start_time = 0



def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


def compute_risk_ratio(run_results_pnl):
    risk_ratio1 = 16.0 * np.mean(run_results_pnl.values()) / np.std(run_results_pnl.values())
    risk_ratio2 = 16.0 * np.median(run_results_pnl.values()) / np.std(run_results_pnl.values())

    risk_ratio = risk_ratio1
    if risk_ratio2 < risk_ratio1:
        risk_ratio = risk_ratio2

    return risk_ratio


def run_parallel_backtest(directory, event_name, instrument_root, date_range_start=None, date_range_end=None, target_dates_lst=None):

    event_dates, event_dates_times_dict = strat_directional.loadEventDates(event_name, instrument_root, date_range_start, date_range_end, target_dates_lst, directory=directory)

    if not event_dates:
        print "ERROR with date range.  Simulation cannot run."
        exit

    md_dict = strat_directional.loadMarketDataForDatesList(instrument_root, event_dates, event_dates_times_dict)

    best_pnl_per_contract_dict = strat_directional.readBestPnLperContract(event_name, instrument_root, directory)

#    global backtest_df
#    index = [i for i in range(0,len(initial_target_pnl_per_contract) * len(time_decay_coeff))]
#    backtest_df = pd.DataFrame(index=index, columns=['initi_pnl_per_contract', 'decay_coeff', 'reversal_percent_target', 'stop_loss_pnl_per_contract', 'rmse', 'total_pnl', 'risk_ratio', 'avg_pnl_per_contract', 'avg_trade_duration', 'trade_duration_indic', 'process_id', 'latency', 'time_started', 'time_finished'])
#    backtest_df = pd.DataFrame(columns=['initi_pnl_per_contract', 'decay_coeff', 'reversal_percent_target', 'stop_loss_pnl_per_contract', 'rmse', 'total_pnl', 'risk_ratio', 'avg_pnl_per_contract', 'avg_trade_duration', 'trade_duration_indic', 'process_id', 'latency', 'time_started', 'time_finished'])

    #param_lst = [initial_target_pnl_per_contract, time_decay_coeff, reversal_percent_target, stop_loss_pnl_per_contract]
    #optim_param_combinations = list(itertools.product(*param_lst))

    global backtest_params
    backtest_params = {'event_name': event_name, 'instrument_root': instrument_root, 'event_dates': event_dates, 'event_dates_times_dict': event_dates_times_dict, 'md_dict': md_dict, 'best_pnl_per_contract_dict': best_pnl_per_contract_dict}

    event_dates = backtest_params['event_dates']
    mp_handler(event_dates)


    return backtest_df


def mp_worker(event_date):
    event_dates_lst = []
    event_dates_lst.append(event_date)
    global global_start_time
    t00 = timeit.default_timer()
    global backtest_params
    my_process_id = multiprocessing.current_process()._identity

    print "BACKTEST: ", event_date, " ", "Process: ", my_process_id

    global backtest_config
    A, c, r, sl = backtest_config

    run_results_pnl, run_results_per_contract_pnl, trade_durations, modified_pnl_results = strat_directional.run(A, c, r, sl, backtest_params['event_name'], backtest_params['instrument_root'], event_dates_lst, backtest_params['event_dates_times_dict'], backtest_params['md_dict'], my_process_id)

    ######Here
    t01 = timeit.default_timer()

    print my_process_id, ":", t00-global_start_time, t01-global_start_time, t01-t00


    return run_results_pnl, run_results_per_contract_pnl, trade_durations, modified_pnl_results


def init(l):
    global lock
    lock = l


def mp_handler(event_dates):
    t0 = timeit.default_timer()
    global global_start_time
    global_start_time = t0
    l = multiprocessing.Lock()
    #print "@@@@@@@@@@@@@@@@@@"
    #print event_dates
    #print "@@@@@@@@@@@@@@@@@@"
    try:
        p = multiprocessing.Pool(16, initializer=init, initargs=(l,))
	global backtest_params
	#print event_dates
  	all_results = p.map(mp_worker, event_dates)#backtest_params['event_dates'])
	p.close()
        p.terminate()
        p.join()
    except KeyboardInterrupt:
        print 'parent received control-c'
        p.terminate()
    t1 = timeit.default_timer()

    ###Here
    pnl_results_complete = {}
    per_contract_pnl_results_complete = {}
    trade_durations_complete = {}
    modified_pnl_results_complete = {}
    for (i_results_pnl, i_results_pnl_per_contract, i_results_trade_durations, i_results_modified_pnl) in all_results:
	pnl_results_complete.update(i_results_pnl)
	per_contract_pnl_results_complete.update(i_results_pnl_per_contract)
	trade_durations_complete.update(i_results_trade_durations)
	modified_pnl_results_complete.update(i_results_modified_pnl)
    t2 = timeit.default_timer()

    print pnl_results_complete
    print modified_pnl_results_complete

    print "Timing -------------------------"
    print "t1 - t0 = ", t1 - t0
    print "t2 - t1 = ", t2 - t1
    print "--------------------------------"






def usage():
    print "Usage: optimize_brute_force.py -e [event_name] -s [symbol_root] -d [(date_1, date_2, date_3...)|(date_1:date_n)] "


def main():
    directory = '/home/egrois/git/code/backtest' + '/'


    try:
        opts, args = getopt.getopt(sys.argv[1:], "he:s:d:", ["help", "event_name=", "symbol_root=", "dates="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)
        usage()
        sys.exit(2)

    event_name = None
    symbol_root = None
    dates_str = None
    for o, a in opts:
        if o == "-e":
            event_name = a
        elif o == "-s":
            symbol_root = a
        elif o == "-d":
            dates_str = a
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        else:
            assert False, "unhandled option"

    if not event_name:
        print "ERROR: missing event name"
        usage()
        sys.exit()
    elif not symbol_root:
        print "ERROR: missing symbol root"
        usage()
        sys.exit()
    elif not dates_str:
        print "ERROR: missing dates"
        usage()
        sys.exit()

    if ':' in dates_str:
        items = dates_str.split(':')
        date_range_start = items[0]
        date_range_end = items[1]

        #run_result = strat_sim_time_decay.run(event_name, symbol_root, date_range_start, date_range_end)
        backtest_df = run_parallel_backtest(#initial_target_pnl_per_contract, time_decay_coeff, directory,
                directory, event_name, symbol_root, date_range_start=date_range_start, date_range_end=date_range_end)

    else:
        dates_lst = [x.strip() for x in dates_str.split(',')]
        if dates_lst[0]:
            #run_result = strat_sim_time_decay.run(event_name, symbol_root, target_dates_lst=dates_lst)
            backtest_df = run_parallel_backtest(#initial_target_pnl_per_contract, time_decay_coeff, directory,
                    directory, event_name, symbol_root, date_range_start=None, date_range_end=None, target_dates_lst=dates_lst)

        else:
            print "ERROR: incorrectly specified event dates"
            usage()
            sys.exit()


    #print_full(backtest_df)




if __name__ == "__main__": main()

