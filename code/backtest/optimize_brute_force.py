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
#import tech_anal

#import strat_sim_time_decay
#import strat_CL_time_decay
#reload(strat_CL_time_decay)
import strat_directional
reload(strat_directional)


""" Strategy Parameters """
initial_target_reversal_coeff = [1.0, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
initial_target_pnl_per_contract = [5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 200.0, 300.0, 400.0, 500.0]
time_decay_coeff = [-0.5, -0.1, -0.05, -0.01, -0.005, -0.001, -0.0005, -0.0001, -0.00005]

initial_target_pnl_per_contract = [300.0, 400.0, 500.0]
time_decay_coeff = [-0.001, -0.0005, -0.0001, -0.00005]

initial_target_pnl_per_contract = [5.0, 10.0, 20.0, 30.0]
time_decay_coeff = [-0.5, -0.1, -0.05, -0.01]

#initial_target_pnl_per_contract = [5.0, 10.0, 20.0, 30.0, 50.0, 75.0]
#time_decay_coeff = [-0.5, -0.1, -0.05, -0.01, -0.005, -0.001]

#initial_target_pnl_per_contract = [5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 200.0]
#time_decay_coeff = [-0.5, -0.1, -0.05, -0.01, -0.005, -0.001, -0.0005, -0.0001]

initial_target_pnl_per_contract = [10.0, 30.0, 50.0, 75.0, 100.0, 200.0, 300.0, 400.0, 500.0]
time_decay_coeff = [-0.5, -0.1, -0.05, -0.01, -0.005, -0.001, -0.0005, -0.0001, -0.00005]
reversal_percent_target = [0.1, 0.2, 0.3, 0.4]

initial_target_pnl_per_contract = [100.0]
time_decay_coeff = [-0.001]
reversal_percent_target = [0.6]

initial_target_pnl_per_contract = [100.0, 500.0]
time_decay_coeff = [-0.01, -0.005, -0.001, -0.0005, -0.0001, -0.00005]
reversal_percent_target = [0.2, 0.3, 0.4]

initial_target_pnl_per_contract = [1000.0]
time_decay_coeff = [-0.001]  # -0.00001
reversal_percent_target = [2.0]
stop_loss_pnl_per_contract = [-1000]



backtest_df = None
backtest_params = None

global_start_time = 0


def one_run(initial_target_pnl_per_contract, time_decay_coeff,
	event_name, instrument_root, date_range_start=None, date_range_end=None, target_dates_lst=None):
    pass


def optimize():
    pass


def load_market_date():
    pass


def load_event_dates():
    pass


def prepare_data():
    pass
    

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


def run_optimization(#initial_target_pnl_per_contract, time_decay_coeff, directory,
            directory, event_name, instrument_root, date_range_start=None, date_range_end=None, target_dates_lst=None):

    event_dates, event_dates_times_dict = strat_directional.loadEventDates(event_name, instrument_root, date_range_start, date_range_end, target_dates_lst, directory=directory)

    if not event_dates:
        print "ERROR with date range.  Simulation cannot run."
        exit

    md_dict = strat_directional.loadMarketDataForDatesList(instrument_root, event_dates, event_dates_times_dict)
    #md_dict = None

    best_pnl_per_contract_dict = strat_directional.readBestPnLperContract(event_name, instrument_root, directory)

    global backtest_df
    index = [i for i in range(0,len(initial_target_pnl_per_contract) * len(time_decay_coeff))]
    backtest_df = pd.DataFrame(index=index, columns=['initi_pnl_per_contract', 'decay_coeff', 'reversal_percent_target', 'stop_loss_pnl_per_contract', 'rmse', 'total_pnl', 'risk_ratio', 'avg_pnl_per_contract', 'avg_trade_duration', 'trade_duration_indic', 'process_id', 'latency', 'time_started', 'time_finished'])
    backtest_df = pd.DataFrame(columns=['initi_pnl_per_contract', 'decay_coeff', 'reversal_percent_target', 'stop_loss_pnl_per_contract', 'rmse', 'total_pnl', 'risk_ratio', 'avg_pnl_per_contract', 'avg_trade_duration', 'trade_duration_indic', 'process_id', 'latency', 'time_started', 'time_finished'])

    param_lst = [initial_target_pnl_per_contract, time_decay_coeff, reversal_percent_target, stop_loss_pnl_per_contract]
    optim_param_combinations = list(itertools.product(*param_lst))

    global backtest_params
    backtest_params = {'event_name': event_name, 'instrument_root': instrument_root, 'event_dates': event_dates, 'event_dates_times_dict': event_dates_times_dict, 'md_dict': md_dict, 'best_pnl_per_contract_dict': best_pnl_per_contract_dict}

    mp_handler(optim_param_combinations)

#    for t in initial_target_pnl_per_contract:
#        for c in time_decay_coeff:
#            print "BACKTEST: ", t, " ", c
#            run_results_pnl, run_results_per_contract_pnl = strat_CL_time_decay.run(t, c, "Crude", "CL", event_dates, event_dates_times_dict, md_dict)
#
#            errors = [best_pnl_per_contract_dict[d] - run_results_per_contract_pnl[d] for d in event_dates]
#            squared_errors = [e * e for e in errors]
#            rmse = math.sqrt(sum(squared_errors) / float(len(squared_errors)))
#            print "rmse = ", rmse
#
#            backtest_df = backtest_df.append({'initi_pnl_per_contract': t, 'decay_coeff': c, 'rmse': rmse}, ignore_index=True)


    return backtest_df



#    params_lst = [initial_target_pnl_per_contract, time_decay_coeff]
#    param_combinations = list(itertoolsd.product(*param_lst))



def mp_worker((A, c, r, sl)):
    global global_start_time
    t00 = timeit.default_timer()
    global backtest_params
    my_process_id = multiprocessing.current_process()._identity

    print "BACKTEST: ", A, " ", c, " ", r, " ", sl, "    ", "Process: ", my_process_id
    run_results_pnl, run_results_per_contract_pnl, trade_durations = strat_directional.run(A, c, r, sl, backtest_params['event_name'], backtest_params['instrument_root'], backtest_params['event_dates'], backtest_params['event_dates_times_dict'], backtest_params['md_dict'], my_process_id)

    errors = [backtest_params['best_pnl_per_contract_dict'][d] - run_results_per_contract_pnl[d] for d in backtest_params['event_dates']]
    squared_errors = [e * e for e in errors]
    rmse = math.sqrt(sum(squared_errors) / float(len(squared_errors)))
    print "rmse = ", rmse, A, c, r, sl
    total_pnl = sum(run_results_pnl.values())
    risk_ratio = compute_risk_ratio(run_results_pnl)
    avg_pnl_per_contract = np.mean(run_results_per_contract_pnl.values())
    avg_trade_duration = np.mean(trade_durations.values())
    trade_duration_indic = 1.0 / np.sqrt(avg_trade_duration)

#    lock.acquire()
#    global backtest_df
#    backtest_df = backtest_df.append({'initi_pnl_per_contract': A, 'decay_coeff': c, 'rmse': rmse}, ignore_index=True)
#    #print backtest_df
#    lock.release()

    t01 = timeit.default_timer()

    #rmse = A - c
    results = {'initi_pnl_per_contract': A, 'decay_coeff': c, 'reversal_percent_target': r, 'stop_loss_pnl_per_contract': sl, 'rmse': "{0:.1f}".format(rmse), 'total_pnl': "{0:.1f}".format(total_pnl), 'risk_ratio': "{0:.1f}".format(risk_ratio), 'avg_pnl_per_contract': avg_pnl_per_contract, 'avg_trade_duration': avg_trade_duration, 'trade_duration_indic': trade_duration_indic, 'process_id': my_process_id, 'latency': "{0:.3f}".format(t01-t00), 'time_started': "{0:.3f}".format(t00-global_start_time), 'time_finished': "{0:.3f}".format(t01-global_start_time)}

    print my_process_id, ":", t00-global_start_time, t01-global_start_time, t01-t00


    return results


def init(l):
    global lock
    lock = l


def mp_handler(optim_param_combinations):
    t0 = timeit.default_timer()
    global global_start_time
    global_start_time = t0
    l = multiprocessing.Lock()
    try:
        p = multiprocessing.Pool(16, initializer=init, initargs=(l,))
    	all_results = p.map(mp_worker, optim_param_combinations)
    	p.close()
    	p.terminate()
    	p.join()
    except KeyboardInterrupt:
	print 'parent received control-c'
	p.terminate()
    t1 = timeit.default_timer()

    global backtest_df

    for i_results in all_results:
	backtest_df = backtest_df.append(i_results, ignore_index=True)
    t2 = timeit.default_timer()

#    print backtest_df

    print "Timing -------------------------"
    print "t1 - t0 = ", t1 - t0
    print "t2 - t2 = ", t2 - t1
    print "--------------------------------"


def usage():
    print "Usage: optimize_brute_force.py -e [event_name] -s [symbol_root] -d [(date_1, date_2, date_3...)|(date_1:date_n)] "


def main():
    #initial_target_pnl_per_contract = [5.0, 10.0]
    #time_decay_coeff = [-0.5, -0.1]

    

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
        backtest_df = run_optimization(#initial_target_pnl_per_contract, time_decay_coeff, directory, 
                directory, event_name, symbol_root, date_range_start=date_range_start, date_range_end=date_range_end)

    else:
	dates_lst = [x.strip() for x in dates_str.split(',')]
	if dates_lst[0]:
	    #run_result = strat_sim_time_decay.run(event_name, symbol_root, target_dates_lst=dates_lst)
            backtest_df = run_optimization(#initial_target_pnl_per_contract, time_decay_coeff, directory, 
		    directory, event_name, symbol_root, date_range_start=None, date_range_end=None, target_dates_lst=dates_lst)
            
	else:
	    print "ERROR: incorrectly specified event dates"
	    usage()
            sys.exit()
    
        
    print_full(backtest_df)        




if __name__ == "__main__": main()
