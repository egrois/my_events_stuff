import sys
from datetime import datetime,time, timedelta
import pandas as pd
import os
sys.path.append('/home/egrois/git/code/pybt')
#import pybt
import utils


def str_to_dt(tt):
    return datetime.strptime(tt, "%Y-%m-%d %H:%M:%S.%f")


def getSymbol(instrument_root, tradingday_datetime_obj):
    #print "trying to get symbol...", instrument_root, tradingday_datetime_obj
    if tradingday_datetime_obj >= datetime(2013, 4, 1):
        tuples = utils.sym_vols(tradingday_datetime_obj)
	#print tuples
        symbols_dict = dict([ (k, v) for k, v, w in tuples ])
        #print symbols_dict
	#s = [k for  k,v in symbols_dict.iteritems() if len(k) == 4]
	#print "s:    ", s
        instr_symbols_dict = dict((k, v) for k,v in symbols_dict.iteritems() if k.startswith(instrument_root))
	#print instr_symbols_dict
        symbol = max(instr_symbols_dict.keys(),  key=(lambda k: instr_symbols_dict[k]))
        #print symbol
    else:
        #print "old date: ", tradingday_datetime_obj.strftime('%Y-%m-%d')
        symbol = getSymbolOldDate(instrument_root, tradingday_datetime_obj)

    return symbol


def getSymbolOldDate(instrument_root, tradingday_datetime_obj):
    date_str = tradingday_datetime_obj.strftime('%Y-%m-%d')
    search_path = '/local/disk1/data/snapshots/0/' + instrument_root + '/'
    #print search_path
    #print os.listdir(search_path)

    files = [i for i in os.listdir(search_path) if os.path.isfile(os.path.join(search_path,i)) and \
         date_str.replace('-', '') in i]
    #print files
    
    instr_symbols_dict = {}
    for f in files:
        symbol = f.split('_')[0]
        store_filename = os.path.join(search_path,f)
        store = pd.HDFStore(store_filename)
        df = store['df']
        store.close()
        volume = df['abs_vol'].sum()
        instr_symbols_dict[symbol] = volume
        #print volume

    symbol = max(instr_symbols_dict.keys(),  key=(lambda k: instr_symbols_dict[k]))
    #print symbol

    return symbol


def getMarketDataFrameForTradingDate(trading_date, instrument_root, symbol, sampling_interval):
    if sampling_interval == "100ms":
        tg = "0"
    elif sampling_interval == "1s":
        tg == "1"
    elif sampling_interval == "5s":
        tg = "2"

    #data_dir_path = '/local/disk1/data2/snapshots/'
    data_dir_path = '/local/disk1/data/snapshots/'
    #data_dir_path = '/local/disk1/temp/' 
    #data_dir_path = '/local/disk1/temp2/'

    store_filename = data_dir_path + tg + '/' + instrument_root + '/' + symbol + '_' + tg + '_' + trading_date.replace('-', '') + '.h5'
    #store_filename = data_dir_path + symbol + '_' + tg + '_' + trading_date.replace('-', '') + '.h5'
    #store_filename = data_dir_path + symbol + '_' + tg + '_' + trading_date.replace('-', '') + '.h5'


    store = pd.HDFStore(store_filename)
    df = store['df']
    store.close()
    return df



def main():
	getSymbol("ES", datetime(2015, 4, 23))



if __name__ == "__main__":
    main()
