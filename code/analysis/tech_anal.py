""" See https://www.quantopian.com/posts/technical-analysis-indicators-without-talib-code
or https://github.com/panpanpandas/ultrafinance/blob/master/ultrafinance/pyTaLib/pandasImpl.py
for implementations of various technical analysis signals.  This library is based on those. """

import numpy as np 
import pandas as pd  
import math as m
from datetime import *


def MACD(df, col, n_fast, n_slow, n_signal):  
    EMAfast = pd.Series(pd.ewma(df[col], span = n_fast, min_periods = n_slow - 1))  
    EMAslow = pd.Series(pd.ewma(df[col], span = n_slow, min_periods = n_slow - 1))  
    MACD = pd.Series(EMAfast - EMAslow, name = 'MACD')#'MACD_' + str(n_fast) + '_' + str(n_slow))  
    MACDsign = pd.Series(pd.ewma(MACD, span = n_signal, min_periods = n_signal - 1), name = 'MACDsign')#'MACDsign_' + str(n_signal))  
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACDdiff')#'MACDdiff_' + str(n_fast) + '_' + str(n_slow))  
    df = df.join(MACD)  
    df = df.join(MACDsign)  
    df = df.join(MACDdiff)  
    return df


def RSI(df, n):
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 < 50:#len(df.axes[0]):
        #UpMove = df.get_value(i + 1, 'high') - df.get_value(i, 'high')
        #DoMove = df.get_value(i, 'low') - df.get_value(i + 1, 'low')
	#print df.ix[i+1]
	print df.ix[i+1, 'trades_high'], df.ix[i, 'trades_high'], df.ix[i, 'trades_low'], df.ix[i+1, 'trades_low']
	UpMove = df.ix[i+1, 'trades_high'] - df.ix[i, 'trades_high']
	print "UpMove:", UpMove
	DoMove = df.ix[i, 'trades_low'] - df.ix[i+1, 'trades_low']
	print "DoMove:", DoMove
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else: UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else: DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    print UpI
    DoI = pd.Series(DoI)
    print DoI
    PosDI = pd.Series(pd.ewma(UpI, span = n, min_periods = n - 1))
    print PosDI
    NegDI = pd.Series(pd.ewma(DoI, span = n, min_periods = n - 1))
    print NegDI
    RSI = pd.Series(PosDI / (PosDI + NegDI), name = 'RSI')#'RSI_' + str(n))
    print PosDI / (PosDI + NegDI)
    print RSI
    print "-----------------------"
    df = df.join(RSI)
    return df


def RSI2(df, n):
    i = 0
    UpI = [0]
    DnI = [0]
    while i + 1 < len(df.axes[0]):
	current = df.ix[i+1, 'trades_close']
	prior = df.ix[i, 'trades_close']
	if current - prior > 0:  # up move
	    Up = current - prior
	else:
	    Up = 0
	if current - prior < 0:  # down move
	    Dn = prior - current
	else:
	    Dn = 0
	UpI.append(Up)
	DnI.append(Dn)
	i = i + 1
    UpI = pd.Series(UpI)
    DnI = pd.Series(DnI)
    smoothUpI = pd.Series(pd.ewma(UpI, span = n, min_periods = n - 1))
    smoothDnI = pd.Series(pd.ewma(DnI, span = n, min_periods = n - 1))
    RS = pd.Series(smoothUpI / (smoothDnI + 0.001))
    RSI = pd.Series(100. - 100. / (1 + RS), name = 'RSI')
    df['RSI'] = RSI.values
    return df


def BollingerBands(df, col, n, std_mult):
    MA = pd.Series(pd.rolling_mean(df[col], n))  
    MSD = pd.Series(pd.rolling_std(df[col], n))  
    bu = MA + (MSD * std_mult)  
    BU = pd.Series(bu, name='bb_upper')
    df = df.join(BU)  
    bl = MA - (MSD * std_mult)  
    BL = pd.Series(bl, name='bb_lower')
    df = df.join(BL)  
    return df


def FullSTO(df, w, n_k, n_d, name, start_dt):
    pre_time_dt = start_dt - timedelta(minutes=14)
    pre_time_loc = df.index.get_loc(start_dt - timedelta(minutes=14.5))
    
    close_s = map(lambda c, m: c if not np.isnan(c) else m, df['close'], df['midquote'])
    
    df['pseudo_lows'] = np.nan
    df.ix[df.time >= pre_time_dt, 'pseudo_lows'] = df.ix[df.time >= pre_time_dt, 'microprice']
    lowest_low_s = pd.rolling_min(df['pseudo_lows'], window=w) 
    
    df['pseudo_highs'] = np.nan
    df.ix[df.time >= pre_time_dt, 'pseudo_highs'] = df.ix[df.time >= pre_time_dt, 'microprice']
    highest_high_s = pd.rolling_max(df['pseudo_highs'], window=w)
    
    SOk = pd.Series(pd.rolling_mean((close_s - lowest_low_s)/(highest_high_s - lowest_low_s), window=n_k, min_periods=0.25*n_k), name = 'SO%k_'+name)
    df = df.join(SOk)

    SOd = pd.Series(pd.rolling_mean(SOk, window=n_d, min_periods=0.25*n_d), name = 'SO%d_'+name)
    df = df.join(SOd)
    
    return df    
