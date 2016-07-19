

























def main():
    print "starting..."
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
        stop_dt = datetime.strptime(event_date + " " + trading_stop_time, '%Y-%m-%d %H:%M:%S')


if __name__ == "__main__": main()
