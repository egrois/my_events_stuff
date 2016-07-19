/*
1. MarketDataServer passes parsed market data and updated book to a user
2. Able to operate both live and in backtest mode  (using md recordings)
3. Does not use virtual inheritance.
4. All messages are timestamped with packet time
5. No information filtering: all the information from exchange for the symbols of interest is passed down to the user
6. The actual interface will be much richer .. This is just a snippet 
*/

class SecurityDefinition;

class PriceLevelBook
{
public:

    bool is_valid();
    bool has_top();

    int64_t top_ask_price();
    int64_t top_bid_price();

    uint32_t ask_size_at_price( int64_t price );
    uint32_t bid_size_at_price( int64_t price );

    uint32_t ask_orders_at_price( int64_t price );
    uint32_t bid_orders_at_price( int64_t price );
    
    timestamp last_ask_update_time_at_price( int64_t price );
    timestamp last_bid_update_time_at_price( int64_t price );
};

struct InstrumentState
{
    SecurityDefinition  & get_security_definition();
    PriceLevelBook      & get_price_level_book();
    void * user_data;   
};

template < typename HANDLER >
class MarketDataServer
{
    public:
        MarketDataServer(HANDLER *h);
        void set_live( bool is_live );
        bool subscribe( const string &  symbol );
        void run();
        void set_date( date &date);
        void set_start_of_day( timestamp &time);
        void set_end_of_day( timestamp & time); 
};

class BaseMDHandler
{
public: 
    void on_start_of_day(){}
    void on_end_of_day(){}
    void on_symbol_initialized( timestamp time, const SecurityDefinition & secdef, InstrumentState & instrument){}
    void on_system_error( timestamp time, const ErrorCondition & error){}
    template <typename MSG>
    void on_market_data( timestamp time, const MSG &msg, list<InstumentState *> &instruments_impacted){}

};


// Example
BaseMDHandler hr;
MarketDataServer<BaseMDHandler> server( &hr );
server.setLive(true);
server.subscribe( "ESM5" );
server.subscribe( "NQM5" );
server.run();
