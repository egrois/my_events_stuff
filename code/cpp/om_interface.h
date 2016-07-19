// Similar to md order book but has OUR orders in it. Also points to the Orders
class OrderBook;

// Some description of exchange sessions
class Session;

// Contains order instructions, order state
struct Order

// Encodes success/failure for an order submission
struct Status;


// Ino regarding risk limits
struct RiskLimits


class RiskManager
{
    public:
        int32_t open_position( int32_t instrument_id )
        double pnl( int32_t instrument_id );
        int32_t total_open_position();
        double total_pnl( );
        RiskLimits & get_risk_limits();            
};


template <typename HANDLER>
class OrderManager
{
public:
    OrderManager(HANDLER *h);
    void run();
    Sessions & get_sessions();
    OrderBook *get_order_book(string & symbol);
    OrderBook *get_order_book(int32_t  instrument_id);
    Status & pre_submit_order( Order & order, Session &session);
    Status & submit_order( Order & order, Session & session);
    RiskManager & get_risk_manager();
};

class BaseHandler
{
public:
    template <typename MSG, typename Order>
    void on_order_event(timestamp ts, Order &order, MSG &msg);
    void on_system_error(timestamp ts, ErrorInfo & info);
};

// Example
BaseHandler h;
OrderManager<BaseHandler> om(&h);
om.run();


