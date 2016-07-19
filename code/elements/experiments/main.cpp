#include <eloop.h>
#include "params.pb.h"


using namespace elements;

template <typename ParamMsg>
class strategy
{
public:
    typedef strategy<ParamMsg>                              StrategyType;
    typedef eloop<ParamMsg,StrategyType>                    EventLoopType;
    typedef parameters_manager<ParamMsg,StrategyType>       ParametersManagerType;
    typedef timers_manager <StrategyType,EventLoopType>     TimersManagerType;

    void on_before_eloop_initialize(){}
    void on_after_eloop_initialize()
    {
        auto start = _event_loop->now();
        auto stop = start + milliseconds(1000000);
        auto interval = milliseconds(1000);
        auto id =  _timers_manager->register_timer(start, stop, interval);
        std::cout << "TIMER id= " << id << std::endl; 
    }

    // PARAMS 
    void on_parameters_update(ParamMsg& msg) {  std::cout << "Param Change" << std::endl; }
    bool validate_parameters(ParamMsg &msg) { return true; }

    // TIMERS
    void on_timer(timestamp & t, uint32_t timer_id) 
    {
        std::cout << "TIMER CALLBACK: " << str(t) << " id= " << timer_id << std::endl; 
    }

    // Used by eloop ==>
    void set_event_loop(EventLoopType *p) { _event_loop = p; }
    void set_parameters_manager(ParametersManagerType *p) { _params_manager = p; }
    void set_timers_manager(TimersManagerType *p) { _timers_manager = p; }
    // <===

private:
    EventLoopType           *_event_loop;
    ParametersManagerType   *_params_manager;
    TimersManagerType       *_timers_manager;
};

typedef strategy<params::Params> Strategy;

int main( int argc, char ** argv)
{
    Strategy handler;
    Strategy::EventLoopType e;
    e.initialize("strategy.param", handler);
    e.run();

    return 0;
}
