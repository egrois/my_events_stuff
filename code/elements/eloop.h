#ifndef ELEMENTS_ELOOP_H
#define ELEMENTS_ELOOP_H

#include <list>
#include <string>
#include <parameters_manager.h>
#include <timers_manager.h>
#include <easy_time.h>
#include <oob_logger.h>
#include <log_messages.h>
#include <tsc.h>

namespace elements
{

template <typename ParamMsg, typename Handler>
class eloop 
{
public:
    
    eloop() {}
    ~eloop() { delete _oob_logger; }

    bool initialize( const std::string & config, Handler &h ) 
    {
        log("INFO", "initializing event loop");

        h.on_before_eloop_initialize();

        // PARAMS
        if( !_parameters_adaptor.init(config, &h, 1.0e9)) 
        {
            log("ERROR", "cannot initialize parameters");       
            return false;
        }
        log("INFO","successfully initialized parameters from file " + config);     
        log("INFO", _parameters_adaptor.msg_to_str(true));
        log("INFO", "\n" + _parameters_adaptor.msg_to_str(false));
        h.set_parameters_manager(&_parameters_adaptor);

        // AFFINITY
        auto params =  _parameters_adaptor.get_parameters().strategy();     
        if(params.eloop_cpu_affinity() >= 0 && params.is_realtime() )
        {
            cpu_set_t mask;
            CPU_ZERO(&mask);
            CPU_SET((unsigned int)params.eloop_cpu_affinity(), &mask);  
            sched_setaffinity(0,sizeof(mask), &mask);
        }
        log("INFO","locked eloop to processor ");     

        // TIMERS
        if( !_timers_adaptor.init(&h, this))
        {
            log("ERROR", "cannot initialize timers");       
            return false;
        } 
        log("INFO","successfully initialized timers manager");     
        h.set_timers_manager(&_timers_adaptor);

        // OOB_LOGGER
        
        std::string sdate = str_date(clock::now());
        _oob_logger =  new oob_logger<log_message>( 
            params.oob_logger_file_prefix() + "." + sdate, 
            10000, 
            params.oob_logger_affinity());

        // ELOOP 
        h.set_event_loop(this);
        h.on_after_eloop_initialize();

        log("INFO","finished initializing event loop");
    }

    void run()
    {
        auto params =  _parameters_adaptor.get_parameters().strategy();     
        while(true)
        {
            auto t1 = _tsc();   
            _parameters_adaptor.poll();
            auto t2 = _tsc();   
            _timers_adaptor.poll();
            auto t3 = _tsc();
            if( params.log_timing() )
            {
                auto ts = clock::now();
                auto epoch = to_epoch_nanos(ts);

                _log_msg.type = MSG_TYPE::MSG_LATENCY;
                _log_msg.uber.a_latency.epoch = epoch;
                _log_msg.uber.a_latency.nanos = _tsc.nanos(t1,t2);
                _log_msg.uber.a_latency.type = LATENCY_TYPE::LT_PARAMS;
                _oob_logger->log(_log_msg);
                
                _log_msg.type = MSG_TYPE::MSG_LATENCY;
                _log_msg.uber.a_latency.epoch = epoch;
                _log_msg.uber.a_latency.nanos = _tsc.nanos(t1,t2);
                _log_msg.uber.a_latency.type = LATENCY_TYPE::LT_TIMERS;
                _oob_logger->log(_log_msg);
            }
        }
    }

    template <typename T>
    void log(std::string prefix, T msg)
    {
        std::cout << str(now()) << " " << prefix << " "<< std::string(msg) << std::endl;
    }

    timestamp now() 
    {
        if(_is_realtime )
            return clock::now(); 
        else
            assert(false);
    }

private:

    bool _is_realtime = true;

    parameters_manager<ParamMsg,Handler>                _parameters_adaptor;
    timers_manager<Handler, eloop<ParamMsg,Handler> >   _timers_adaptor;
    oob_logger<log_message> *                           _oob_logger;
    log_message                                         _log_msg;
    TSC                                                 _tsc; 
    
};

}

#endif // ELEMENTS_ELOOP_H
