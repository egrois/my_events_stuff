#include <iostream>
#include "parameters_manager.h"
#include "params.pb.h"

using namespace elements;


class dummy_handler
{
public:
   
    template<typename MSG> 
    void on_parameters_update(MSG& msg)
    {
        auto strategy = &msg.strategy();
        std::cout << strategy->name() << std::endl;  
        for( int i = 0 ; i < strategy->instrument_size(); ++i)
        {
            std::cout << strategy->instrument(i).symbol() << std::endl;
        }       
    }

    template<typename MSG>
    bool validate_parameters(MSG &msg)
    {
        return true;
    }
};

int main(int argc,char **argv)
{
    parameters_manager < params::Params, dummy_handler > p;
    dummy_handler h;
    p.init("param_test_in", &h, 1.0e10;
   
    while(1)
    {
       p.poll(); 
    }
 
    return 1;
}
