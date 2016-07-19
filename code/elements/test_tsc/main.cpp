#include <tsc.h>
#include <iostream>


using namespace elements;

int main(int argc, char **argv)
{
    TSC tsc;

    for(int i =0; i<10; ++i)
    {
        uint64_t t1 = tsc();
        uint64_t t2 = tsc();

        std::cout  << tsc.nanos(t1, t2) << " " << tsc.nanos_per_tick() <<  std::endl;
    }
    return 0;
}
