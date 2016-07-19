enum LATENCY_TYPE 
{
    LT_PARAMS = 1,
    LT_TIMERS = 2,
    LT_DEFAULT = 3
}


struct latency
{
    LATENCY_TYPE type;
    int nanos;
};
