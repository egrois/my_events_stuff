
#include "gtest/gtest.h"
#include "easy_time.h"


using namespace std;
using namespace elements;

TEST(easy_time, test1)
{
    timestamp stamp = clock::now();
    timespec ts = to_timespec(stamp);
    timestamp stamp1 = from_timespec(ts);
    EXPECT_EQ(stamp,stamp1); 
    
    timeval tv = to_timeval(stamp);
    timestamp stamp2 = from_timeval(tv);
    timeval tv2 = to_timeval(stamp2);
    EXPECT_EQ(tv.tv_sec,tv2.tv_sec); 
    EXPECT_EQ(tv.tv_usec,tv2.tv_usec); 
}

