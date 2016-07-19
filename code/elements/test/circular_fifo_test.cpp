#include "gtest/gtest.h"
#include "circular_fifo.h"
#include <string>
#include <thread>
#include <memory>
#include <chrono>
#include "zero_copy_buffer.h"

namespace {
struct Message {
    Message( ){}
    Message( int aa, double bb, std::string ss )
        : a( aa ), b( bb ), s( ss ) {}
    int a;
    double b;
    std::string s;
};
}

TEST(circular_fifo, test1){

    elements::circular_fifo<Message> fifo(1000);

    Message msg1(3,10.0,"blah");
    fifo.push(msg1);
    
    Message msg2;
    fifo.pop(msg2);

    EXPECT_EQ( msg2.a, msg1.a );
    EXPECT_EQ( msg2.b, msg1.b );
    EXPECT_EQ( msg2.s, msg1.s);
}

TEST(circular_fifo,test2) {
    
    elements::circular_fifo<Message> fifo(1000);

    for(int i = 0; i < 2000; ++i){
       fifo.push(Message(i, i*i, "blah") );
    }

    EXPECT_EQ(fifo.full(), true);
    
    Message msg2;
    for(int i = 0; i < 999; ++i){
       fifo.pop(msg2);
    }
    
    EXPECT_EQ(fifo.empty(), false);
    fifo.pop(msg2);
    EXPECT_EQ(msg2.a, 999);
    EXPECT_EQ(fifo.empty(), true);
}


TEST(circular_fifo, test3){

    typedef elements::circular_fifo<Message> buf;
    typedef std::shared_ptr< buf > buf_ptr; 
    buf_ptr fifo( new buf(1000000) );

    auto th = std::thread([fifo](){
        int msgId = 0;
        while(true){
            Message msg2;
            while(!fifo->empty()){
                fifo->pop(msg2);
                EXPECT_EQ(msg2.a, msgId);
                ++msgId;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            if(fifo->empty())
                break;
        }
    });

    for(int i = 0; i < 20000; ++i){
        fifo->push(Message(i, 2.0 * i, "blah"));
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }

    th.join();
}

TEST(circular_fifo, test4)
{
    typedef std::shared_ptr< elements::zero_copy_buffer > buf_ptr; 
    buf_ptr buf( new elements::zero_copy_buffer(1000000,10) );

    bool is_finished = false;
    bool *pis_finished = &is_finished;
    char *p;
    int m = 0;

    auto th = std::thread([buf, pis_finished]()
    {
        char *rp;
        size_t sz;
        char last = 57;
        
        while(true)
        {
            if(*pis_finished)
                break;

            if(!buf->read_acquire(&rp, &sz))
                continue;

            if(last > 56)
                EXPECT_EQ(rp[0], 48);
            else
                EXPECT_EQ(rp[0], last + 1);
            last = rp[ sz - 1];  
            buf->read_release();
        }
    });

    for(int i = 0; i < 1000; ++i)
    {
        buf->write_acquire(&p);
        for(int j = 0; j < 8; j++)
        {
            p[j] = 48 + m % 10 ;
            EXPECT_LE(p[j], 57);
            m++;
        }
        buf->write_release(8);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    is_finished = true;
   
    th.join();
}
