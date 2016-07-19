#include "gtest/gtest.h"
#include "linux_event_loop.h"
#include "udp_sender.h"
#include <string>
#include <thread>
#include <memory>
#include <chrono>
#include <iostream>
#include <sys/time.h>

using namespace std;

class timer_handler
{
public:
    typedef elements::linux_event_loop<timer_handler> event_loop_type;
    void on_udp_packet(int fd, char *buf, int size, elements::timestamp ts) {}
    void on_tcp_packet(int fd, char *buf, int size, elements::timestamp ts) {}
    void register_linux_event_loop(event_loop_type *c) { _loop = c; }
    void on_disconnect(int fd) {}
    void on_timer(int fd, char *buf, int size, elements::timestamp ts) 
    {
        if(test_id == 1) test1();
        if(test_id == 3) test3(fd);
    }

    int test_id  = 0;
    event_loop_type * _loop = nullptr;
    
    int stop_id =  -1;


    void test1()
    {
        static int count = 0;
        count++;
        if( count > 10)
        {
            _loop->stop();
            count = 0;
        }
    }

    void test3(int fd)
    {
        if(fd == stop_id) _loop->stop();
    }

};

TEST(linux_event_loop, timer1)
{
    using namespace elements;
    timer_handler handler;
    handler.test_id = 1;
    linux_event_loop<timer_handler> loop(&handler);
    timestamp ts = clock::now();
    int fd = loop.add_timer( false, ts, milliseconds(1), milliseconds(1), true);
    loop.run();
    timestamp ts2 = clock::now();
    EXPECT_GE(ts2 - ts, microseconds(10500));
    EXPECT_LE(ts2 - ts, microseconds(11500));

}

TEST(linux_event_loop, timer2)
{
    using namespace elements;
    timer_handler handler;
    handler.test_id = 1;
    linux_event_loop<timer_handler> loop(&handler);
    timestamp ts = clock::now();
    int fd = loop.add_timer( true, ts + milliseconds(1), milliseconds(0), milliseconds(1), true);
    loop.run();
    timestamp ts2 = clock::now();
    EXPECT_GE(ts2 - ts, microseconds(10500));
    EXPECT_LE(ts2 - ts, microseconds(11500));
}

TEST(linux_event_loop, timer3)
{
    using namespace elements;
    timer_handler handler;
    handler.test_id = 3;
    linux_event_loop<timer_handler> loop(&handler);
    timestamp ts = clock::now();
    int fd = loop.add_timer( false, ts, milliseconds(1), milliseconds(1), true);
    handler.stop_id = loop.add_timer( false, ts, milliseconds(200), milliseconds(0), true);
    loop.run();
    timestamp ts2 = clock::now();
    EXPECT_GE(ts2 - ts, microseconds(199000));
    EXPECT_LE(ts2 - ts, microseconds(201000));
}

class random_udp_writer
{
public:
    random_udp_writer(
        std::string ip, 
        std::string port, 
        int max_buffer,
        elements::microseconds avg_time,
        int id)
        :   _ip(ip), 
            _port(port), 
            _max_buffer(max_buffer), 
            _avg_time(avg_time), 
            _id(id) 
    {
    }

    void run()
    {
        _is_running = true; 
        using namespace std;
        _thread = new std::thread( bind( &random_udp_writer::_run, this ) );
    }

    void stop()
    {
        _is_running = false;
        _thread->join();
        delete _thread;
    }

private:

    void _run()
    {
        elements::udp_sender sender(_ip.c_str(), _port.c_str());
        std::this_thread::sleep_for(elements::milliseconds(100));
        while(true)
        {
            if(_is_running == false) 
                break;
            
            int mics = rand() % _avg_time.count();
            std::this_thread::sleep_for(elements::microseconds(mics));
            std::string sent_time = elements::str(elements::clock::now());
            std::string msg =  sent_time + " " + std::to_string(mics) + " " + std::string(_port) + " " +  std::to_string(_sequence++);
            sender.send(msg);
        }
    }

    std::string _ip, _port;
    int _max_buffer;
    elements::microseconds _avg_time;
    bool _is_running = false;
    std::thread * _thread;
    int _id;
    int _sequence = 0;
};

class udp_handler
{
public:
    typedef elements::linux_event_loop<udp_handler> event_loop_type;
    void on_udp_packet(int fd, char *buf, int size, elements::timestamp ts) 
    {
        buf[size] = '\0';
        _captured_messages.push_back(elements::str(ts)  + " " + std::string(buf));
    }
    void on_tcp_packet(int fd, char *buf, int size, elements::timestamp ts) {}
    void register_linux_event_loop(event_loop_type *c) { _loop = c; }
    void on_disconnect(int fd) {}
    void on_timer(int fd, char *buf, int size, elements::timestamp ts) { _loop->stop(); }

    std::vector<std::string> get_messages() { return _captured_messages; } 
    
private:
    event_loop_type * _loop = nullptr;
    std::vector<std::string> _captured_messages;
};

TEST(linux_event_loop, udp1)
{
    using namespace elements;
    udp_handler handler;
    linux_event_loop<udp_handler> loop(&handler);
    for(int i = 0; i < 24; ++i)
        loop.add_udp( "127.0.0.1", std::to_string(5000 + i), 2000, false, true);
    loop.add_timer( false, clock::now(), seconds(1), milliseconds(1), true);

    std::vector<random_udp_writer * > writers;
    for(int i = 0; i < 24; ++i)
        writers.push_back(new random_udp_writer(
            "127.0.0.1", std::to_string(5000 + i), 100, elements::microseconds(20), i) );  
   
    for(auto writer: writers)
        writer->run();

    loop.run();

    for(auto writer :writers)
    {
        writer->stop();
        delete writer;
    }
}
