#ifndef ELEMENTS_MESSENGER_H
#define ELEMENTS_MESSENGER_H

#include <string>
#include <thread>
#include <functional>
#include "easy_time.h"
#include "threaded_runner.h"

namespace elements
{
template <typename msg_type, typename queue_type, typename writer_type>
class messenger
{
public:
    messenger(size_t queue_size, writer_type & writer) : _queue(queue_size), _writer(writer) { }
    bool send(msg_type& msg) { return _queue.push( msg ); }
    queue_type * get_queue() { return &_queue; }
    bool write(msg_type& msg) { _writer.write( msg ); }
private:
    queue_type _queue;
    writer_type & _writer;
};

template <typename msg_type, typename messenger_type>
class message_router : public threaded_runner
{
public:
    message_router(){}
    virtual ~message_router(){}
    bool add(messenger_type * messenger)
    { 
        _messengers.push_back(messenger);
    }
private:
    virtual void _process()
    {
        msg_type msg;
        for(auto messenger: _messengers)
        {
            while(!messenger->get_queue()->empty())
            {
                if(!messenger->get_queue()->pop(msg))
                {
                    break;
                }
                messenger->write(msg);
            }
        }
    }
private:
    std::vector< messenger_type * > _messengers;    
};

}

#endif // ELEMENTS_MESSENGER_H

