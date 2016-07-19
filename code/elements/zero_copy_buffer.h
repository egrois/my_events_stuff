#ifndef ELEMENTS_ZERO_COPY_BUFFER_H
#define ELEMENTS_ZERO_COPY_BUFFER_H

#include "circular_fifo.h"

// Thread safe producer consumer "queue" that uses preallocated memory
// passes raw bytes between threads. Producer calls write_acquire
// and write_release consumer calls read_acquire and read_release. 
// assumes that internal storage is very large ...


namespace elements
{

class zero_copy_buffer
{
public:

    // capacity - total preallocated memory in bytes
    // max_size - maximum message size
    zero_copy_buffer(size_t capacity, size_t max_size) 
        : _capacity(capacity), _max_size(max_size), _queue( ( 100 * capacity) / max_size) 
    { 
        _buffer = new char[_capacity];
        _wp = _buffer;
        _beyond_buffer = _wp + _capacity;
    }
    
    ~zero_copy_buffer() { delete [] _buffer; }
    
    bool write_acquire(char ** p) 
    {
        if(_queue.full())
        {
            return false;
        }
        else
        { 
            *p = _wp;
            return true; 
        }
    }

    bool write_release(size_t size)
    {
        static message m;
        m.p = _wp;
        m.size = size;
        _wp += size;
        if(_wp + _max_size >= _beyond_buffer)
            _wp = _buffer;
        return _queue.push(m);
    }
    
    bool read_acquire(char **p, size_t *size)
    {
        static message m;
        if(_queue.empty())
        {
            return false;
        }
        else
        {
            if(!_queue.peek(m))
            {
                return false;
            }
            *p = m.p;
            *size = m.size;
        }
        return true;
    }

    bool read_release()
    {
        _queue.drop();
    }

private:

    struct message
    {
        size_t size;
        char *p;
    };

    elements::circular_fifo<message> _queue;
    char * _buffer;
    char * _beyond_buffer;
    char * _wp;

    size_t _capacity;
    size_t _max_size;
};

}

#endif // ELEMENTS_ZERO_COPY_BUFFER_H
