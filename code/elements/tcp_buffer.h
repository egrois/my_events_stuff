#ifndef ELEMENTS_TCP_BUFFER_H
#define ELEMENTS_TCP_BUFFER_H

namespace elements
{

// Single threaded buffer to read chopped tcp
// messages or anything like that. The CAPACITY is just set big
// to avoid copying memory often. Will break under ureasonable 
// use

class tcp_buffer
{
    static const uint32_t CAPACITY = 1024000;
    static const uint32_t WATERMARK = 1000000;
public:
    tcp_buffer()
    {
        _buf = new char[CAPACITY];
        reset();
    }

    void reset() 
    {
        _rp = _wp = _buf;
    }

    uint32_t bytes() { return _wp -_rp; }

    char const * read_ptr() 
    { 
        return _rp;
    }

    bool read_advance(uint32_t size) 
    {
        _rp += size; 
        return true;
    }

    char * write_ptr() 
    {
        return _wp;
    }

    bool write_advance(uint32_t size) 
    {
        _wp += size;

        if ( _wp > _buf + WATERMARK )
        {
            if ( bytes() < _rp -_buf )
            {
                uint32_t available = bytes();
                memcpy(_buf, _rp, available );
                _rp = _buf;
                _wp = _buf + available;
                return true;
            }
            else
            {
                return false;
            }
        }
        return true;
    }

private:
    char * _buf;
    char * _rp;
    char * _wp;
};


}

#endif // ELEMENTS_TCP_BUFFER_H
