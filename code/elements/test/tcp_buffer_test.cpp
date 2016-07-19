#include "gtest/gtest.h"
#include "tcp_buffer.h"

using namespace std;
using namespace elements;

TEST( tcp_buffer, test1)
{
    char data[] = "1234567890123456789012345678901234567890_";
    tcp_buffer buf;
    int index = 0;
    for(int k = 0; k < 100000; k++)
    {
        char *wp = buf.write_ptr();
        int bytes_to_write = rand() % 100 + 5;
        for (int m = 0; m < bytes_to_write; ++m )
        {
            wp[m] = data[(index + m) % (sizeof(data) - 1)];  
        }
        index += bytes_to_write;
        buf.write_advance(bytes_to_write);
        while (buf.bytes() > 0)
        {
            char const *rp = buf.read_ptr();
            bool found = false;        
            for(int i = 0; i < buf.bytes(); ++i)
            {
                if(rp[i] == '_')
                {
                    buf.read_advance(i + 1);
                    found = true;
                    EXPECT_EQ(41, i + 1); 
                    break;
                }
            }
            if (!found)
                break;
        }
    }
}

