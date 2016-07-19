#include "gtest/gtest.h"
#include "out_of_band_logger.h"

enum MSG_TYPE
{
    MSG1,
    MSG2,
    MSG3
};

struct msg1
{
    double a;
};

std::ostream& operator <<(std::ostream& os, const msg1 & msg)
{
    return os << "msg1 " << msg.a;
}

struct msg2
{
    double a;
};
std::ostream& operator <<(std::ostream& os, const msg2 & msg)
{
    return os << "msg2 " << msg.a;
}

struct msg3
{
    double a;
};
std::ostream& operator <<(std::ostream& os, const msg3 & msg)
{
    return os << "msg3 " << msg.a;
}

struct some_message
{
    MSG_TYPE type;
    union
    {
        msg1 d_msg1;
        msg2 d_msg2;
        msg3 d_msg3;
    } uber;
};

std::ostream& operator <<(std::ostream& os, const some_message & msg)
{
    switch(msg.type)
    {
        case MSG_TYPE::MSG1:
            os << msg.uber.d_msg1;
            break;
        case MSG_TYPE::MSG2:
            os << msg.uber.d_msg2;
            break;
        case MSG_TYPE::MSG3:
            os << msg.uber.d_msg3;
            break;
        default:
            std::cout << "ERROR" << std::endl;
    };
    return os << std::endl;
}

TEST(out_of_band_logger, test1)
{
    some_message msg;
    elements::out_of_band_logger<some_message> logger("log_test_file", 10000, -1);
    for (int i = 0; i < 1000; i++)
    {
        if(i % 3 == 0)
        {
            msg.type = MSG_TYPE::MSG1;
            msg.uber.d_msg1.a = 13;
        }
        else if(i % 3 == 1)
        {
            msg.type = MSG_TYPE::MSG2;
            msg.uber.d_msg2.a = 173;
        }
        else
        {
            msg.type = MSG_TYPE::MSG3;
            msg.uber.d_msg3.a = 88;
        }

        logger.log(msg);
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}
