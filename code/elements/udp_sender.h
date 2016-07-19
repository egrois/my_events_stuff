#ifndef ELEMENTS_UDP_SENDER_H
#define ELEMENTS_UDP_SENDER_H

#include <iostream>
#include <sys/socket.h>
#include <netdb.h>
#include <time.h>

using namespace std;

namespace elements
{
class udp_sender
{
public:
    
    udp_sender(char const * addr, char const * port) 
    {
        int err;
        struct addrinfo hint;
        bzero(&hint, sizeof(struct addrinfo));
        hint.ai_family = AF_INET;
        hint.ai_socktype = SOCK_DGRAM;
        struct addrinfo *ailist;
        if ( (err = getaddrinfo(addr, port, &hint, &ailist)) < 0 )
        {
	        std::cerr << "getaddrinfo error: " << std::endl;
	        throw;
        }
        sinp_= (struct sockaddr * ) ailist->ai_addr;
        if((_socketfd = socket(AF_INET, SOCK_DGRAM, 0))<0)
        {
	        std::cerr << "socket error" << std::endl;
	        throw;
        }
    }

    ~udp_sender(){}
    
    void send(string const &msg)
    {
        int err;
        if((err = sendto(_socketfd, msg.c_str(), msg.length(), MSG_DONTWAIT, sinp_,  sizeof(struct sockaddr)))<0 )
	        std::cerr << "cannot send message" << std::endl;
    }

    void send(const char * buf, uint32_t size)
    {
        int err;
        if((err = sendto(_socketfd, buf, size, MSG_DONTWAIT, sinp_,  sizeof(struct sockaddr)))<0 )
	        std::cerr << "cannot send message" << std::endl;
    }
    


private:
    enum {MAX_HOST_NAME = 100};
    int _socketfd;
    struct sockaddr *sinp_;  
    
    int get_send_buffer_size(int s)
    {
        int bufsize;
        socklen_t len = sizeof(bufsize);
        getsockopt(s, SOL_SOCKET, SO_SNDBUF,  (void*)&bufsize, &len);
        return bufsize;
    }

    bool set_send_buffer_size(int s, int bufsize)
    {
        socklen_t len =sizeof(bufsize);
        if( setsockopt(s, SOL_SOCKET, SO_SNDBUF,(void*)&bufsize, len ) == 0)
	        return true;
        else
	        return false;
    }
};
}

#endif // ELEMENTS_UDP_SENDER_H
 
