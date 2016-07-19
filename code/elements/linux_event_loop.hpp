#include <linux_event_loop.h>
#include <time.h>
#include <iostream>
#include <sys/socket.h>
#include <netdb.h>
#include <sys/timerfd.h>
#include <strings.h>
#include <string.h>
#include <fcntl.h>
#include <sys/time.h>
#include <sys/types.h>

template <typename handler>
void linux_event_loop<handler>::init()
{
    _fd_epoll = epoll_create(20);

    if(_fd_epoll  == -1)
    {
        std::cerr << "cannot initialize linux_event_loop: epoll_create" << std::endl;
        throw;
    }
}

template <typename handler>
int linux_event_loop<handler>::add_udp(
    std::string ip,
    std::string port,
    int buffer_size,
    bool is_mc,
    bool enabled)
 { 
    int err;
    struct addrinfo hint;
    bzero( &hint, sizeof( struct addrinfo ) );
    hint.ai_family = AF_INET;
    hint.ai_socktype = SOCK_DGRAM;
    struct addrinfo * ailist;

    if( ( err = getaddrinfo( ip.c_str(), port.c_str(), &hint, &ailist ) ) < 0 )
    {
        std::cerr << "cannot add_udp : getaddrinfo error: " << std::endl;
        return -1;
    }

    int socket_fd;
    if( ( socket_fd = socket(AF_INET, SOCK_DGRAM, 0 ) ) < 0 )
    {
        std::cerr << "cannot create socket:  socket error: " << std::endl; 
        return -1;
    }

    if( !set_kernel_ts( socket_fd ) ) 
        return -1;

    if( !set_recv_buffer_size( socket_fd, buffer_size ) )
        return -1;

    err = bind( socket_fd, ailist->ai_addr, ailist->ai_addrlen );

    if( err < 0 ) 
    {
        std::cerr << strerror( err ) << std::endl;
        return -1;
    }
        
    if( is_mc && !mc_join( socket_fd, ailist ) )
    {
        std::cerr << "error: mc_join" << std::endl;
        return -1;
    }

    connection c;
    c.ip = ip;
    c.port = port;
    c.type = UDP;
    c.is_enabled = false;
    c.ev.events = EPOLLIN | EPOLLRDHUP;
    c.ev.data.fd = socket_fd;
    _cons[socket_fd] =  c;

    if( enabled && ( enable(socket_fd ) < 0 ) )
        return -1;

    return socket_fd; 
}

template <typename handler>
int linux_event_loop<handler>::add_tcp(
    std::string ip, 
    std::string port,
    int rec_buf_size,
    int send_buf_size,
    bool enabled)
{ 

    int err;
    struct addrinfo hint;
    bzero(&hint, sizeof(struct addrinfo));    
    hint.ai_family = AF_INET;
    hint.ai_socktype = SOCK_STREAM;
    struct addrinfo * ailist;
    if((err = getaddrinfo(ip.c_str(), port.c_str(), &hint, &ailist)) < 0)
    {
        std::cerr << "cannot add_tcp : getaddrinfo error: " << std::endl;
        return -1;
    }

    int socket_fd;
    if( (socket_fd = socket(AF_INET, SOCK_STREAM, 0 )) < 0 ) 
    {
        std::cerr << "cannot create socket:  socket error: " << std::endl; 
        return -1;
    }

    if(!set_recv_buffer_size(socket_fd, rec_buf_size))
        return -1;

    if(!set_send_buffer_size(socket_fd, send_buf_size))
        return -1;

    err = connect(socket_fd, ailist->ai_addr, ailist->ai_addrlen);

    if(err < 0)
    {
        std::cerr << "connect: " << strerror(err) << std::endl;
        close(socket_fd);
        return -1;
    }
    
    connection c;
    c.ip = ip;
    c.port = port;
    c.type = TCP;
    c.is_enabled = false;
    c.ev.events = EPOLLIN | EPOLLRDHUP;
    c.ev.data.fd = socket_fd;
    _cons[socket_fd] =  c;

    if( enabled && ( enable( socket_fd ) < 0 ) )
        return -1;
    
    return socket_fd; 
}

template <typename handler>
int linux_event_loop<handler>::add_timer(
    bool is_absolute,
    timestamp absolute_time,
    nanoseconds relative_time,
    nanoseconds freq,
    bool enabled) 
{
    int timer_fd = timerfd_create( CLOCK_REALTIME, TFD_NONBLOCK );
    if( timer_fd == -1 )
    {
        std::cerr << "cannot create timer" << std::endl;
        return -1;
    }

    if( set_nonblock(timer_fd) )
    {
        std::cerr << "cannot set nonblock on timer" << std::endl;
        return -1;
    }
    
    struct itimerspec tspec;

    tspec.it_value = is_absolute ? to_timespec( absolute_time ) : to_timespec( relative_time );

    if( freq != nanoseconds(0) )
    {
        auto sec = std::chrono::duration_cast<seconds>(freq);
        tspec.it_interval.tv_sec = sec.count();
        tspec.it_interval.tv_nsec = std::chrono::duration_cast<nanoseconds>( freq - sec ).count();
    } 
    else 
    {
        tspec.it_interval.tv_sec = 0;
        tspec.it_interval.tv_nsec = 0;
    }

    int flags =  is_absolute ? TFD_TIMER_ABSTIME : 0; 
    if( timerfd_settime(timer_fd, flags, &tspec, NULL) != 0 )
    {
        std::cerr << "cannot set timer" << std::endl;
        return -1;
    }
    
    connection c;
    c.type = TIMER;
    c.is_enabled = false;
    c.ev.events = EPOLLIN;
    c.ev.data.fd = timer_fd;
    _cons[timer_fd] =  c;
    
    if( enabled )
    {
        int err = enable(timer_fd);
        if( err < 0 )
        {
            remove(timer_fd);
            std::cerr << "cannot enable timer: " << std::endl;
            return -1;
        }
    }

    return timer_fd; 
}

template <typename handler>
int linux_event_loop<handler>::remove(int fd) {
 
    auto it =_cons.find(fd);
    
    if( it  == _cons.end() )
    {
        std::cerr << "cannot remove, descriptor fd= " << fd << " does not exist" << std::endl;
        return -1;
    }
    
    close(fd);
    _cons.erase(it);
   
    return 0;
}

template <typename handler>
int linux_event_loop<handler>::enable(int fd)
{
    auto it =_cons.find(fd);
    if( it == _cons.end() )
    {
        std::cerr << "cannot enable epoll, descriptor does not exist" << std::endl;
        return -1;
    }
    
    if( it->second.is_enabled )
    {
        std::cerr << "Connection already enabled" << std::endl;
        return -1;
    }
    
    if( epoll_ctl( _fd_epoll, EPOLL_CTL_ADD, fd, &it->second.ev) < 0 )
    {
        std::cerr << "Error epoll_ctl" << std::endl;
        return -1;
    }
    
    it->second.is_enabled = true;
    
    return 0; 
}

template <typename handler>
int linux_event_loop<handler>::disable(int fd)
{
    auto it =_cons.find(fd);

    if( it == _cons.end() )
    {
        std::cerr << "cannot disable epoll, descriptor does not exist" << std::endl;
	    return -1;
    }

    if( !it->second.is_enabled )
    {
	    std::cerr << "Connection already disabled" << std::endl;
	    return -1;
    }

    if( epoll_ctl( _fd_epoll, EPOLL_CTL_DEL, fd, NULL ) == -1 )
    {
	    std::cerr << "Error epoll_ctl EPOLL_CTL_DEL" << std::endl;
	    return -1;
    }

    it->second.is_enabled = false;

    return 0;  
}

template <typename handler>
void linux_event_loop<handler>::run()
{
    int ready_cnt;
    int buf_size;
    struct epoll_event evlist[MAX_EVENTS];
    char buffer[MAX_BUFFER];
 
    struct msghdr msg;
    struct iovec iov;
    char ctrl[CMSG_SPACE(sizeof(struct timeval))];
    struct cmsghdr *cmsg = (struct cmsghdr *) & ctrl;    
    struct timeval kernel_time; 
    
    iov.iov_base = buffer;
    iov.iov_len = MAX_BUFFER;
    msg.msg_name = NULL;
    msg.msg_namelen = 0;
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = (char *)ctrl;
    msg.msg_controllen = sizeof(ctrl);
   
    struct timeval tcp_time; 

    _is_running =  true;
    while( _is_running )
    {
        ready_cnt = epoll_wait( _fd_epoll, evlist, MAX_EVENTS, -1 );

        if( ready_cnt < 0 ) 
        {
            char m[1000];
            perror(m);
            std::cerr  << "Error in epoll_wait " << m <<  std::endl;
            continue;
        }

        if( ready_cnt == 0 )
            continue;

        for( int i = 0; i < ready_cnt; ++i)
        {
            if (evlist[i].events & (EPOLLHUP | EPOLLERR | EPOLLRDHUP))
            {
                std::string what_exactly;

                if( evlist[i].events & EPOLLHUP ) what_exactly = "EPOLLHUP";
                else if ( evlist[i].events & EPOLLERR ) what_exactly = "EPOLLERR";
                else if ( evlist[i].events & EPOLLRDHUP ) what_exactly = "EPOLLRDHUP";	

                std::cerr << "Error " << what_exactly << " on fd =" << evlist[i].data.fd << std::endl;
                disable(evlist[i].data.fd);
                remove(evlist[i].data.fd);
                _handler->on_disconnect(evlist[i].data.fd);
                continue;
            }

            con_type type = _cons[ evlist[i].data.fd ].type;
            if(type == UDP)
            {
                if(evlist[i].events & EPOLLIN) 
                {
                    iov.iov_len = MAX_BUFFER;
                    while( (buf_size = recvmsg(evlist[i].data.fd, &msg, MSG_DONTWAIT)) > 0 ) 
                    {
                        if(cmsg->cmsg_level != SOL_SOCKET ||  
                            cmsg->cmsg_type != SCM_TIMESTAMP || 
                            cmsg->cmsg_len != CMSG_LEN(sizeof(kernel_time)))
                        {
                            std::cerr << "cannot obtain kernel timestamp ignoring UDP packet" << std::endl;
                        } 
                        else 
                        {
                            _handler->on_udp_packet(evlist[i].data.fd, buffer, buf_size, 
                                from_timeval(*(struct timeval *)CMSG_DATA(cmsg)));
                        }
                        iov.iov_len = MAX_BUFFER;
                    }	    
                }
            }
            else if(type == TCP) 
            {
                if(evlist[i].events & EPOLLIN) 
                {
                    while( ( buf_size = recv(evlist[i].data.fd, buffer, MAX_BUFFER, MSG_DONTWAIT)) > 0) 
                    {
                        gettimeofday((struct timeval *)&tcp_time, NULL);
                        _handler->on_tcp_packet(evlist[i].data.fd, buffer, buf_size, from_timeval(tcp_time));
                    }
                }
            }
            else if(type == TIMER) 
            {
                if(evlist[i].events & EPOLLIN) 
                {
                    if ( (buf_size = read(evlist[i].data.fd, buffer, MAX_BUFFER)) > 0) 
                    {
                        gettimeofday((struct timeval *)&tcp_time, NULL);
                        _handler->on_timer(evlist[i].data.fd, buffer, buf_size, from_timeval(tcp_time));
                    }
                }	
            }
        }
    }
}

template <typename handler>
void linux_event_loop<handler>::stop()
{
    _is_running = false;
}

template <typename handler>
int linux_event_loop<handler>::write_tcp(int fd, std::string &msg)
{
    auto it =_cons.find(fd);   
    if( it == _cons.end()) 
    { 
        std::cerr << "cannot write_tcp, descriptor does not exist" << std::endl;
        return -1;
    } 

    if (send(fd, msg.c_str(), msg.size(), MSG_DONTWAIT) < 0)
    {
        std::cerr << "cannot send on tcp" << std::endl; 
        return -1;
    }
}

template <typename handler>
void linux_event_loop<handler>::print_connections()
{
    std::cout << "connections: " ;
    auto it =_cons.begin();
    for(it = _cons.begin(); it != _cons.end(); ++it)
    {
        std::cout << it->first << ",";
    }
    std::cout << std::endl;
}

template <typename handler>
int linux_event_loop<handler>::get_recv_buffer_size(int s)
{
    int bufsize;
    socklen_t len = sizeof(bufsize);
    getsockopt(s, SOL_SOCKET, SO_RCVBUF, (void*)&bufsize, &len );
    return bufsize;
}

template <typename handler>
bool linux_event_loop<handler>::set_recv_buffer_size(int s, int bufsize)
{
    socklen_t len = sizeof(bufsize);
    if(setsockopt(s, SOL_SOCKET, SO_RCVBUF, (void*) &bufsize, len) == 0)
    { 
        return true;
    }
    else
    {
        std::cerr << "cannot set receive buf size to " << bufsize << std::endl;
        return false;
    }
}

template <typename handler>
bool linux_event_loop<handler>::set_send_buffer_size(int s, int bufsize)
{
    socklen_t len = sizeof(bufsize);
    if(setsockopt(s, SOL_SOCKET, SO_SNDBUF, (void*) &bufsize, len) == 0)
    { 
        return true;
    }
    else
    {
        std::cerr << "cannot set send buf size to " << bufsize << std::endl;
        return false;
    }
}

template <typename handler>
bool linux_event_loop<handler>::set_recv_timeout(int s, int timeout)
{
    timeval tv;
    tv.tv_sec = timeout;
    tv.tv_usec = 0;
    if(setsockopt(s, SOL_SOCKET, SO_RCVTIMEO,(void *)&tv, sizeof(tv)) == 0)
        return true;
    else
        return false;
}

template <typename handler>
bool linux_event_loop<handler>::mc_join(int s, struct addrinfo *ailist)
{
    int err;
    struct group_req req;
    req.gr_interface = 0;
    memcpy( &req.gr_group, ailist->ai_addr, ailist->ai_addrlen);
    if(err = setsockopt(s, IPPROTO_IP, MCAST_JOIN_GROUP, &req, sizeof(struct group_req)) <  0)
    {
        std::cerr << "cannot join multicast group: " << strerror(err) << std::endl;
        return false;
    }
}

template <typename handler>
int linux_event_loop<handler>::set_nonblock(int fd)
{
    int flags = fcntl(fd, F_GETFL);
    if(flags == -1)
    {
        std::cerr << "cannot read fcntl " << std::endl;
        return -1;
    }
    flags |= O_NONBLOCK;
    if(fcntl(fd, F_SETFL ,flags) == -1)
    {
        std::cerr << "cannot set fcntl " << std::endl;
        return -1;
    }
    return 0;
}

template <typename handler>
bool linux_event_loop<handler>::set_kernel_ts(int fd)
{  
    int on = 1;
    if(setsockopt(fd, SOL_SOCKET, SO_TIMESTAMP,(void *)&on, sizeof(on)) == 0)
        return true;
    else
    {
        std::cerr << "cannot set kernel ts" << std::endl;
        return false;
    }
}
