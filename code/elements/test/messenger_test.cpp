#include "gtest/gtest.h"
#include "messenger.h"
#include "pb_writer.h"
#include "easy_time.h"
#include "tutorial.pb.h"
#include "circular_fifo.h"

TEST(messenger, test1)
{
    using namespace elements;
    using namespace std;

    typedef messenger < tutorial::AddressBook, circular_fifo<tutorial::AddressBook>, pb_text_file_writer<tutorial::AddressBook> > messenger_type;

    pb_text_file_writer<tutorial::AddressBook> writer("test_messenger_text", true);
    
    messenger_type m (10000000, writer);
    
    message_router< tutorial::AddressBook, messenger_type > router;
    
    router.add( &m );
    router.run(); 

    std::string name = "John Smith";
    std::string email = "john.smith@gmail.com";
    std::string phone_num = "555-5555-22-11";

    tutorial::AddressBook address_book;
    for( int i = 0; i < 100000; ++i)
    {
        address_book.Clear();
        auto person = address_book.add_person();
        person->set_id(1);
        person->set_name(name);
        person->set_email(email);
        auto phone_number = person->add_phone();
        phone_number->set_type(tutorial::Person::MOBILE);
        phone_number->set_number(phone_num.c_str());
        auto phone_number2 = person->add_phone();
        phone_number2->set_type(tutorial::Person::MOBILE);
        phone_number2->set_number(phone_num.c_str());
        m.send(address_book);
        if( i == 50000 )
        {
            router.stop();
            break;
        }
    }
}


TEST(messenger, test2)
{
    using namespace elements;
    using namespace std;

    typedef messenger < tutorial::AddressBook, circular_fifo<tutorial::AddressBook>, pb_binary_file_writer<tutorial::AddressBook> > messenger_type;

    pb_binary_file_writer<tutorial::AddressBook> writer("test_messenger_bin");
    
    messenger_type m (10000000, writer);
    
    message_router< tutorial::AddressBook, messenger_type > router;
    
    router.add( &m );
    router.run(); 

    std::string name = "John Smith";
    std::string email = "john.smith@gmail.com";
    std::string phone_num = "555-5555-22-11";

    tutorial::AddressBook address_book;
    for( int i = 0; i < 100000; ++i)
    {
        address_book.Clear();
        auto person = address_book.add_person();
        person->set_id(1);
        person->set_name(name);
        person->set_email(email);
        auto phone_number = person->add_phone();
        phone_number->set_type(tutorial::Person::MOBILE);
        phone_number->set_number(phone_num.c_str());
        m.send(address_book);
        if( i == 50000 )
        {
            router.stop();
            break;
        }
    }
}




