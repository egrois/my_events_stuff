#ifndef ELEMENTS_PB_WRITER_H
#define ELEMENTS_PB_WRITER_H

#include <google/protobuf/text_format.h>
#include <fstream>
#include <string>

namespace elements
{

template <typename msg_type>
class pb_text_file_writer
{
public:
    pb_text_file_writer(std::string file_name, bool is_single_line) 
        : _os( new std::ofstream( file_name, std::ios::out | std::ios::app ))
    {
        _pb_printer.SetSingleLineMode(is_single_line);
    }
    void write(msg_type &msg)
    {
        std::string s;
        _pb_printer.PrintToString(msg, &s);
        (*_os) << s << std::endl;
        
    }
private:
    google::protobuf::TextFormat::Printer _pb_printer;
    std::ofstream * _os;
};

template <typename msg_type>
class pb_binary_file_writer
{
public:
    pb_binary_file_writer(std::string file_name) 
        : _os(new std::ofstream( file_name, std::ios::out | std::ios::app |std::ios::binary ))
    {
    }
    void write(msg_type &msg)
    {
        msg.SerializeToOstream( _os ); 
    }
private:
    std::ofstream * _os;
};

}
#endif // ELEMENTS_PB_WRITER_H
