
project('elements_test', 'cpp')

add_global_arguments('-std=gnu++11', language : 'cpp')

run_command('rm','*.pb.*')
run_command('protoc','-I=.','--cpp_out=.', 'tutorial.proto')
run_command('protoc','-I=.','--cpp_out=.', 'params.proto')

src  = [
    '../easy_time.cpp',
    'circular_fifo_test.cpp',
    'easy_time_test.cpp',
    'linux_event_loop_test.cpp',
    'messenger_test.cpp',
    'out_of_band_logger_test.cpp',
    'tcp_buffer_test.cpp',
    'tutorial.pb.cc',
    'params.pb.cc',
    ] 

incdir = include_directories([  
        '../', 
        '/opt/toolchain/gtest-1.7.0/include',
        '/opt/toolchain/boost_1_58_0',
        '/opt/toolchain/protobuf-2.6.1/usr/include'
    ])

link = [
    '-L/opt/toolchain/gtest-1.7.0/lib/.libs', 
    '-L/opt/toolchain/boost_1_58_0/',
    '-L/opt/toolchain/protobuf-2.6.1/usr/lib',
    '-lgtest',
    '-lgtest_main',
    '-lpthread',
    '-lprotobuf',
    '-lrt'
]

executable(
    'elements_test',
    sources: src,
    include_directories: incdir,
    link_args: link
    )
