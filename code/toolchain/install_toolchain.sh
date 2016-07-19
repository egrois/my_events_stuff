#!/bin/bash

mkdir /opt/toolchain
cd /opt/toolchain

#anaconda
wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda-2.3.0-Linux-x86_64.sh
./Anaconda-2.3.0-Linux-x86_64.sh -b -p /opt/toolchain/anaconda
cd /opt/toolchain

#boost
wget http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.bz2
tar --bzip2 -xf boost_1_58_0.tar.bz2
chmod a+rx boost_1_58_0
cd boost_1_58_0
sudo find . -type d -exec chmod +rx {} \;
cd /opt/toolchain

#gtest
wget http://googletest.googlecode.com/files/gtest-1.7.0.zip
unzip gtest-1.7.0.zip
cd gtest-1.7.0
./configure
make
cd /opt/toolchain

#Python3
wget https://www.python.org/ftp/python/3.4.2/Python-3.4.2.tgz
tar -zxvf Python-3.4.2.tgz
./configure --prefix=/opt/toolchain/Python-3.4.2/usr
make && make install
cd /opt/toolchain

#ninja
git clone git://github.com/martine/ninja.git && cd ninja
git checkout release
./configure.py --bootstrap
cd /opt/toolchain

#meson
wget https://github.com/mesonbuild/meson/releases/download/0.25.0/meson_0.25.0.tar.gz
tar -zxvf meson_0.25.0.tar.gz
cd /opt/toolchain/meson-0.25.0
export PATH=$PATH:/opt/toolchain/Python-3.4.2/usr/bin:/opt/toolchain/ninja
./run_tests.py
./install_meson.py --prefix /opt/toolchain/meson-0.25.0/usr
cd /opt/toolchain

#gcc
wget http://mirrors.concertpass.com/gcc/releases/gcc-4.9.3/gcc-4.9.3.tar.bz2
tar --bzip2 -xf gcc-4.9.3.tar.bz2
cd gcc-4.9.3
mkdir usr
./contrib/download_prerequisites
mkdir objdir
cd objdir
/opt/toolchain/gcc-4.9.3/configure --prefix=/opt/toolchain/gcc-4.9.3/usr --disable-multilib
make -j 8 
make install
cd /opt/toolchain

#protobuf
wget https://github.com/google/protobuf/releases/download/v2.6.1/protobuf-2.6.1.tar.gz
tar -zxvf protobuf-2.6.1.tar.gz
./configure --prefix=/opt/toolchain/prottobuf-2.6.1/usr
make
make install
cd /opt/toolchain





