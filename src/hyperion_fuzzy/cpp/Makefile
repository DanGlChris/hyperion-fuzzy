CXX = g++
CXXFLAGS = -O3 -std=c++11 -fPIC -I./dlib
LDFLAGS = -shared -L./dlib/build -ldlib

all: dlib distance.so fuzzy_contribution.so optimize_hypersphere.so

dlib:
    git clone https://github.com/davisking/dlib.git
    cd dlib && mkdir build && cd build && cmake .. -G "Unix Makefiles" && make

fuzzy_contribution.so: fuzzy_contribution.cpp
    $(CXX) $(CXXFLAGS) $(LDFLAGS) -o ../build/fuzzy_contribution.so fuzzy_contribution.cpp

optimize_hypersphere.so: optimize_hypersphere.cpp
    $(CXX) $(CXXFLAGS) $(LDFLAGS) -o ../build/optimize_hypersphere.so optimize_hypersphere.cpp

clean:
    rm -f ../build/*.so
    rm -rf dlib