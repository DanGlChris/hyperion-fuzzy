CXX = g++
CXXFLAGS = -O3 -std=c++11 -fPIC
LDFLAGS = -shared

all: distance.so fuzzy_contribution.so optimize_hypersphere.so

fuzzy_contribution.so: fuzzy_contribution.cpp
        $(CXX) $(CXXFLAGS) $(LDFLAGS) -o ../build/fuzzy_contribution.so fuzzy_contribution.cpp

optimize_hypersphere.so: optimize_hypersphere.cpp
        $(CXX) $(CXXFLAGS) $(LDFLAGS) -o ../build/optimize_hypersphere.so optimize_hypersphere.cpp

clean:
        rm -f ../build/*.so