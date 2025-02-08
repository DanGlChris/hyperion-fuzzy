CXX = g++
CXXFLAGS = -shared -fPIC -O3

all: distance.so fuzzy_contribution.so optimize_hypersphere.so

distance.so: distance.cpp
        $(CXX) $(CXXFLAGS) -o ../build/distance.so distance.cpp

fuzzy_contribution.so: fuzzy_contribution.cpp
        $(CXX) $(CXXFLAGS) -o ../build/fuzzy_contribution.so fuzzy_contribution.cpp

optimize_hypersphere.so: optimize_hypersphere.cpp
        $(CXX) $(CXXFLAGS) -o ../build/optimize_hypersphere.so optimize_hypersphere.cpp