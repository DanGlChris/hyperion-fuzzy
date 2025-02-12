CXX = g++
CXXFLAGS = -std=c++17 -fPIC -O2
LDFLAGS = -shared

SRCS = optimize_hypersphere.cpp fuzzy_contribution.cpp
OBJS = $(SRCS:.cpp=.o)
TARGETS = ../build/optimize_hypersphere.dll ../build/fuzzy_contribution.dll

all: $(TARGETS)

../build/optimize_hypersphere.dll: optimize_hypersphere.o
	$(CXX) $(LDFLAGS) -o $@ $^

../build/fuzzy_contribution.dll: fuzzy_contribution.o
	$(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGETS)

.PHONY: all clean