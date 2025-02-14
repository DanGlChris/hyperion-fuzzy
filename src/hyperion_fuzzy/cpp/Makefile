# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -fPIC -O2
LDFLAGS = -shared

# Source files
SRCS = optimize_hypersphere.cpp fuzzy_contribution.cpp hypersphere.cpp test.cpp
OBJS = $(SRCS:.cpp=.o)

# Target shared libraries
TARGETS = ../build/optimize_hypersphere.dll ../build/fuzzy_contribution.dll ../build/hypersphere.dll ../build/test.dll

# Default target (compile everything)
all: $(TARGETS)

# Compile optimize_hypersphere.dll
../build/optimize_hypersphere.dll: optimize_hypersphere.o
	$(CXX) $(LDFLAGS) -o $@ $^

# Compile fuzzy_contribution.dll (including hypersphere.o)
../build/fuzzy_contribution.dll: fuzzy_contribution.o
	$(CXX) $(LDFLAGS) -o $@ $^

# Compile hypersphere.dll
../build/hypersphere.dll: hypersphere.o
	$(CXX) $(LDFLAGS) -o $@ $^

# Compile test.dll
../build/test.dll: test.o
	$(CXX) $(LDFLAGS) -o $@ $^

# Compile object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up all compiled files
clean:
	rm -f $(OBJS) $(TARGETS)

.PHONY: all clean
