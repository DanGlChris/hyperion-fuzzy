# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -fPIC -O2
LDFLAGS = -shared

# Source files
SRCS = optimize_hypersphere.cpp fuzzy_contribution.cpp hypersphere.cpp test.cpp
OBJS = $(SRCS:.cpp=.o)

# Target shared libraries
TARGET_DIR = ../build
TARGETS = $(TARGET_DIR)/optimize_hypersphere.dll $(TARGET_DIR)/fuzzy_contribution.dll $(TARGET_DIR)/hypersphere.dll $(TARGET_DIR)/test.dll

# Default target (compile everything)
all: $(TARGETS)

# Create the build directory if it doesn't exist
$(TARGET_DIR):
	mkdir -p $(TARGET_DIR)

# Compile optimize_hypersphere.dll (depends on hypersphere)
$(TARGET_DIR)/optimize_hypersphere.dll: optimize_hypersphere.o hypersphere.o | $(TARGET_DIR)
	$(CXX) $(LDFLAGS) -o $@ $^

# Compile fuzzy_contribution.dll (depends on hypersphere)
$(TARGET_DIR)/fuzzy_contribution.dll: fuzzy_contribution.o hypersphere.o | $(TARGET_DIR)
	$(CXX) $(LDFLAGS) -o $@ $^

# Compile hypersphere.dll
$(TARGET_DIR)/hypersphere.dll: hypersphere.o | $(TARGET_DIR)
	$(CXX) $(LDFLAGS) -o $@ $^

# Compile test.dll (depends on hypersphere)
$(TARGET_DIR)/test.dll: test.o hypersphere.o | $(TARGET_DIR)
	$(CXX) $(LDFLAGS) -o $@ $^

# Compile object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up all compiled files
clean:
	rm -f $(OBJS) $(TARGETS)

.PHONY: all clean
