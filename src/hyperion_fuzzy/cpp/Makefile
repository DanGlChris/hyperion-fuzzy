CXX = g++
CXXFLAGS = -O3 -std=c++11 -fPIC -I./dlib
LDFLAGS = -shared -L./dlib/build -ldlib

BUILD_DIR = ../build

# Check the operating system
ifeq ($(OS),Windows_NT)
    # Windows
    EXT = dll
    LDFLAGS += -Wl,--out-implib,libdlib.a
else
    # Linux
    EXT = so
endif

.PHONY: all dlib clean

all: dlib $(BUILD_DIR) fuzzy_contribution.$(EXT) optimize_hypersphere.$(EXT)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

dlib:
	git clone https://github.com/davisking/dlib.git
	cd dlib && mkdir build && cd build && cmake .. -G "Unix Makefiles" && $(MAKE)

fuzzy_contribution.$(EXT): fuzzy_contribution.cpp $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(BUILD_DIR)/fuzzy_contribution.$(EXT) fuzzy_contribution.cpp

optimize_hypersphere.$(EXT): optimize_hypersphere.cpp $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(BUILD_DIR)/optimize_hypersphere.$(EXT) optimize_hypersphere.cpp

clean:
	rm -f $(BUILD_DIR)/*.$(EXT)
	rm -rf dlib
