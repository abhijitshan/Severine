# Makefile for HyperTune
# Compatible with Windows (MinGW) and other platforms

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++14 -Wall

# Detect OS
ifeq ($(OS),Windows_NT)
	# Windows specific settings
	TARGET = HyperTune.exe
	RM = del /Q
	CP = copy
	OPENMP_DIR = openmp
	CXXFLAGS += -I$(OPENMP_DIR)
	LDFLAGS = -L$(OPENMP_DIR) -lomp
	# Add runtime path for DLL
	RPATH =
else
	# Unix/Mac specific settings
	TARGET = HyperTune
	RM = rm -f
	CP = cp
	# Try to use system OpenMP if available
	OPENMP_CHECK := $(shell if [ -f "/usr/local/opt/libomp/include/omp.h" ]; then echo "found"; else echo "not-found"; fi)
	ifeq ($(OPENMP_CHECK),found)
		# Use Homebrew OpenMP on macOS
		CXXFLAGS += -I/usr/local/opt/libomp/include -Xpreprocessor -fopenmp
		LDFLAGS = -L/usr/local/opt/libomp/lib -lomp
	else
		# Use local OpenMP
		OPENMP_DIR = openmp
		CXXFLAGS += -I$(OPENMP_DIR)
		LDFLAGS = -L$(OPENMP_DIR) -lomp
	endif
	# Add runtime path for dylib
	RPATH = -Wl,-rpath,$(OPENMP_DIR)
endif

# Source directory
SRC_DIR = HyperTune

# Source files
SOURCES = $(SRC_DIR)/main.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Build rules
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(RPATH)
ifeq ($(OS),Windows_NT)
	$(CP) $(OPENMP_DIR)\libomp.dll .
endif

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Clean rule
clean:
	$(RM) $(OBJECTS) $(TARGET)
ifeq ($(OS),Windows_NT)
	$(RM) libomp.dll
endif

.PHONY: all clean
