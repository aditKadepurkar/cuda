# Compiler and flags
NVCC = nvcc
CXXFLAGS = -O2 -std=c++11

# Source files and executable
SRC = vector_add.cu
OBJ = $(SRC:.cu=.o)
EXEC = vector_add

# Default target: build the executable
all: $(EXEC)

# Link object files to create the executable
$(EXEC): $(OBJ)
	$(NVCC) $(CXXFLAGS) -o $@ $^

# Compile source files into object files
%.o: %.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# Clean up compiled files
clean:
	rm -f $(OBJ) $(EXEC)

# Phony targets
.PHONY: all clean
