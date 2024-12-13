# Compiler and flags
# NVCC = nvcc
# CXXFLAGS = -O2 -std=c++11

all:

# Clean up compiled files
clean:
	find . -type f ! -name 'Makefile' ! -name '*.cu' ! -name '*.md' ! -name '*.c' ! -name '*.cpp' ! -path './.git/*' -delete
	find . -type d -empty ! -path './.git/*' -delete

# Phony targets
.PHONY: all clean
