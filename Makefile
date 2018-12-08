NVCC=nvcc
NVCCOPTIONS=-arch=sm_30 -std=c++14 -I./src/cutf -I./src/cxxopts/include
TARGET=atsukan

$(TARGET): src/main.cu
	$(NVCC) $(NVCCOPTIONS) -o $@ $+

clean:
	rm -f $(TARGET)
