NVCC=nvcc
NVCCOPTIONS=-arch=sm_30 -std=c++14 -I./cutf -I./cxxopts/include
TARGET=atsukan

$(TARGET): main.cu
	$(NVCC) $(NVCCOPTIONS) -o $@ $<

clean:
	rm -f $(TARGET)
