
# Makefile - Puzzle #71 CUDA Searcher
NVCC     = nvcc
TARGET   = puzzle71

ARCH     = -gencode arch=compute_86,code=sm_86

NVCCFLAGS = $(ARCH) \
            -O3 \
            -use_fast_math \
            -std=c++14 \
            -Xcompiler -O3 \
            -diag-suppress 177 \
            -lineinfo

all:
	$(NVCC) $(NVCCFLAGS) kernel.cu main.cpp -o $(TARGET)
	@echo ""
	@echo "  OK Compilado com sucesso: ./$(TARGET)"
	@echo ""

clean:
	rm -f $(TARGET) *.o puzzle71.ckpt FOUND_KEY.txt

.PHONY: all clean
