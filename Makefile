# Makefile - Puzzle #71 CUDA Searcher
# Compatível com CUDA 12.0, RTX 3050 (Ampere = sm_86)

NVCC     = nvcc
TARGET   = puzzle71

# sm_86 = RTX 3050 (Ampere)
# Se der erro, tente sm_75 (Turing) ou sm_80 (A100)
ARCH     = -gencode arch=compute_86,code=sm_86

# Flags de otimização
NVCCFLAGS = $(ARCH) \
            -O3 \
            -use_fast_math \
            -Xcompiler -O3 \
            -Xcompiler -Wall \
            --expt-relaxed-constexpr \
            -lineinfo

# Compilação: junta kernel.cu e main.cpp em um único binário
all:
	$(NVCC) $(NVCCFLAGS) kernel.cu main.cpp -o $(TARGET)
	@echo ""
	@echo "  ✓ Compilado com sucesso: ./$(TARGET)"
	@echo ""

clean:
	rm -f $(TARGET) *.o puzzle71.ckpt FOUND_KEY.txt

.PHONY: all clean
