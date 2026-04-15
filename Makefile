# CDNA3 attention forward kernel — Makefile.
#
#  Default build: single-wave HIP (`attention_kernel.hip`, 64 threads, BM=64, two Q passes).
#  Exports `launch_attention_forward` (V [seq][hd] row-major, same as Q/K).
#  Alternate sources live under temp/. Example: `cp temp/attention_kernel_174tflop.hip attention_kernel.hip && make`.
#
#  Experimental AITER wrapper is still available in-tree for reference, but it
#  is not the default build path.

HIPCC ?= hipcc
TARGET = libattention.so
TARGET_V1 = libattention_v1.so
TARGET_V2 = libattention_v2.so
TARGET_AITER = libattention_aiter.so
TARGET_32X32 = libattention_32x32.so
TARGET_OPT = libattention_opt.so
ARCH = gfx942
CK_TILE_INC = /opt/rocm-7.2.0/include

# Optional: make EXTRA_LLVM_FLAGS='-mllvm -print-after-all' all  (debug only)
EXTRA_LLVM_FLAGS ?=
HIPCC_FLAGS = -shared -fPIC -O3 --offload-arch=$(ARCH) \
              -ffast-math -fno-math-errno \
              -mllvm -amdgpu-early-inline-all=true \
              -mllvm --amdgpu-mfma-vgpr-form \
              $(EXTRA_LLVM_FLAGS)

CK_FLAGS = $(HIPCC_FLAGS) -std=c++17 -I$(CK_TILE_INC) -DCK_TILE_FMHA_FWD_FAST_EXP2=1

all: $(TARGET)

$(TARGET): attention_kernel.hip
	$(HIPCC) $(HIPCC_FLAGS) -o $(TARGET) $<

aiter: $(TARGET_AITER)

$(TARGET_AITER): attention_kernel_aiter_v3.cpp
	$(HIPCC) $(HIPCC_FLAGS) -std=c++17 -o $@ $<

32x32: $(TARGET_32X32)

$(TARGET_32X32): temp/attention_kernel_32x32.hip
	$(HIPCC) $(HIPCC_FLAGS) -o $@ $<

v1: $(TARGET_V1)

$(TARGET_V1): temp/attention_kernel_v1.hip
	$(HIPCC) $(HIPCC_FLAGS) -o $@ $<

v2: $(TARGET_V2)

$(TARGET_V2): temp/attention_kernel_v2.hip
	$(HIPCC) $(HIPCC_FLAGS) -o $@ $<

opt: $(TARGET_OPT)

$(TARGET_OPT): temp/attention_kernel_opt.hip
	$(HIPCC) $(HIPCC_FLAGS) -o $@ $<

# ── flash-attn wrapper build (`make fa`) ─────────────────────────────
FA_ENV     = /home/tarik/miniconda3/envs/lite_attn
FA_SO_DIR  = $(FA_ENV)/lib/python3.11/site-packages
TORCH_LIB  = $(FA_ENV)/lib/python3.11/site-packages/torch/lib

FA_FLAGS = -shared -fPIC -O3 --offload-arch=$(ARCH) -std=c++17 \
           -ffast-math -fno-math-errno

FA_LDFLAGS = -Wl,-rpath,$(FA_SO_DIR) -Wl,-rpath,$(TORCH_LIB) \
             -L$(FA_SO_DIR) -l:flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so \
             -L$(TORCH_LIB) -ltorch -lc10 -ltorch_cpu -ltorch_hip -lc10_hip \
             -ltorch_python

fa: temp/attention_kernel.cpp
	$(HIPCC) $(FA_FLAGS) -o $(TARGET) $< $(FA_LDFLAGS)

isa-report: attention_kernel.hip
	mkdir -p temp
	$(HIPCC) $(HIPCC_FLAGS) -save-temps -o /tmp/libattn_isa.so $<
	-mv -f attention_kernel-hip-amdgcn-amd-amdhsa-gfx942.s attention_kernel-host-x86_64-unknown-linux-gnu.s \
		attention_kernel-hip-amdgcn-amd-amdhsa-gfx942.hipi attention_kernel-host-x86_64-unknown-linux-gnu.hipi \
		attention_kernel-hip-amdgcn-amd-amdhsa-gfx942.bc attention_kernel-hip-amdgcn-amd-amdhsa-gfx942.o \
		attention_kernel-hip-amdgcn-amd-amdhsa-gfx942.out attention_kernel-hip-amdgcn-amd-amdhsa-gfx942.out.resolution.txt \
		attention_kernel.hip-hip-amdgcn-amd-amdhsa.hipfb \
		attention_kernel-host-x86_64-unknown-linux-gnu.bc attention_kernel-host-x86_64-unknown-linux-gnu.o \
		temp/ 2>/dev/null
	python3 tools/isa_report.py

aiter-asm:
	bash tools/disassemble_aiter_v3.sh

clean:
	rm -f $(TARGET) $(TARGET_V1) $(TARGET_V2) $(TARGET_AITER) $(TARGET_32X32) $(TARGET_OPT)

.PHONY: aiter all clean fa v1 v2 32x32 opt isa-report aiter-asm
