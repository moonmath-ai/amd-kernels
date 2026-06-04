# CDNA3 attention forward kernel
#
# Build:    make                                   — builds RTNE and RTZ variants
#           make rtne                              — RTNE pack only
#           make rtz                               — RTZ pack only
#           make EXTRA_LLVM_FLAGS="--save-temps"   — keep IR/asm intermediates
#
# Run:      python runner.py                       — bench HIP RTNE/RTZ vs AITER

HIPCC ?= hipcc
ARCH = gfx942

TARGET_RTNE = libattention_rtne.so
TARGET_RTZ  = libattention_rtz.so
SRC         = attention_kernel.hip

# LiteAttention-capable build (dense bit-exact champion + cross-timestep skip).
# On the `sparsity` branch the skip kernel IS attention_kernel.hip (templated
# attention_forward<kLite>), so the lite libs build from the same source.
TARGET_LITE_RTNE = libattention_lite_rtne.so
TARGET_LITE_RTZ  = libattention_lite_rtz.so
SRC_LITE         = attention_kernel.hip

EXTRA_LLVM_FLAGS ?=
OPUS_INC = third_party/aiter/csrc/include/opus
HIPCC_FLAGS = -shared -fPIC -O3 --offload-arch=$(ARCH) \
              -ffast-math -fno-math-errno \
              -mllvm -amdgpu-early-inline-all=true \
              -I$(OPUS_INC) \
              $(EXTRA_LLVM_FLAGS)

all: $(TARGET_RTNE) $(TARGET_RTZ)

rtne: $(TARGET_RTNE)
rtz:  $(TARGET_RTZ)
lite: $(TARGET_LITE_RTNE) $(TARGET_LITE_RTZ)

$(TARGET_RTNE): $(SRC)
	$(HIPCC) $(HIPCC_FLAGS) -DBF16_ROUND=RTNE -o $@ $<

$(TARGET_RTZ): $(SRC)
	$(HIPCC) $(HIPCC_FLAGS) -DBF16_ROUND=RTZ -o $@ $<

$(TARGET_LITE_RTNE): $(SRC_LITE)
	$(HIPCC) $(HIPCC_FLAGS) -DBF16_ROUND=RTNE -o $@ $<

$(TARGET_LITE_RTZ): $(SRC_LITE)
	$(HIPCC) $(HIPCC_FLAGS) -DBF16_ROUND=RTZ -o $@ $<

clean:
	rm -f libattention*.so \
	      attention_kernel*-hip-amdgcn-amd-amdhsa-gfx942.* \
	      attention_kernel*-host-x86_64-unknown-linux-gnu.* \
	      attention_kernel*.hip-hip-amdgcn-amd-amdhsa.hipfb

.PHONY: all rtne rtz lite clean
