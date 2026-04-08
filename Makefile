HIPCC ?= hipcc
TARGET = libattention.so
ARCH = gfx942  # CDNA3 (MI300X)

HIPCC_FLAGS = -shared -fPIC -O2 -g --offload-arch=$(ARCH)

all: $(TARGET)

$(TARGET): attention_kernel.hip
	$(HIPCC) $(HIPCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
