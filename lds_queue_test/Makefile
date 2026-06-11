# LDS queueing test — gfx942
HIPCC ?= hipcc
ARCH  = gfx942

TARGET = liblds_queue_test.so
SRC    = kernel.hip

# -g for ATT source matching, -O3 for realistic codegen.
HIPCC_FLAGS = -shared -fPIC -O3 --offload-arch=$(ARCH) \
              -ffast-math -fno-math-errno -g

all: $(TARGET) kernel.s

$(TARGET): $(SRC)
	$(HIPCC) $(HIPCC_FLAGS) -o $@ $<

# Emit ISA for inspection (verify ds_read_b128 / b64 are emitted).
kernel.s: $(SRC)
	$(HIPCC) $(HIPCC_FLAGS) -S -o $@ $<

clean:
	rm -f $(TARGET) kernel.s *.bc *.hipi *.o *.out *.txt *.hipfb

.PHONY: all clean
