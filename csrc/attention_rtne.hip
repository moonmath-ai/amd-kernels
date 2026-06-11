// Round mode shim: RTNE (round-ties-to-nearest-even)
// Compiles attention_kernel.hip with BF16_ROUND=RTNE and renames the extern "C" symbols
// so all three round modes can coexist in the same extension.

#define BF16_ROUND RTNE
#define launch_attention_forward launch_attention_forward_rtne
#define launch_v_transpose launch_v_transpose_rtne

#include "attention_kernel.hip"
