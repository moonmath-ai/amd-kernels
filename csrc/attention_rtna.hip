// Round mode shim: RTNA (round-ties-away-from-zero)
// Compiles attention_kernel.hip with BF16_ROUND=RTNA and renames the extern "C" symbols
// so all three round modes can coexist in the same extension.

#define BF16_ROUND RTNA
#define launch_attention_forward launch_attention_forward_rtna
#define launch_attention_forward_lite launch_attention_forward_lite_rtna
#define launch_v_transpose launch_v_transpose_rtna

#include "attention_kernel.hip"
