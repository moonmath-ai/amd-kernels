// Round mode shim: RTZ (round-toward-zero / truncate)
// Compiles attention_kernel.hip with BF16_ROUND=RTZ and renames the extern "C" symbols
// so all three round modes can coexist in the same extension.

#define BF16_ROUND RTZ
#define launch_attention_forward launch_attention_forward_rtz
#define launch_attention_forward_lite launch_attention_forward_lite_rtz
#define launch_v_transpose launch_v_transpose_rtz

#include "attention_kernel.hip"
