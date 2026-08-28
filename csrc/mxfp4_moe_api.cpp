// Torch bindings for the MXFP4 MoE GEMM kernels on CDNA3 (MI300X / gfx942):
// mxfp4_moe_gateup.hip (gate/up projection) and mxfp4_moe_down.hip (down projection).
//
// Both compute  C[m][n] = sum_k A[m][k] * dequant(B[e][k][n])  for the expert e that owns
// block m, over a token list that a Triton-style moe_align_block_size has already sorted by
// expert and padded to a multiple of block_m.
//
// WEIGHT LAYOUT IS A PRECONDITION, NOT A CONVENTION. B and Bs must be the REPACKED layouts
// produced by moonmath_amd.moe.repack_mxfp4 / repack_mxfp4_scales:
//
//     B  : [E, K/32, N, 16] uint8   nibble-relabelled, n-minor
//     Bs : [E, K/32, N]     uint8   E8M0, one byte per 32 k per column
//
// Passing stock [E, N, K/2] weights does not fail — it silently computes wrong numbers. The
// repack is a one-time transform at weight load and costs nothing at run time.
//
// Both ops run on the current stream with no host sync.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <stdexcept>
#include <string>

extern "C" {
int  moe_gateup_block_m(int num_valid_tokens, int num_active_experts);
int  moe_gateup_supports_k(int K);
void moe_gateup_launch(
    int epilogue, int num_m_blocks, int block_m,
    const void* A, const void* B, const void* Bs, void* C, const void* topk_weights,
    const void* sorted_token_ids, const void* expert_ids, const void* num_tokens_post_padded,
    int N, int K, int num_valid_tokens, int top_k, long stride_am,
    long stride_be, long stride_bkg, long stride_bn,
    long stride_se, long stride_skg, long stride_sn,
    long stride_cm, long stride_cn,
    int mul_routed_weight, float situ_beta, float situ_linear_beta, void* stream);

int  moe_down_block_m(int num_valid_tokens, int num_active_experts, int K);
int  moe_down_nt(int num_valid_tokens, int num_active_experts);
int  moe_down_n_steps(int num_m_blocks, int N, int num_cus, int block_m, int nt);
void moe_down_launch(
    int num_m_blocks, int block_m, int n_steps, int nt,
    const void* A, const void* B, const void* Bs, void* C, const void* topk_weights,
    const void* sorted_token_ids, const void* expert_ids, const void* num_tokens_post_padded,
    int N, int K, int num_valid_tokens, int top_k, long stride_am,
    long stride_be, long stride_bkg, long stride_bn,
    long stride_se, long stride_skg, long stride_sn,
    long stride_cm, long stride_cn,
    int mul_routed_weight, void* stream);
}

namespace moonmath_mxfp4_moe {
namespace {

//  The kernels only instantiate these tile heights; anything else gets no launch at all, so
//  reject it here rather than let a silent no-op reach the caller.
bool gateup_tile_ok(int bm) { return bm == 16 || bm == 32 || bm == 48; }
bool down_tile_ok(int bm)   { return bm == 16 || bm == 32 || bm == 64; }

//  K/BK for the down kernel, which is instantiated for 3, 4 and 6 only.
bool down_k_ok(int K) { const int ks = K / 128; return K % 128 == 0 && (ks == 3 || ks == 4 || ks == 6); }

void check_common(const at::Tensor& A, const at::Tensor& B, const at::Tensor& Bs,
                  const at::Tensor& C, const at::Tensor& sorted_token_ids,
                  const at::Tensor& expert_ids, const at::Tensor& num_tokens_post_padded,
                  const char* who) {
  const std::string w(who);
  if (A.scalar_type() != at::kBFloat16 || C.scalar_type() != at::kBFloat16)
    throw std::invalid_argument(w + ": A and C must be bfloat16");
  if (B.scalar_type() != at::kByte || Bs.scalar_type() != at::kByte)
    throw std::invalid_argument(w + ": B and Bs must be uint8 (the repacked MXFP4 layouts)");
  if (sorted_token_ids.scalar_type() != at::kInt || expert_ids.scalar_type() != at::kInt ||
      num_tokens_post_padded.scalar_type() != at::kInt)
    throw std::invalid_argument(
        w + ": sorted_token_ids/expert_ids/num_tokens_post_padded must be int32");
  if (!A.device().is_cuda())
    throw std::invalid_argument(w + ": tensors must be on a CUDA/HIP device");
  if (B.dim() != 4 || B.size(3) != 16)
    throw std::invalid_argument(
        w + ": B must be the repacked [E, K/32, N, 16] uint8 layout (see repack_mxfp4)");
  if (Bs.dim() != 3)
    throw std::invalid_argument(
        w + ": Bs must be the repacked [E, K/32, N] uint8 layout (see repack_mxfp4_scales)");
  if (A.dim() != 2 || C.dim() != 2)
    throw std::invalid_argument(w + ": A and C must be 2-D [rows, K] and [rows, N]");
  //  num_tokens_post_padded is read on the DEVICE by every workgroup, so it has to be a live
  //  device tensor rather than a host scalar the caller happens to know.
  if (!num_tokens_post_padded.device().is_cuda())
    throw std::invalid_argument(w + ": num_tokens_post_padded must be a device tensor");
}

// ---------------------------------------------------------------------------
//  gate/up. `epilogue` selects what the kernel writes:
//     0 (EPI_NONE)  C[m][n] = dot, optionally scaled by the routing weight
//     1 (EPI_SITU)  paired wave halves compute gate and up and apply the Kimi-K3 SituGLU,
//                   so C is INTER wide while B holds 2*INTER columns
// ---------------------------------------------------------------------------
void gateup_op(const at::Tensor& A, const at::Tensor& B, const at::Tensor& Bs, at::Tensor& C,
               const c10::optional<at::Tensor>& topk_weights,
               const at::Tensor& sorted_token_ids, const at::Tensor& expert_ids,
               const at::Tensor& num_tokens_post_padded,
               int64_t num_m_blocks, int64_t block_m, int64_t num_valid_tokens, int64_t top_k,
               int64_t epilogue, bool mul_routed_weight,
               double situ_beta, double situ_linear_beta) {
  check_common(A, B, Bs, C, sorted_token_ids, expert_ids, num_tokens_post_padded,
               "mxfp4_moe_gateup");
  if (epilogue != 0 && epilogue != 1)
    throw std::invalid_argument("mxfp4_moe_gateup: epilogue must be 0 (none) or 1 (SituGLU)");
  if (!gateup_tile_ok((int)block_m))
    throw std::invalid_argument("mxfp4_moe_gateup: block_m must be 16, 32 or 48; got " +
                                std::to_string(block_m));
  if (mul_routed_weight && !topk_weights.has_value())
    throw std::invalid_argument("mxfp4_moe_gateup: mul_routed_weight needs topk_weights");

  const int N = (int)C.size(1);
  const int K = (int)A.size(1);
  if (B.size(1) != K / 32)
    throw std::invalid_argument("mxfp4_moe_gateup: B's k-group extent must be K/32");
  //  EPI_SITU consumes gate and up from one B, so B carries 2N columns; EPI_NONE carries N.
  const int64_t want_n = (epilogue == 1) ? 2 * (int64_t)N : (int64_t)N;
  if (B.size(2) != want_n)
    throw std::invalid_argument("mxfp4_moe_gateup: B's column extent must be " +
                                std::to_string(want_n) + " for this epilogue; got " +
                                std::to_string(B.size(2)));

  const c10::cuda::CUDAGuard g(A.device());
  const auto stream = (void*)at::cuda::getCurrentCUDAStream(A.device().index()).stream();
  moe_gateup_launch(
      (int)epilogue, (int)num_m_blocks, (int)block_m,
      A.data_ptr(), B.data_ptr(), Bs.data_ptr(), C.data_ptr(),
      topk_weights.has_value() ? topk_weights->data_ptr() : nullptr,
      sorted_token_ids.data_ptr(), expert_ids.data_ptr(), num_tokens_post_padded.data_ptr(),
      N, K, (int)num_valid_tokens, (int)top_k, A.stride(0),
      B.stride(0), B.stride(1), B.stride(2),
      Bs.stride(0), Bs.stride(1), Bs.stride(2),
      C.stride(0), C.stride(1),
      mul_routed_weight ? 1 : 0, (float)situ_beta, (float)situ_linear_beta, stream);
}

// ---------------------------------------------------------------------------
//  down. `n_steps` is how many BN-wide column chunks one workgroup sweeps against its staged A
//  tile, and `nt` is column tiles per wave; both come from the planners below.
// ---------------------------------------------------------------------------
void down_op(const at::Tensor& A, const at::Tensor& B, const at::Tensor& Bs, at::Tensor& C,
             const c10::optional<at::Tensor>& topk_weights,
             const at::Tensor& sorted_token_ids, const at::Tensor& expert_ids,
             const at::Tensor& num_tokens_post_padded,
             int64_t num_m_blocks, int64_t block_m, int64_t n_steps, int64_t nt,
             int64_t num_valid_tokens, int64_t top_k, bool mul_routed_weight) {
  check_common(A, B, Bs, C, sorted_token_ids, expert_ids, num_tokens_post_padded,
               "mxfp4_moe_down");
  if (!down_tile_ok((int)block_m))
    throw std::invalid_argument("mxfp4_moe_down: block_m must be 16, 32 or 64; got " +
                                std::to_string(block_m));
  if (nt != 1 && nt != 2)
    throw std::invalid_argument("mxfp4_moe_down: nt must be 1 or 2");

  const int N = (int)C.size(1);
  const int K = (int)A.size(1);
  if (!down_k_ok(K))
    throw std::invalid_argument("mxfp4_moe_down: K must be 384, 512 or 768 (K/128 in {3,4,6}); got " +
                                std::to_string(K));
  if (B.size(1) != K / 32 || B.size(2) != N)
    throw std::invalid_argument("mxfp4_moe_down: B must be [E, K/32, N, 16] matching A's K and C's N");
  //  The paired dword store and the packed scale load both assume unit column strides, and at
  //  nt == 2 an odd N would let the last passing lane write one column past the row.
  if (nt == 2) {
    if (C.stride(1) != 1 || Bs.stride(2) != 1)
      throw std::invalid_argument(
          "mxfp4_moe_down: nt=2 needs C and Bs contiguous in n (stride 1)");
    if (N % 2 != 0)
      throw std::invalid_argument("mxfp4_moe_down: nt=2 needs an even N");
  }

  const c10::cuda::CUDAGuard g(A.device());
  const auto stream = (void*)at::cuda::getCurrentCUDAStream(A.device().index()).stream();
  moe_down_launch(
      (int)num_m_blocks, (int)block_m, (int)n_steps, (int)nt,
      A.data_ptr(), B.data_ptr(), Bs.data_ptr(), C.data_ptr(),
      topk_weights.has_value() ? topk_weights->data_ptr() : nullptr,
      sorted_token_ids.data_ptr(), expert_ids.data_ptr(), num_tokens_post_padded.data_ptr(),
      N, K, (int)num_valid_tokens, (int)top_k, A.stride(0),
      B.stride(0), B.stride(1), B.stride(2),
      Bs.stride(0), Bs.stride(1), Bs.stride(2),
      C.stride(0), C.stride(1),
      mul_routed_weight ? 1 : 0, stream);
}

// ── planners ── pure host arithmetic, no device work, safe to call at capture time.
int64_t gateup_block_m_op(int64_t num_valid_tokens, int64_t num_active_experts) {
  return moe_gateup_block_m((int)num_valid_tokens, (int)num_active_experts);
}
int64_t gateup_supports_k_op(int64_t K) { return moe_gateup_supports_k((int)K); }
int64_t down_block_m_op(int64_t num_valid_tokens, int64_t num_active_experts, int64_t K) {
  return moe_down_block_m((int)num_valid_tokens, (int)num_active_experts, (int)K);
}
int64_t down_nt_op(int64_t num_valid_tokens, int64_t num_active_experts) {
  return moe_down_nt((int)num_valid_tokens, (int)num_active_experts);
}
int64_t down_n_steps_op(int64_t num_m_blocks, int64_t N, int64_t num_cus, int64_t block_m,
                        int64_t nt) {
  return moe_down_n_steps((int)num_m_blocks, (int)N, (int)num_cus, (int)block_m, (int)nt);
}

}  // namespace

void register_pybind(pybind11::module_& m) {
  namespace py = pybind11;

  m.def("mxfp4_moe_gateup", &gateup_op,
        "CDNA3 MXFP4 MoE gate/up GEMM. B/Bs must be the REPACKED [E,K/32,N,16] / [E,K/32,N] uint8 "
        "layouts (moonmath_amd.moe.repack_mxfp4). epilogue=1 applies the Kimi-K3 SituGLU "
        "across paired wave halves, so B carries 2*N columns and C is N wide. Current stream, no "
        "host sync.",
        py::arg("A"), py::arg("B"), py::arg("Bs"), py::arg("C"),
        py::arg("topk_weights"), py::arg("sorted_token_ids"), py::arg("expert_ids"),
        py::arg("num_tokens_post_padded"), py::arg("num_m_blocks"), py::arg("block_m"),
        py::arg("num_valid_tokens"), py::arg("top_k"), py::arg("epilogue") = 0,
        py::arg("mul_routed_weight") = false, py::arg("situ_beta") = 1.0,
        py::arg("situ_linear_beta") = 1.0);

  m.def("mxfp4_moe_down", &down_op,
        "CDNA3 MXFP4 MoE down-projection GEMM. One workgroup stages its whole A tile in LDS and "
        "sweeps n_steps column chunks against it. B/Bs must be the REPACKED layouts. Current "
        "stream, no host sync.",
        py::arg("A"), py::arg("B"), py::arg("Bs"), py::arg("C"),
        py::arg("topk_weights"), py::arg("sorted_token_ids"), py::arg("expert_ids"),
        py::arg("num_tokens_post_padded"), py::arg("num_m_blocks"), py::arg("block_m"),
        py::arg("n_steps"), py::arg("nt"), py::arg("num_valid_tokens"), py::arg("top_k"),
        py::arg("mul_routed_weight") = false);

  m.def("mxfp4_moe_gateup_block_m", &gateup_block_m_op,
        "Tile height for gate/up from rows-per-expert (16, 32 or 48)",
        py::arg("num_valid_tokens"), py::arg("num_active_experts"));
  m.def("mxfp4_moe_gateup_supports_k", &gateup_supports_k_op,
        "Whether the gate/up kernel can serve this K. Its slab pipeline only "
        "balances over a whole pair, so K must be a multiple of 256.",
        py::arg("K"));
  m.def("mxfp4_moe_down_block_m", &down_block_m_op,
        "Tile height for down from rows-per-expert (16, 32 or 64), or 0 when this "
        "kernel cannot serve the shape -- route those to gate/up with EPI_NONE",
        py::arg("num_valid_tokens"), py::arg("num_active_experts"), py::arg("K"));
  m.def("mxfp4_moe_down_nt", &down_nt_op,
        "Column tiles per wave for down (1 or 2)",
        py::arg("num_valid_tokens"), py::arg("num_active_experts"));
  m.def("mxfp4_moe_down_n_steps", &down_n_steps_op,
        "Column chunks one down workgroup sweeps, sized to keep the grid a few times the CU count",
        py::arg("num_m_blocks"), py::arg("N"), py::arg("num_cus"), py::arg("block_m"),
        py::arg("nt"));
}

}  // namespace moonmath_mxfp4_moe
