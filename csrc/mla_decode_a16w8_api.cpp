// Torch binding for the A16W8 absorbed-decode MLA kernel (mla_decode_a16w8.hip).
//
// Dense batch: every request decodes the same number of draft positions, so q_lat/q_pe/o_lat are
// [B*q_len, H, *] and lse is [B*q_len, H], with request b owning rows [b*q_len, (b+1)*q_len).
// q_len is inferred from the row count and B = kv_indptr.numel() - 1. q_len 1 is plain decode.
//
// DCP is optional. With cp_world > 1 the KV pool is position-sharded: global position p lives on
// rank p % cp_world at pool row p // cp_world. Q is head-replicated, so this rank runs all H heads
// over its share of the positions and returns a rank-local partial plus the base-2 LSE that weights
// it; `mla_dcp_lse_merge_ranks` combines the ranks after the all-to-all.
//
// Domain: H 1..128, any q_len, B * ceil(q_len*H/96) <= 304 row slices. Graph-capture safe: the
// launch is planned from tensor shapes, never from a host read of a device tensor. Paged only.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <stdexcept>
#include <string>

extern "C" {
size_t mla_decode_a16w8_workspace_for(int B, int q_len, int H);
int launch_mla_decode_a16w8(
    const void* q_lat, const void* q_pe, const void* kv_pool, void* o_lat, void* lse,
    const void* seq_lens, const void* kv_indices, const void* kv_indptr, const void* glen,
    int cp_rank, int cp_world, int B, int q_len, int H, int lat_dim, int rope_dim,
    float scale, float kv_scale, void* workspace, void* stream, size_t kv_bytes, int kv_len);
int launch_mla_dcp_lse_merge_ranks(
    const void* parts_in, const void* lse_in, void* out, void* lse_out,
    int R, int T, int HL, void* stream);
}

namespace moonmath_mla_a16w8 {
namespace {

constexpr int64_t kLat = 512, kRope = 64, kFused = kLat + kRope;   // 576
constexpr int64_t kMaxHeads = 128;

void check_i32dev(const at::Tensor& t, const char* who, const char* name, int64_t min_numel) {
  if (t.scalar_type() != at::kInt || !t.is_contiguous() || t.numel() < min_numel ||
      !t.device().is_cuda())
    throw std::invalid_argument(std::string(who) + ": " + name +
                                " must be a contiguous int32 device tensor with enough entries");
}

// q_lat [T,H,512], q_pe [T,H,64] and o_lat [T,H,512] bf16, all contiguous. Returns T.
int64_t check_q(const at::Tensor& q_lat, const at::Tensor& q_pe, const at::Tensor& o_lat,
                const char* who) {
  if (q_lat.dim() != 3 || q_pe.dim() != 3 || o_lat.dim() != 3)
    throw std::invalid_argument(std::string(who) + ": q_lat/q_pe/o_lat must be [T,H,*] rank 3");
  if (q_lat.scalar_type() != at::kBFloat16 || q_pe.scalar_type() != at::kBFloat16 ||
      o_lat.scalar_type() != at::kBFloat16)
    throw std::invalid_argument(std::string(who) + ": q_lat/q_pe/o_lat must be bfloat16");
  if (!q_lat.device().is_cuda())
    throw std::invalid_argument(std::string(who) + ": tensors must be on a CUDA/HIP device");
  if (q_lat.size(2) != kLat || q_pe.size(2) != kRope || o_lat.size(2) != kLat)
    throw std::invalid_argument(std::string(who) + ": expected q_lat[...,512], q_pe[...,64], o_lat[...,512]");
  if (q_lat.size(1) < 1 || q_lat.size(1) > kMaxHeads)
    throw std::invalid_argument(std::string(who) + ": H must be in [1,128]");
  if (q_pe.size(0) != q_lat.size(0) || q_pe.size(1) != q_lat.size(1) ||
      o_lat.size(0) != q_lat.size(0) || o_lat.size(1) != q_lat.size(1))
    throw std::invalid_argument(std::string(who) + ": q_pe/o_lat must share q_lat's [T,H] prefix");
  if (!q_lat.is_contiguous() || !q_pe.is_contiguous() || !o_lat.is_contiguous())
    throw std::invalid_argument(std::string(who) + ": q_lat/q_pe/o_lat must be contiguous");
  return q_lat.size(0);
}

// Decode: kv_pool is the fused [num_slots, 1, 576] fp8-e4m3fnuz pool. Without DCP, seq_lens [B]
// are the KV counts including the draft window. Under DCP they are the rank-local counts and glen
// [B] the global ones that fix the causal limit. Writes normalized o_lat, plus the base-2 LSE when
// given one.
void mla_decode_a16w8_op(const at::Tensor& q_lat, const at::Tensor& q_pe, const at::Tensor& kv_pool,
                         at::Tensor& o_lat, const at::Tensor& seq_lens, const at::Tensor& kv_indices,
                         const at::Tensor& kv_indptr, double scale, double kv_scale,
                         const c10::optional<at::Tensor>& lse, const c10::optional<at::Tensor>& glen,
                         int64_t cp_rank, int64_t cp_world) {
  const char* who = "mla_decode_a16w8";
  const int64_t T = check_q(q_lat, q_pe, o_lat, who), H = q_lat.size(1);

  if (kv_pool.scalar_type() != at::kFloat8_e4m3fnuz || !kv_pool.is_contiguous() ||
      kv_pool.size(-1) != kFused)
    throw std::invalid_argument(std::string(who) + ": kv_pool must be contiguous fp8-e4m3fnuz [.., 576]");
  check_i32dev(kv_indptr, who, "kv_indptr", 2);
  const int64_t B = kv_indptr.numel() - 1;
  if (B < 1 || T % B != 0)
    throw std::invalid_argument(std::string(who) + ": T must be B*q_len with B = kv_indptr.numel()-1");
  const int64_t q_len = T / B;
  check_i32dev(seq_lens, who, "seq_lens", B);
  check_i32dev(kv_indices, who, "kv_indices", 1);
  if (cp_world < 1 || cp_rank < 0 || cp_rank >= cp_world)
    throw std::invalid_argument(std::string(who) + ": need 0 <= cp_rank < cp_world");
  if (glen.has_value())
    check_i32dev(*glen, who, "glen", B);
  else if (cp_world > 1)
    throw std::invalid_argument(std::string(who) + ": glen (the GLOBAL lengths) is required when cp_world > 1");
  if (lse.has_value() &&
      (lse->dim() != 2 || lse->scalar_type() != at::kFloat || !lse->is_contiguous() ||
       !lse->device().is_cuda() || lse->size(0) != T || lse->size(1) != H))
    throw std::invalid_argument(std::string(who) + ": lse must be a contiguous fp32 device tensor [T,H]");

  const c10::cuda::CUDAGuard g(q_lat.device());
  const auto stream = (void*)at::cuda::getCurrentCUDAStream(q_lat.device().index()).stream();
  const size_t ws_bytes = mla_decode_a16w8_workspace_for((int)B, (int)q_len, (int)H);
  auto ws = at::empty({(int64_t)ws_bytes}, q_lat.options().dtype(at::kByte));
  // The kernel always writes an LSE; a caller that does not want one gets a scratch buffer.
  auto lse_buf = lse.has_value() ? *lse : at::empty({T, H}, q_lat.options().dtype(at::kFloat));

  // kv_indices is [sum(seq_lens)], so its shape gives the mean KV length per request. That is
  // host-side metadata, not a device read, so passing it to the planner keeps the launch graph-safe.
  const int rc = launch_mla_decode_a16w8(
      q_lat.data_ptr(), q_pe.data_ptr(), kv_pool.data_ptr(), o_lat.data_ptr(), lse_buf.data_ptr(),
      seq_lens.data_ptr(), kv_indices.data_ptr(), kv_indptr.data_ptr(),
      glen.has_value() ? glen->data_ptr() : nullptr,
      (int)cp_rank, (int)cp_world, (int)B, (int)q_len, (int)H, (int)kLat, (int)kRope,
      (float)scale, (float)kv_scale, ws.data_ptr(), stream, kv_pool.nbytes(),
      (int)(kv_indices.numel() / B));
  if (rc != 0)
    throw std::runtime_error(std::string(who) + " returned error code " + std::to_string(rc));
}

// Cross-rank merge, after the DCP all-to-all. Consumes the per-rank (normalized partial, base-2
// lse) pairs stacked over R ranks and writes this rank's owned H_local heads:
//   parts_in [R,T,HL,512] bf16, lse_in [R,T,HL] fp32, out [T,HL,512] bf16, lse_out [T,HL] fp32|None.
void mla_dcp_lse_merge_ranks_op(const at::Tensor& parts_in, const at::Tensor& lse_in,
                                at::Tensor& out, c10::optional<at::Tensor>& lse_out) {
  const char* who = "mla_dcp_lse_merge_ranks";
  if (parts_in.dim() != 4 || parts_in.scalar_type() != at::kBFloat16 || !parts_in.is_contiguous())
    throw std::invalid_argument(std::string(who) + ": parts_in must be contiguous bf16 [R,T,HL,512]");
  if (lse_in.dim() != 3 || lse_in.scalar_type() != at::kFloat || !lse_in.is_contiguous())
    throw std::invalid_argument(std::string(who) + ": lse_in must be contiguous fp32 [R,T,HL]");
  const int64_t R = parts_in.size(0), T = parts_in.size(1), HL = parts_in.size(2);
  if (parts_in.size(3) != kLat || lse_in.size(0) != R || lse_in.size(1) != T || lse_in.size(2) != HL)
    throw std::invalid_argument(std::string(who) + ": parts_in [R,T,HL,512] and lse_in [R,T,HL] must agree");
  if (R < 1 || R > 16)
    throw std::invalid_argument(std::string(who) + ": R must be in [1,16]");
  if (out.dim() != 3 || out.scalar_type() != at::kBFloat16 || !out.is_contiguous() ||
      out.size(0) != T || out.size(1) != HL || out.size(2) != kLat)
    throw std::invalid_argument(std::string(who) + ": out must be contiguous bf16 [T,HL,512]");
  void* lo = nullptr;
  if (lse_out.has_value()) {
    if (lse_out->dim() != 2 || lse_out->scalar_type() != at::kFloat || !lse_out->is_contiguous() ||
        lse_out->size(0) != T || lse_out->size(1) != HL)
      throw std::invalid_argument(std::string(who) + ": lse_out must be contiguous fp32 [T,HL]");
    lo = lse_out->data_ptr();
  }
  const c10::cuda::CUDAGuard g(parts_in.device());
  const auto stream = (void*)at::cuda::getCurrentCUDAStream(parts_in.device().index()).stream();
  const int rc = launch_mla_dcp_lse_merge_ranks(parts_in.data_ptr(), lse_in.data_ptr(),
                                                out.data_ptr(), lo, (int)R, (int)T, (int)HL, stream);
  if (rc != 0)
    throw std::runtime_error(std::string(who) + " returned error code " + std::to_string(rc));
}

}  // namespace

void register_pybind(pybind11::module_& m) {
  namespace py = pybind11;
  m.def("mla_decode_a16w8", &mla_decode_a16w8_op,
        "CDNA3 A16W8 absorbed-decode MLA over a dense paged batch (q_lat/o_lat [B*q_len,H,*]), any "
        "q_len, H 1..128: bf16 Q against the fused fp8 KV pool, end-aligned causal. Optional DCP: with "
        "cp_world > 1 the pool is this rank's shard (position p on rank p%cp_world, slot p//cp_world) and "
        "glen carries the global lengths. Writes normalized o_lat and, if given, the base-2 LSE [B*q_len,H]. "
        "Cuda-graph capturable.",
        py::arg("q_lat"), py::arg("q_pe"), py::arg("kv_pool"), py::arg("o_lat"), py::arg("seq_lens"),
        py::arg("kv_indices"), py::arg("kv_indptr"), py::arg("scale"), py::arg("kv_scale"),
        py::arg("lse") = py::none(), py::arg("glen") = py::none(), py::arg("cp_rank") = 0,
        py::arg("cp_world") = 1);
  m.def("mla_dcp_lse_merge_ranks", &mla_dcp_lse_merge_ranks_op,
        "Cross-rank log-sum-exp merge of per-rank (normalized partial, base-2 lse) pairs into this "
        "rank's owned heads, after the DCP all-to-all. parts_in [R,T,HL,512] bf16, lse_in [R,T,HL] "
        "fp32, out [T,HL,512] bf16, lse_out [T,HL] fp32 (optional).",
        py::arg("parts_in"), py::arg("lse_in"), py::arg("out"), py::arg("lse_out") = py::none());
}

}  // namespace moonmath_mla_a16w8
