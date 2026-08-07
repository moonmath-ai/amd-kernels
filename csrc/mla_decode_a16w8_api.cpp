// Torch binding for the A16W8 16-QUERY-HEAD absorbed-decode MLA kernel
// (mla_decode_a16w8.hip — 8-wave warp-spec, CONS=NSub=4/PROD=4, TileTok=64, s_barrier handshake).
// bf16 activations against fp8 KV: Q stays bf16 (no quantization), KV is fp8 unpacked to bf16 for both
// QK and PV (v_mfma_f32_16x16x16bf16_1k). Probabilities are carried in bf16 (exact v_exp_f32 softmax).
//
// TENSOR ABI: fused fp8 e4m3-fnuz 576-wide rows
// ([...,:512]=latent, [512:576]=rope) at one per-tensor kv_scale, same contiguous [B,S,576] slab and
// same [num_slots,1,576] paged pool, same argument order.
//
// Both paths are current-stream, no host sync, cuda-graph capturable (paged takes a FIXED `parts`).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <stdexcept>
#include <string>

extern "C" {
size_t mla_decode_a16w8_workspace_bytes(void);
int launch_mla_decode_a16w8(
    const void* q_lat, const void* q_pe, const void* c_kv, const void* k_pe, void* o_lat,
    int B, int H, int S, int lat_dim, int rope_dim, float scale, float kv_scale, float scale_p,
    void* workspace, void* stream);
int launch_mla_decode_a16w8_paged_dev(
    const void* q_lat, const void* q_pe, const void* kv_pool, void* o_lat,
    const void* seq_lens, const void* rows, const void* kv_indices, const void* kv_indptr,
    int B, int H, int parts, int lat_dim, int rope_dim, float scale, float kv_scale, float scale_p,
    void* workspace, void* stream, size_t kv_bytes, int q_len);
int mla_decode_a16w8_plan_parts(int B, int H, int max_seq_len, int lat_dim);
int mla_decode_a16w8_plan_parts_q(int B, int max_seq_len, int q_len, int H);
int mla_decode_a16w8_plan_parts_capped(int B, int H, int max_seq_len, int lat_dim);
}

namespace moonmath_mla_a16w8 {
namespace {

constexpr int64_t kLat = 512, kRope = 64, kQK = kLat + kRope;   // 576

// Query layout: [B,H,*] (q_len==1) or [B,q_len,H,*] (speculative/MTP draft, q_len<=8). Returns q_len.
int check_q(const at::Tensor& q_lat, const at::Tensor& q_pe, const at::Tensor& o_lat, const char* who) {
  const auto D = q_lat.dim();
  if (D != 3 && D != 4)
    throw std::invalid_argument(std::string(who) + ": q_lat must be [B,H,512] or [B,q_len,H,512]");
  if (q_pe.dim() != D || o_lat.dim() != D)
    throw std::invalid_argument(std::string(who) + ": q_lat/q_pe/o_lat must have the same rank");
  if (q_lat.scalar_type() != at::kBFloat16 || q_pe.scalar_type() != at::kBFloat16 ||
      o_lat.scalar_type() != at::kBFloat16)
    throw std::invalid_argument(std::string(who) + ": q_lat/q_pe/o_lat must be bfloat16");
  if (!q_lat.device().is_cuda())
    throw std::invalid_argument(std::string(who) + ": tensors must be on a CUDA/HIP device");
  if (q_lat.size(D - 1) != kLat || q_pe.size(D - 1) != kRope || o_lat.size(D - 1) != kLat)
    throw std::invalid_argument(std::string(who) + ": expected q_lat[...,512], q_pe[...,64], o_lat[...,512]");
  if (q_lat.size(D - 2) < 1 || q_lat.size(D - 2) > 16)
    throw std::invalid_argument(std::string(who) + ": H must be in [1,16] (dedicated 16-head kernel)");
  const int q_len = (D == 4) ? (int)q_lat.size(1) : 1;
  if (q_len < 1 || q_len > 8)
    throw std::invalid_argument(std::string(who) + ": q_len must be in [1,8]");
  if (!q_lat.is_contiguous() || !q_pe.is_contiguous() || !o_lat.is_contiguous())
    throw std::invalid_argument(std::string(who) + ": q_lat/q_pe/o_lat must be contiguous");
  return q_len;
}

// ── CONTIGUOUS, FUSED-576 ──
// q_lat [B,H,512] bf16, q_pe [B,H,64] bf16, kv [B,S,576] fp8 e4m3-fnuz, o_lat [B,H,512] bf16.
//   kv[..., 0:512] = c_KV latent, kv[..., 512:576] = k_pe rope, BOTH fp8 at the SAME per-tensor kv_scale.
//   scale = layer.scaling (1/sqrt(qk_head_dim) x YaRN mscale). scale_p is accepted for signature symmetry
//   with the split-precision op and is unused (the probability lift is a compile-time constant here).
void mla_decode_a16w8_op(const at::Tensor& q_lat, const at::Tensor& q_pe, const at::Tensor& kv,
                            at::Tensor& o_lat, double scale, double kv_scale, double scale_p) {
  // The contiguous path has no window: only the paged launcher takes q_len.
  if (check_q(q_lat, q_pe, o_lat, "mla_decode_a16w8") != 1)
    throw std::invalid_argument("mla_decode_a16w8: the contiguous entry point is q_len==1 only; "
                                "use mla_decode_a16w8_paged_dev for a draft window");
  if (kv.scalar_type() != at::kFloat8_e4m3fnuz)
    throw std::invalid_argument("mla_decode_a16w8: kv must be float8_e4m3fnuz");
  if (kv.dim() != 3 || kv.size(2) != kQK)
    throw std::invalid_argument("mla_decode_a16w8: kv must be the FUSED [B,S,576] slab "
                                "([...,:512]=latent, [512:576]=rope), not the split c_kv/k_pe pair");
  if (kv.size(0) != q_lat.size(0))
    throw std::invalid_argument("mla_decode_a16w8: kv batch must match q_lat batch");
  if (!kv.is_contiguous())
    throw std::invalid_argument("mla_decode_a16w8: kv must be contiguous");
  const int B = (int)q_lat.size(0);
  const int H = (int)q_lat.size(1);
  const int S = (int)kv.size(1);
  const c10::cuda::CUDAGuard g(q_lat.device());
  const auto stream = (void*)at::cuda::getCurrentCUDAStream(q_lat.device().index()).stream();
  auto ws = at::empty({(int64_t)mla_decode_a16w8_workspace_bytes()}, q_lat.options().dtype(at::kByte));
  const int rc = launch_mla_decode_a16w8(
      q_lat.data_ptr(), q_pe.data_ptr(), kv.data_ptr(), /*k_pe=*/nullptr, o_lat.data_ptr(),
      B, H, S, (int)kLat, (int)kRope, (float)scale, (float)kv_scale, (float)scale_p,
      ws.data_ptr(), stream);
  if (rc != 0)
    throw std::runtime_error("launch_mla_decode_a16w8 returned error code " + std::to_string(rc));
}

// ── GRAPH-SAFE DEVICE-DRIVEN PAGED ──
// kv_pool: the layer's FUSED [num_slots, 1, 576] float8_e4m3fnuz slab at the model's per-tensor scale
//   ([...,:512]=c_KV, [512:576]=k_pe).  Request b owns the FLAT per-token slots
//   kv_indices[kv_indptr[b] : kv_indptr[b]+seq_lens[b]] (MLA page_size=1).  seq_lens[B]/kv_indices[total]/
//   kv_indptr[B+1] are int32 DEVICE tensors; rows[B] int32 device or None maps req_pool_index -> q/out batch
//   row.  `parts` is FIXED by the caller at capture (mla_decode_a16w8_plan_parts_capped).  No host read
//   and no data-dependent grid => safe inside a cuda-graph capture; update the device tensors in place on
//   replay.  kv_scale = the model's per-tensor fp8 KV descale (layer.k_scale; 1.0 for a scale-1.0 pool).
void mla_decode_a16w8_paged_dev_op(const at::Tensor& q_lat, const at::Tensor& q_pe,
                                      const at::Tensor& kv_pool, at::Tensor& o_lat,
                                      const at::Tensor& seq_lens, const c10::optional<at::Tensor>& rows,
                                      const at::Tensor& kv_indices, const at::Tensor& kv_indptr,
                                      int64_t parts, double scale, double kv_scale, double scale_p) {
  const int q_len = check_q(q_lat, q_pe, o_lat, "mla_decode_a16w8_paged_dev");
  if (kv_pool.scalar_type() != at::kFloat8_e4m3fnuz)
    throw std::invalid_argument("mla_decode_a16w8_paged_dev: kv_pool must be float8_e4m3fnuz");
  if (kv_pool.size(-1) != kQK)
    throw std::invalid_argument("mla_decode_a16w8_paged_dev: kv_pool rows must be 576 wide "
                                "([...,:512]=c_KV, [512:576]=k_pe)");
  if (seq_lens.scalar_type() != at::kInt || kv_indices.scalar_type() != at::kInt ||
      kv_indptr.scalar_type() != at::kInt)
    throw std::invalid_argument(
        "mla_decode_a16w8_paged_dev: seq_lens/kv_indices/kv_indptr must be int32");
  const int B = (int)q_lat.size(0);
  const int H = (int)q_lat.size(q_lat.dim() - 2);
  const c10::cuda::CUDAGuard g(q_lat.device());
  const auto stream = (void*)at::cuda::getCurrentCUDAStream(q_lat.device().index()).stream();
  auto ws = at::empty({(int64_t)mla_decode_a16w8_workspace_bytes()}, q_lat.options().dtype(at::kByte));
  const void* rows_p = (rows.has_value() && rows->defined()) ? rows->data_ptr() : nullptr;
  const int rc = launch_mla_decode_a16w8_paged_dev(
      q_lat.data_ptr(), q_pe.data_ptr(), kv_pool.data_ptr(), o_lat.data_ptr(),
      seq_lens.data_ptr(), rows_p, kv_indices.data_ptr(), kv_indptr.data_ptr(),
      B, H, (int)parts, (int)kLat, (int)kRope, (float)scale, (float)kv_scale, (float)scale_p,
      ws.data_ptr(), stream,
      // KV working set: one 576-B row per slot. numel() == sum(seq_lens) at page_size=1, so this is the
      //   true footprint including ragged batches, and it comes from metadata (graph-capture safe).
      (size_t)kv_indices.numel() * 576ull, q_len);
  if (rc != 0)
    throw std::runtime_error("launch_mla_decode_a16w8_paged_dev returned error code " +
                             std::to_string(rc));
}

int64_t plan_parts_op(int64_t B, int64_t H, int64_t max_seq_len, int64_t lat_dim) {
  return mla_decode_a16w8_plan_parts((int)B, (int)H, (int)max_seq_len, (int)lat_dim);
}
int64_t plan_parts_q_op(int64_t B, int64_t max_seq_len, int64_t q_len, int64_t H) {
  return mla_decode_a16w8_plan_parts_q((int)B, (int)max_seq_len, (int)q_len, (int)H);
}
int64_t plan_parts_capped_op(int64_t B, int64_t H, int64_t max_seq_len, int64_t lat_dim) {
  return mla_decode_a16w8_plan_parts_capped((int)B, (int)H, (int)max_seq_len, (int)lat_dim);
}

}  // namespace

void register_pybind(pybind11::module_& m) {
  namespace py = pybind11;
  m.def("mla_decode_a16w8", &mla_decode_a16w8_op,
        "CDNA3 A16W8 16-head absorbed-decode MLA (q_len=1): bf16 Q against fp8 KV. ONE fused fp8 "
        "e4m3-fnuz kv[B,S,576] ([...,:512]=latent, [512:576]=rope) at a single per-tensor kv_scale; Q "
        "stays bf16 (no q quantization error). Current-stream, no host sync.",
        py::arg("q_lat"), py::arg("q_pe"), py::arg("kv"), py::arg("o_lat"),
        py::arg("scale"), py::arg("kv_scale"), py::arg("scale_p"));
  m.def("mla_decode_a16w8_paged_dev", &mla_decode_a16w8_paged_dev_op,
        "CDNA3 A16W8 (bf16 Q / fp8 KV) 16-head absorbed-decode MLA, DEVICE-DRIVEN PAGED. ABI-identical "
        "paged: fused fp8 [num_slots,1,576] pool + flat kv_indices/kv_indptr + "
        "device seq_lens + FIXED parts; cuda-graph capturable; per-tensor kv_scale.",
        py::arg("q_lat"), py::arg("q_pe"), py::arg("kv_pool"), py::arg("o_lat"),
        py::arg("seq_lens"), py::arg("rows"), py::arg("kv_indices"), py::arg("kv_indptr"),
        py::arg("parts"), py::arg("scale"), py::arg("kv_scale"), py::arg("scale_p"));
  m.def("mla_decode_a16w8_plan_parts", &plan_parts_op,
        "FIXED kv-split for an a16w8 16h graph captured at (B,H,max_seq_len)",
        py::arg("B"), py::arg("H"), py::arg("max_seq_len"), py::arg("lat_dim"));
  m.def("mla_decode_a16w8_plan_parts_q", &plan_parts_q_op,
        "q_len-aware FIXED kv-split for an a16w8 graph captured at (B,max_seq_len,q_len). At QLC=1 "
        "each draft position is its own CTA, so the window multiplies the CTA count and the KV split "
        "gives parts back to stay inside the grid cap.",
        py::arg("B"), py::arg("max_seq_len"), py::arg("q_len") = 1, py::arg("H") = 16);
  m.def("mla_decode_a16w8_plan_parts_capped", &plan_parts_capped_op,
        "FIXED kv-split for an a16w8 16h graph captured at batch bs, CLAMPED so bs*parts <= MaxCTAs",
        py::arg("B"), py::arg("H"), py::arg("max_seq_len"), py::arg("lat_dim"));
}

}  // namespace moonmath_mla_a16w8
