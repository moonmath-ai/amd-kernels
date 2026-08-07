// Torch binding for the A16W8 multi-query absorbed-decode MLA kernel (mla_decode_a16w8_multiq.hip:
// 8 computing waves, a bf16 LDS tile of 16 KV tokens, the fp8->bf16 unpack on the fill, 4 draft
// positions per CTA, one barrier per tile).
//
// SUPPORTED DOMAIN, matching the kernel's own contract: H 1..16, q_len 4..8, B 1..32 (more
// precisely B*ceil(q_len/4) <= 152), S >= 2048 for the tuned range. These ops reject H and q_len
// out of range here, with a message; the launchers reject B and re-check the rest by error code.
// q_len 1 is a DIFFERENT CTA shape and lives behind separate ops (mla_decode_a16w8.hip).
//
// TENSOR ABI, shared with the mla_decode_a16w8 ops: fused fp8 e4m3-fnuz 576-wide KV rows
// ([...,:512] latent, [512:576] rope) at one per-tensor kv_scale, a contiguous [B,S,576] slab or a
// [num_slots,1,576] paged pool, same argument order. Both paths are current-stream, take no host
// sync and are cuda-graph capturable (paged takes a FIXED `parts`).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <stdexcept>
#include <string>

extern "C" {
size_t mla_decode_a16w8_multiq_workspace_for(int B, int H, int parts, int q_len);
int mla_decode_a16w8_multiq_plan_parts_q(int B, int max_seq_len, int q_len, int H);
int launch_mla_decode_a16w8_multiq(
    const void* q_lat, const void* q_pe, const void* c_kv, void* o_lat,
    int B, int H, int S, int q_len, int lat_dim, int rope_dim, float scale, float kv_scale,
    void* workspace, void* stream);
int launch_mla_decode_a16w8_multiq_paged_dev(
    const void* q_lat, const void* q_pe, const void* kv_pool, void* o_lat,
    const void* seq_lens, const void* rows, const void* kv_indices, const void* kv_indptr,
    int B, int H, int parts, int q_len, int lat_dim, int rope_dim, float scale, float kv_scale,
    void* workspace, void* stream, size_t kv_bytes);
}

namespace moonmath_mla_a16w8_multiq {
namespace {

constexpr int64_t kLat = 512, kRope = 64, kQK = kLat + kRope;   // 576

// Query layout: [B,q_len,H,*], q_len in [4,8]. Returns q_len.
int check_q(const at::Tensor& q_lat, const at::Tensor& q_pe, const at::Tensor& o_lat, const char* who) {
  if (q_lat.dim() != 4)
    throw std::invalid_argument(std::string(who) + ": q_lat must be [B,q_len,H,512] (q_len 4..8)");
  if (q_pe.dim() != 4 || o_lat.dim() != 4)
    throw std::invalid_argument(std::string(who) + ": q_lat/q_pe/o_lat must have the same rank");
  if (q_lat.scalar_type() != at::kBFloat16 || q_pe.scalar_type() != at::kBFloat16 ||
      o_lat.scalar_type() != at::kBFloat16)
    throw std::invalid_argument(std::string(who) + ": q_lat/q_pe/o_lat must be bfloat16");
  if (!q_lat.device().is_cuda())
    throw std::invalid_argument(std::string(who) + ": tensors must be on a CUDA/HIP device");
  if (q_lat.size(3) != kLat || q_pe.size(3) != kRope || o_lat.size(3) != kLat)
    throw std::invalid_argument(std::string(who) + ": expected q_lat[...,512], q_pe[...,64], o_lat[...,512]");
  if (q_lat.size(2) < 1 || q_lat.size(2) > 16)
    throw std::invalid_argument(std::string(who) + ": H must be in [1,16] (one MFMA N-tile of heads)");
  const int q_len = (int)q_lat.size(1);
  if (q_len < 4 || q_len > 8)
    throw std::invalid_argument(std::string(who) + ": q_len must be in [4,8]");
  if (!q_lat.is_contiguous() || !q_pe.is_contiguous() || !o_lat.is_contiguous())
    throw std::invalid_argument(std::string(who) + ": q_lat/q_pe/o_lat must be contiguous");
  return q_len;
}

// ── CONTIGUOUS, FUSED-576 ──
// q_lat [B,q_len,H,512] bf16, q_pe [B,q_len,H,64] bf16, kv [B,S,576] fp8 e4m3-fnuz, o_lat like q_lat.
//   kv[..., 0:512] = c_KV latent, kv[..., 512:576] = k_pe rope, BOTH fp8 at the SAME per-tensor kv_scale.
//   scale = layer.scaling (1/sqrt(qk_head_dim) x YaRN mscale). scale_p is accepted for signature symmetry
//   with the split-precision op and is unused: the probability lift is a compile-time constant here.
void mla_decode_a16w8_multiq_op(const at::Tensor& q_lat, const at::Tensor& q_pe, const at::Tensor& kv,
                            at::Tensor& o_lat, double scale, double kv_scale, double scale_p) {
  (void)scale_p;
  const int q_len = check_q(q_lat, q_pe, o_lat, "mla_decode_a16w8_multiq");
  if (kv.scalar_type() != at::kFloat8_e4m3fnuz)
    throw std::invalid_argument("mla_decode_a16w8_multiq: kv must be float8_e4m3fnuz");
  if (kv.dim() != 3 || kv.size(2) != kQK)
    throw std::invalid_argument("mla_decode_a16w8_multiq: kv must be the FUSED [B,S,576] slab "
                                "([...,:512]=latent, [512:576]=rope), not the split c_kv/k_pe pair");
  if (kv.size(0) != q_lat.size(0))
    throw std::invalid_argument("mla_decode_a16w8_multiq: kv batch must match q_lat batch");
  if (!kv.is_contiguous())
    throw std::invalid_argument("mla_decode_a16w8_multiq: kv must be contiguous");
  const int B = (int)q_lat.size(0);
  const int H = (int)q_lat.size(2);
  const int S = (int)kv.size(1);
  const c10::cuda::CUDAGuard g(q_lat.device());
  const auto stream = (void*)at::cuda::getCurrentCUDAStream(q_lat.device().index()).stream();
  const int parts = mla_decode_a16w8_multiq_plan_parts_q(B, S, q_len, H);
  auto ws = at::empty({(int64_t)mla_decode_a16w8_multiq_workspace_for(B, H, parts, q_len)},
                      q_lat.options().dtype(at::kByte));
  const int rc = launch_mla_decode_a16w8_multiq(
      q_lat.data_ptr(), q_pe.data_ptr(), kv.data_ptr(), o_lat.data_ptr(),
      B, H, S, q_len, (int)kLat, (int)kRope, (float)scale, (float)kv_scale,
      ws.data_ptr(), stream);
  if (rc != 0)
    throw std::runtime_error("launch_mla_decode_a16w8_multiq returned error code " + std::to_string(rc));
}

// ── GRAPH-SAFE DEVICE-DRIVEN PAGED ── ABI-identical to mla_decode_a16w8_paged_dev.
// kv_pool: the layer's FUSED [num_slots, 1, 576] float8_e4m3fnuz slab at the model's per-tensor scale
//   ([...,:512]=c_KV, [512:576]=k_pe).  Request b owns the FLAT per-token slots
//   kv_indices[kv_indptr[b] : kv_indptr[b]+seq_lens[b]] (MLA page_size=1).  seq_lens[B]/kv_indices[total]/
//   kv_indptr[B+1] are int32 DEVICE tensors; rows[B] int32 device or None maps req_pool_index -> q/out
//   batch row.  `parts` is FIXED by the caller at capture (mla_decode_a16w8_multiq_plan_parts_q).  No host
//   read and no data-dependent grid => safe inside a cuda-graph capture; update the device tensors in
//   place on replay.  kv_scale = the model's per-tensor fp8 KV descale (layer.k_scale).
void mla_decode_a16w8_multiq_paged_dev_op(const at::Tensor& q_lat, const at::Tensor& q_pe,
                                      const at::Tensor& kv_pool, at::Tensor& o_lat,
                                      const at::Tensor& seq_lens, const c10::optional<at::Tensor>& rows,
                                      const at::Tensor& kv_indices, const at::Tensor& kv_indptr,
                                      int64_t parts, double scale, double kv_scale, double scale_p) {
  (void)scale_p;
  const int q_len = check_q(q_lat, q_pe, o_lat, "mla_decode_a16w8_multiq_paged_dev");
  if (kv_pool.scalar_type() != at::kFloat8_e4m3fnuz)
    throw std::invalid_argument("mla_decode_a16w8_multiq_paged_dev: kv_pool must be float8_e4m3fnuz");
  if (kv_pool.size(-1) != kQK)
    throw std::invalid_argument("mla_decode_a16w8_multiq_paged_dev: kv_pool rows must be 576 wide "
                                "([...,:512]=c_KV, [512:576]=k_pe)");
  if (!kv_pool.is_contiguous())
    throw std::invalid_argument("mla_decode_a16w8_multiq_paged_dev: kv_pool must be contiguous");
  if (seq_lens.scalar_type() != at::kInt || kv_indices.scalar_type() != at::kInt ||
      kv_indptr.scalar_type() != at::kInt)
    throw std::invalid_argument(
        "mla_decode_a16w8_multiq_paged_dev: seq_lens/kv_indices/kv_indptr must be int32");
  const int B = (int)q_lat.size(0);
  const int H = (int)q_lat.size(2);
  // Metadata extents. All SHAPE queries -- host side, no device read -- so they are free and
  //   graph-capture safe. The kernel indexes seq_lens[b] and kv_indptr[b+1] for every b < B.
  if (seq_lens.numel() < B || !seq_lens.is_contiguous())
    throw std::invalid_argument("mla_decode_a16w8_multiq_paged_dev: seq_lens must be contiguous with >= B entries");
  if (kv_indptr.numel() < (int64_t)B + 1 || !kv_indptr.is_contiguous())
    throw std::invalid_argument("mla_decode_a16w8_multiq_paged_dev: kv_indptr must be contiguous with >= B+1 entries");
  if (!kv_indices.is_contiguous())
    throw std::invalid_argument("mla_decode_a16w8_multiq_paged_dev: kv_indices must be contiguous");
  // `rows` maps KV-request b -> q/out batch row. It addresses the OUTPUT STORE as well as the query
  //   load, so a short tensor is an out-of-bounds device read that decides where the kernel writes.
  //   The VALUES cannot be checked without a host sync, which would break graph capture, so the
  //   kernel clamps them into [0,B); this check covers the extent, which is knowable for free.
  if (rows.has_value() && rows->defined()) {
    if (rows->scalar_type() != at::kInt)
      throw std::invalid_argument("mla_decode_a16w8_multiq_paged_dev: rows must be int32");
    if (rows->numel() < B || !rows->is_contiguous())
      throw std::invalid_argument("mla_decode_a16w8_multiq_paged_dev: rows must be contiguous with >= B entries");
    if (!rows->device().is_cuda())
      throw std::invalid_argument("mla_decode_a16w8_multiq_paged_dev: rows must be on the device");
  }
  // `parts` is a pure performance knob -- every value >= 1 is numerically correct -- and the launcher
  //   clamps it, but it arrives as an unvalidated int64: reject values that would overflow the int the
  //   launcher and the workspace sizer both use. Values <= 0 keep their meaning (clamped to 1).
  if (parts > ((int64_t)1 << 20))
    throw std::invalid_argument("mla_decode_a16w8_multiq_paged_dev: parts is absurd (> 2^20)");
  const c10::cuda::CUDAGuard g(q_lat.device());
  const auto stream = (void*)at::cuda::getCurrentCUDAStream(q_lat.device().index()).stream();
  auto ws = at::empty({(int64_t)mla_decode_a16w8_multiq_workspace_for(B, H, (int)parts, q_len)},
                      q_lat.options().dtype(at::kByte));
  const void* rows_p = (rows.has_value() && rows->defined()) ? rows->data_ptr() : nullptr;
  const int rc = launch_mla_decode_a16w8_multiq_paged_dev(
      q_lat.data_ptr(), q_pe.data_ptr(), kv_pool.data_ptr(), o_lat.data_ptr(),
      seq_lens.data_ptr(), rows_p, kv_indices.data_ptr(), kv_indptr.data_ptr(),
      B, H, (int)parts, q_len, (int)kLat, (int)kRope, (float)scale, (float)kv_scale,
      ws.data_ptr(), stream,
      // KV working set: one 576-B row per slot. numel() == sum(seq_lens) at page_size=1, so this is
      //   the true footprint including ragged batches, and it comes from metadata (capture safe).
      (size_t)kv_indices.numel() * 576ull);
  if (rc != 0)
    throw std::runtime_error("launch_mla_decode_a16w8_multiq_paged_dev returned error code " +
                             std::to_string(rc));
}

// `parts` is fixed at graph capture and must leave room for the QGroups = q_len/4 CTAs that share a
//   KV stream. H is accepted for ABI symmetry and no longer shapes the grid (H <= 16 is one tile).
int64_t plan_parts_q_op(int64_t B, int64_t max_seq_len, int64_t q_len, int64_t H) {
  return mla_decode_a16w8_multiq_plan_parts_q((int)B, (int)max_seq_len, (int)q_len, (int)H);
}

}  // namespace

void register_pybind(pybind11::module_& m) {
  namespace py = pybind11;
  m.def("mla_decode_a16w8_multiq", &mla_decode_a16w8_multiq_op,
        "CDNA3 A16W8 multi-query absorbed-decode MLA, q_len 4..8 (bf16 Q against fp8 KV). ONE fused fp8 "
        "e4m3-fnuz kv[B,S,576] ([...,:512]=latent, [512:576]=rope) at a single per-tensor kv_scale. "
        "q_len 1 is a different CTA shape and is not served here. Current-stream, no host sync.",
        py::arg("q_lat"), py::arg("q_pe"), py::arg("kv"), py::arg("o_lat"),
        py::arg("scale"), py::arg("kv_scale"), py::arg("scale_p"));
  m.def("mla_decode_a16w8_multiq_paged_dev", &mla_decode_a16w8_multiq_paged_dev_op,
        "CDNA3 A16W8 (bf16 Q / fp8 KV) multi-query absorbed-decode MLA, q_len 4..8, DEVICE-DRIVEN "
        "PAGED. Fused fp8 [num_slots,1,576] pool + flat "
        "kv_indices/kv_indptr + device seq_lens + FIXED parts; cuda-graph capturable.",
        py::arg("q_lat"), py::arg("q_pe"), py::arg("kv_pool"), py::arg("o_lat"),
        py::arg("seq_lens"), py::arg("rows"), py::arg("kv_indices"), py::arg("kv_indptr"),
        py::arg("parts"), py::arg("scale"), py::arg("kv_scale"), py::arg("scale_p"));
  m.def("mla_decode_a16w8_multiq_plan_parts_q", &plan_parts_q_op,
        "KV-split count for an a16w8 multi-query graph captured at (B, max_seq_len, q_len, H). Call "
        "at capture, reuse on replay.",
        py::arg("B"), py::arg("max_seq_len"), py::arg("q_len"), py::arg("H"));
}

}  // namespace moonmath_mla_a16w8_multiq
