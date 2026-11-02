// PyTorch binding for dsv4_sparse_decode.hip (DeepSeek-V4 sparse attention decode on gfx942).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <stdexcept>
#include <string>

extern "C" {
size_t dsv4_sparse_decode_workspace_for(int B, int H);
int launch_dsv4_sparse_decode(
    const void* q, const void* pool_main, const void* pool_extra, long page_stride_main,
    long page_stride_extra, int page_tokens_main, int page_tokens_extra, const void* attn_sink,
    void* out, void* lse, const void* seq_lens, const void* kv_indices, const void* kv_indptr,
    int B, int H, float scale, void* workspace, void* stream, int kv_len);
}

namespace moonmath_dsv4_sparse_decode {
namespace {

constexpr int64_t kDim = 512, kTokenBytes = 576 + 8, kMaxHeads = 128;

void check_i32dev(const at::Tensor& t, const char* name, int64_t numel) {
  if (t.scalar_type() != at::kInt || !t.is_contiguous() || t.numel() != numel || !t.device().is_cuda())
    throw std::invalid_argument(std::string("dsv4_sparse_decode: ") + name +
                                " must be a contiguous int32 device tensor of " + std::to_string(numel));
}

// A paged cache as the DeepSeek-V4 backend holds it: [pages, page_tokens * 584 bytes] in any
// byte-sized dtype, pages possibly strided.
void check_pool(const at::Tensor& pool, int64_t page_tokens, const char* name) {
  if (pool.element_size() != 1 || !pool.device().is_cuda() || pool.dim() < 2)
    throw std::invalid_argument(std::string("dsv4_sparse_decode: ") + name +
                                " must be a byte-typed device tensor [pages, ...]");
  if (page_tokens < 1 || pool.numel() / pool.size(0) != page_tokens * kTokenBytes)
    throw std::invalid_argument(std::string("dsv4_sparse_decode: ") + name +
                                " pages must hold page_tokens * 584 bytes");
  if (pool.stride(0) % 8 != 0)
    throw std::invalid_argument(std::string("dsv4_sparse_decode: ") + name + " pages must be 8-B aligned");
}

void dsv4_sparse_decode_op(const at::Tensor& q, const at::Tensor& pool_main, int64_t page_tokens_main,
                           const at::Tensor& pool_extra, int64_t page_tokens_extra,
                           const at::Tensor& kv_indices, const at::Tensor& kv_indptr,
                           const at::Tensor& seq_lens, double scale,
                           const c10::optional<at::Tensor>& attn_sink, at::Tensor& out) {
  if (q.dim() != 3 || q.scalar_type() != at::kBFloat16 || !q.is_contiguous() || !q.device().is_cuda() ||
      q.size(2) != kDim)
    throw std::invalid_argument("dsv4_sparse_decode: q must be contiguous bf16 [B, H, 512]");
  const int64_t B = q.size(0), H = q.size(1);
  if (H < 1 || H > kMaxHeads) throw std::invalid_argument("dsv4_sparse_decode: H must be in [1,128]");
  if (out.sizes() != q.sizes() || out.scalar_type() != at::kBFloat16 || !out.is_contiguous())
    throw std::invalid_argument("dsv4_sparse_decode: out must be contiguous bf16 shaped like q");
  check_pool(pool_main, page_tokens_main, "pool_main");
  check_pool(pool_extra, page_tokens_extra, "pool_extra");
  check_i32dev(kv_indptr, "kv_indptr", B + 1);
  check_i32dev(seq_lens, "seq_lens", B);
  if (kv_indices.scalar_type() != at::kInt || !kv_indices.is_contiguous() || kv_indices.numel() < 1)
    throw std::invalid_argument("dsv4_sparse_decode: kv_indices must be contiguous int32");
  if (attn_sink.has_value() && (attn_sink->scalar_type() != at::kFloat || !attn_sink->is_contiguous() ||
                                attn_sink->numel() != H))
    throw std::invalid_argument("dsv4_sparse_decode: attn_sink must be contiguous fp32 [H]");

  const c10::cuda::CUDAGuard g(q.device());
  const auto stream = (void*)at::cuda::getCurrentCUDAStream(q.device().index()).stream();
  auto ws = at::empty({(int64_t)dsv4_sparse_decode_workspace_for((int)B, (int)H)},
                      q.options().dtype(at::kByte));
  auto lse = at::empty({B, H}, q.options().dtype(at::kFloat));
  const int rc = launch_dsv4_sparse_decode(
      q.data_ptr(), pool_main.data_ptr(), pool_extra.data_ptr(), (long)pool_main.stride(0),
      (long)pool_extra.stride(0), (int)page_tokens_main, (int)page_tokens_extra,
      attn_sink.has_value() ? attn_sink->data_ptr() : nullptr, out.data_ptr(), lse.data_ptr(),
      seq_lens.data_ptr(), kv_indices.data_ptr(), kv_indptr.data_ptr(), (int)B, (int)H, (float)scale,
      ws.data_ptr(), stream, (int)(kv_indices.numel() / B));
  if (rc != 0) throw std::runtime_error("dsv4_sparse_decode returned error code " + std::to_string(rc));
}

}  // namespace

void register_pybind(pybind11::module_& m) {
  namespace py = pybind11;
  m.def("dsv4_sparse_decode", &dsv4_sparse_decode_op,
        "CDNA3 DeepSeek-V4 sparse attention decode: bf16 q [B,H,512] against per-request CSR slot "
        "lists into two paged fp8 caches (bit 30 of a slot picks pool_extra), E8M0-scaled nope + bf16 "
        "rope rows, optional attention sink. Writes out [B,H,512]. Cuda-graph capturable.",
        py::arg("q"), py::arg("pool_main"), py::arg("page_tokens_main"), py::arg("pool_extra"),
        py::arg("page_tokens_extra"), py::arg("kv_indices"), py::arg("kv_indptr"), py::arg("seq_lens"),
        py::arg("scale"), py::arg("attn_sink") = py::none(), py::arg("out"));
}

}  // namespace moonmath_dsv4_sparse_decode
