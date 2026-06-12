// Pybind11 API for moonmath_attention
// ROCm-only CDNA3 (MI300X/gfx942) fused attention kernel

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace py = pybind11;

// Extern "C" launchers from the three round-mode TUs
extern "C" {
int launch_v_transpose_rtna(
    const void* v, void* v_t, int n_rows, int seq_len_per_head, int heads, int layout, void* stream);
int launch_attention_forward_rtna(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int batch,
    int heads,
    int seq_len,
    int seq_len_kv,
    int head_dim,
    int layout,
    void* stream);

int launch_v_transpose_rtne(
    const void* v, void* v_t, int n_rows, int seq_len_per_head, int heads, int layout, void* stream);
int launch_attention_forward_rtne(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int batch,
    int heads,
    int seq_len,
    int seq_len_kv,
    int head_dim,
    int layout,
    void* stream);

int launch_v_transpose_rtz(
    const void* v, void* v_t, int n_rows, int seq_len_per_head, int heads, int layout, void* stream);
int launch_attention_forward_rtz(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int batch,
    int heads,
    int seq_len,
    int seq_len_kv,
    int head_dim,
    int layout,
    void* stream);

// LiteAttention launchers (one per round-mode TU). read_list/write_list/must_do_list are int16.
int launch_attention_forward_lite_rtna(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int batch,
    int heads,
    int seq_len,
    int seq_len_kv,
    int head_dim,
    const void* read_list,
    void* write_list,
    const void* must_do_list,
    float threshold,
    int phase,
    int layout,
    void* stream);
int launch_attention_forward_lite_rtne(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int batch,
    int heads,
    int seq_len,
    int seq_len_kv,
    int head_dim,
    const void* read_list,
    void* write_list,
    const void* must_do_list,
    float threshold,
    int phase,
    int layout,
    void* stream);
int launch_attention_forward_lite_rtz(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int batch,
    int heads,
    int seq_len,
    int seq_len_kv,
    int head_dim,
    const void* read_list,
    void* write_list,
    const void* must_do_list,
    float threshold,
    int phase,
    int layout,
    void* stream);
}

// Helper: validate tensor properties
static void check_tensor(const at::Tensor& t, const std::string& name, bool check_contiguous = true) {
  if (!t.defined()) {
    throw std::invalid_argument(name + " must be a valid tensor");
  }
  if (t.scalar_type() != at::kBFloat16) {
    throw std::invalid_argument(name + " must be bfloat16, got " + std::string(at::toString(t.scalar_type())));
  }
  if (t.dim() != 4) {
    throw std::invalid_argument(name + " must be 4-D, got " + std::to_string(t.dim()) + "-D");
  }
  if (check_contiguous && !t.is_contiguous()) {
    throw std::invalid_argument(name + " must be contiguous");
  }
  if (!t.device().is_cuda()) {
    throw std::invalid_argument(name + " must be on a CUDA/HIP device (GPU-only), got device=" + t.device().str());
  }
}

at::Tensor forward(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    std::optional<at::Tensor> out_,
    std::string round_mode,
    std::string layout) {
  // Validate inputs
  check_tensor(q, "q");
  check_tensor(k, "k");
  check_tensor(v, "v");

  // Validate round_mode
  if (round_mode != "rtna" && round_mode != "rtne" && round_mode != "rtz") {
    throw std::invalid_argument("round_mode must be 'rtna', 'rtne', or 'rtz', got '" + round_mode + "'");
  }

  // Validate layout
  bool bshd = (layout == "bshd");
  if (!bshd && layout != "bhsd") {
    throw std::invalid_argument("layout must be 'bhsd' or 'bshd', got '" + layout + "'");
  }
  int layout_int = bshd ? 1 : 0;

  // Extract dimensions (layout-aware)
  int sax = bshd ? 1 : 2;  // seq axis
  int hax = bshd ? 2 : 1;  // heads axis

  int B = q.size(0);
  int H = q.size(hax);
  int Sq = q.size(sax);
  int D = q.size(3);
  int Skv = k.size(sax);

  // Validate shapes
  if (D != 128) {
    throw std::invalid_argument("head_dim must be 128, got " + std::to_string(D));
  }

  if (k.size(0) != B || k.size(hax) != H || k.size(3) != D) {
    std::ostringstream oss;
    oss << "k shape mismatch: expected batch=" << B << ", heads=" << H << ", head_dim=" << D
        << ", got k.shape=" << k.sizes();
    throw std::invalid_argument(oss.str());
  }

  if (v.size(0) != B || v.size(hax) != H || v.size(sax) != Skv || v.size(3) != D) {
    std::ostringstream oss;
    oss << "v shape mismatch: must match k.shape, got v.shape=" << v.sizes() << ", k.shape=" << k.sizes();
    throw std::invalid_argument(oss.str());
  }

  if (Sq <= 0 || Skv <= 0) {
    throw std::invalid_argument(
        "seq_len must be positive, got Sq=" + std::to_string(Sq) + ", Skv=" + std::to_string(Skv));
  }

  // Check devices match
  if (q.device() != k.device() || q.device() != v.device()) {
    throw std::invalid_argument("q, k, v must be on the same device");
  }

  // Validate or allocate output
  at::Tensor out;
  if (out_.has_value()) {
    out = out_.value();
    check_tensor(out, "out");
    if (out.sizes() != q.sizes()) {
      std::ostringstream oss;
      oss << "out shape must match q.shape, got out.shape=" << out.sizes() << ", q.shape=" << q.sizes();
      throw std::invalid_argument(oss.str());
    }
    if (out.device() != q.device()) {
      throw std::invalid_argument("out must be on the same device as q");
    }
  } else {
    out = at::empty_like(q);
  }

  // Set device guard and get stream
  const c10::cuda::CUDAGuard device_guard(q.device());
  void* stream = (void*)at::cuda::getCurrentCUDAStream(q.device().index()).stream();

  // Allocate v_t (per-head padded to 64-row boundary)
  int Skv_pad = ((Skv + 63) / 64) * 64;
  at::Tensor v_t = at::empty({B, H, Skv_pad, D}, v.options());

  // Dispatch on round_mode
  int (*vt_fn)(const void*, void*, int, int, int, int, void*);
  int (*fwd_fn)(const void*, const void*, const void*, void*, int, int, int, int, int, int, void*);

  if (round_mode == "rtna") {
    vt_fn = launch_v_transpose_rtna;
    fwd_fn = launch_attention_forward_rtna;
  } else if (round_mode == "rtne") {
    vt_fn = launch_v_transpose_rtne;
    fwd_fn = launch_attention_forward_rtne;
  } else {  // rtz
    vt_fn = launch_v_transpose_rtz;
    fwd_fn = launch_attention_forward_rtz;
  }

  // Launch V transpose
  int rc = vt_fn(v.data_ptr(), v_t.data_ptr(), B * H * Skv, Skv, H, layout_int, stream);
  if (rc != 0) {
    throw std::runtime_error("launch_v_transpose returned error code " + std::to_string(rc));
  }

  // Launch attention forward
  rc = fwd_fn(q.data_ptr(), k.data_ptr(), v_t.data_ptr(), out.data_ptr(), B, H, Sq, Skv, D, layout_int, stream);
  if (rc != 0) {
    throw std::runtime_error("launch_attention_forward returned error code " + std::to_string(rc));
  }

  return out;
}

// LiteAttention forward: process only the K-blocks named in read_list, emit the next-step skip list
// into write_list. Mirrors forward() plus the int16 skip-list args (read/write/must_do).
at::Tensor forward_lite(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor read_list,
    at::Tensor write_list,
    float threshold,
    int phase,
    std::optional<at::Tensor> must_do_list_,
    std::optional<at::Tensor> out_,
    std::string round_mode,
    std::string layout) {
  check_tensor(q, "q");
  check_tensor(k, "k");
  check_tensor(v, "v");

  if (round_mode != "rtna" && round_mode != "rtne" && round_mode != "rtz") {
    throw std::invalid_argument("round_mode must be 'rtna', 'rtne', or 'rtz', got '" + round_mode + "'");
  }
  bool bshd = (layout == "bshd");
  if (!bshd && layout != "bhsd") {
    throw std::invalid_argument("layout must be 'bhsd' or 'bshd', got '" + layout + "'");
  }
  int layout_int = bshd ? 1 : 0;
  int sax = bshd ? 1 : 2;
  int hax = bshd ? 2 : 1;

  int B = q.size(0);
  int H = q.size(hax);
  int Sq = q.size(sax);
  int D = q.size(3);
  int Skv = k.size(sax);
  if (D != 128) {
    throw std::invalid_argument("head_dim must be 128, got " + std::to_string(D));
  }
  if (q.device() != k.device() || q.device() != v.device()) {
    throw std::invalid_argument("q, k, v must be on the same device");
  }

  // Skip lists must be contiguous int16 on q's device (must_do_list is optional).
  auto check_list = [&](const at::Tensor& t, const std::string& name) {
    if (t.scalar_type() != at::kShort) {
      throw std::invalid_argument(name + " must be int16, got " + std::string(at::toString(t.scalar_type())));
    }
    if (!t.is_contiguous()) {
      throw std::invalid_argument(name + " must be contiguous");
    }
    if (t.device() != q.device()) {
      throw std::invalid_argument(name + " must be on the same device as q");
    }
  };
  check_list(read_list, "read_list");
  check_list(write_list, "write_list");
  const void* must_do_ptr = nullptr;
  if (must_do_list_.has_value()) {
    check_list(must_do_list_.value(), "must_do_list");
    must_do_ptr = must_do_list_.value().data_ptr();
  }

  at::Tensor out;
  if (out_.has_value()) {
    out = out_.value();
    check_tensor(out, "out");
    if (out.sizes() != q.sizes()) {
      throw std::invalid_argument("out shape must match q.shape");
    }
    if (out.device() != q.device()) {
      throw std::invalid_argument("out must be on the same device as q");
    }
  } else {
    out = at::empty_like(q);
  }

  const c10::cuda::CUDAGuard device_guard(q.device());
  void* stream = (void*)at::cuda::getCurrentCUDAStream(q.device().index()).stream();

  int Skv_pad = ((Skv + 63) / 64) * 64;
  at::Tensor v_t = at::empty({B, H, Skv_pad, D}, v.options());

  int (*vt_fn)(const void*, void*, int, int, int, int, void*);
  int (*lite_fn)(
      const void*,
      const void*,
      const void*,
      void*,
      int,
      int,
      int,
      int,
      int,
      const void*,
      void*,
      const void*,
      float,
      int,
      int,
      void*);
  if (round_mode == "rtna") {
    vt_fn = launch_v_transpose_rtna;
    lite_fn = launch_attention_forward_lite_rtna;
  } else if (round_mode == "rtne") {
    vt_fn = launch_v_transpose_rtne;
    lite_fn = launch_attention_forward_lite_rtne;
  } else {  // rtz
    vt_fn = launch_v_transpose_rtz;
    lite_fn = launch_attention_forward_lite_rtz;
  }

  int rc = vt_fn(v.data_ptr(), v_t.data_ptr(), B * H * Skv, Skv, H, layout_int, stream);
  if (rc != 0) {
    throw std::runtime_error("launch_v_transpose returned error code " + std::to_string(rc));
  }
  rc = lite_fn(
      q.data_ptr(),
      k.data_ptr(),
      v_t.data_ptr(),
      out.data_ptr(),
      B,
      H,
      Sq,
      Skv,
      D,
      read_list.data_ptr(),
      write_list.data_ptr(),
      must_do_ptr,
      threshold,
      phase,
      layout_int,
      stream);
  if (rc != 0) {
    throw std::runtime_error("launch_attention_forward_lite returned error code " + std::to_string(rc));
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "moonmath_attention: ROCm CDNA3 (MI300X) fused attention kernel";

  m.def(
      "forward",
      &forward,
      "Fused forward attention: O = softmax(QK^T / sqrt(D)) V",
      py::arg("q"),
      py::arg("k"),
      py::arg("v"),
      py::arg("out") = py::none(),
      py::arg("round_mode") = "rtna",
      py::arg("layout") = "bhsd");

  m.def(
      "forward_lite",
      &forward_lite,
      "LiteAttention forward with K-block skipping (not yet implemented)",
      py::arg("q"),
      py::arg("k"),
      py::arg("v"),
      py::arg("read_list"),
      py::arg("write_list"),
      py::arg("threshold"),
      py::arg("phase"),
      py::arg("must_do_list") = py::none(),
      py::arg("out") = py::none(),
      py::arg("round_mode") = "rtna",
      py::arg("layout") = "bhsd");
}
