#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using bf16_t = hip_bfloat16;

struct p3 {
    unsigned int _p0;
    unsigned int _p1;
    unsigned int _p2;
};

struct p2 {
    unsigned int _p0;
    unsigned int _p1;
};

struct __attribute__((packed)) fmha_fwd_v3_args {
    void* ptr_o;
    p2 _p0;
    const void* ptr_q;
    p2 _p1;
    const void* ptr_k;
    p2 _p2;
    const void* ptr_v;
    p2 _p3;
    void* ptr_lse;
    p2 _p4;
    float scalar;
    p3 _p5;
    unsigned int s_seq_len;
    p3 _p6;
    unsigned int s_Seqs;
    p3 _p7;
    unsigned int s_Ts;
    p3 _p8;
    unsigned int s_Hs;
    p3 _p9;
    unsigned int s_Bs;
    p3 _p10;
    unsigned int s_gqa;
    p3 _p11;
    unsigned int s_k_Seqs;
    p3 _p12;
    unsigned int s_k_Hs;
    p3 _p13;
    unsigned int s_k_Bs;
    p3 _p14;
    unsigned int s_opt;
    p3 _p15;
    unsigned int s_lse;
    p3 _p16;
    unsigned int s_kv_seq_len;
    p3 _p17;
    unsigned int s_qk_head_dim;
    p3 _p18;
    unsigned int s_v_head_dim;
    p3 _p19;
    unsigned int s_q_head_num;
    p3 _p20;
    unsigned int s_v_Seqs;
    p3 _p21;
    unsigned int s_v_Hs;
    p3 _p22;
    unsigned int s_v_Bs;
    p3 _p23;
    unsigned int s_o_Seqs;
    p3 _p24;
    unsigned int s_o_Hs;
    p3 _p25;
    unsigned int s_o_Bs;
    p3 _p26;
    const void* ptr_qseq;
    p2 _p27;
    const void* ptr_kseq;
    p2 _p28;
    unsigned int s_lse_Hs;
    p3 _p29;
    const void* ptr_qseq_padding;
    p2 _p30;
    const void* ptr_kseq_padding;
    p2 _p31;
    const void* ptr_q_descale;
    p2 _p32;
    const void* ptr_k_descale;
    p2 _p33;
    const void* ptr_v_descale;
    p2 _p34;
    unsigned int s_descale_q_Bs;
    p3 _p35;
    unsigned int s_descale_q_Hs;
    p3 _p36;
    unsigned int s_descale_k_Bs;
    p3 _p37;
    unsigned int s_descale_k_Hs;
    p3 _p38;
    unsigned int s_descale_v_Bs;
    p3 _p39;
    unsigned int s_descale_v_Hs;
    p3 _p40;
};

hipModule_t g_module = nullptr;
hipFunction_t g_func = nullptr;
bool g_loaded = false;
std::string g_kernel_variant = "rtne";
// AITER's own gfx942 hd128 batch-mode forward path defaults tune_opt/s_opt to 5
// for the standard non-causal, non-group BF16 kernel.
unsigned int g_s_opt = 5;

inline void hip_check(hipError_t err) {
    if (err != hipSuccess) {
        abort();
    }
}

std::string get_arch_name() {
    int dev = 0;
    hip_check(hipGetDevice(&dev));
    hipDeviceProp_t prop{};
    hip_check(hipGetDeviceProperties(&prop, dev));
    std::string arch = prop.gcnArchName;
    size_t pos = arch.find(':');
    return pos == std::string::npos ? arch : arch.substr(0, pos);
}

bool is_mi308() {
    int dev = 0;
    hip_check(hipGetDevice(&dev));
    int chip_id = 0;
    hip_check(hipDeviceGetAttribute(&chip_id, hipDeviceAttributePciChipId, dev));
    return chip_id == 0x74a2 || chip_id == 0x74a8 || chip_id == 0x74b6 || chip_id == 0x74bc;
}

std::filesystem::path get_library_dir() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&get_library_dir), &info) == 0 || !info.dli_fname) {
        return {};
    }
    return std::filesystem::path(info.dli_fname).parent_path();
}

std::filesystem::path find_hsaco_root() {
    std::vector<std::filesystem::path> candidates;
    if (const char* root = std::getenv("AITER_ASM_DIR")) {
        candidates.emplace_back(root);
    }

    const std::filesystem::path lib_dir = get_library_dir();
    if (!lib_dir.empty()) {
        candidates.emplace_back(lib_dir / "vendor" / "aiter_hsa");
    }

    candidates.emplace_back("/tmp/aiter_rocm_791921/aiter_meta/hsa");

    const std::string arch = get_arch_name();
    const std::string mi_dir = is_mi308() ? "MI308" : "MI300";
    const std::string file = "fwd_hd128_bf16_" + g_kernel_variant + ".co";
    std::error_code ec;
    for (const auto& root : candidates) {
        const auto hsaco = root / arch / "fmha_v3_fwd" / mi_dir / file;
        if (std::filesystem::exists(hsaco, ec)) {
            return root;
        }
    }
    return {};
}

void ensure_kernel_loaded() {
    if (g_loaded) return;
    if (const char* variant = std::getenv("AITER_BF16_VARIANT")) {
        const std::string value = variant;
        if (value == "rtne" || value == "rtna" || value == "rtz") {
            g_kernel_variant = value;
        }
    }
    if (const char* opt = std::getenv("AITER_S_OPT")) {
        const long value = std::strtol(opt, nullptr, 10);
        if (value >= 0 && value <= 5) {
            g_s_opt = static_cast<unsigned int>(value);
        }
    }
    const std::filesystem::path root = find_hsaco_root();
    if (root.empty()) {
        std::fprintf(stderr,
                     "AITER HSACO not found. Set AITER_ASM_DIR or use repo-local vendor/aiter_hsa.\n");
        std::abort();
    }
    const std::string base = root.string();
    std::string hsaco = base + "/" + get_arch_name() + "/fmha_v3_fwd/" +
                        (is_mi308() ? "MI308/" : "MI300/") + "fwd_hd128_bf16_" + g_kernel_variant + ".co";
    const size_t slash = hsaco.find_last_of('/');
    const std::string hsaco_name = slash == std::string::npos ? hsaco : hsaco.substr(slash + 1);
    // This wrapper is intentionally pinned to the plain forward kernel:
    // - no causal mask
    // - no grouped/varlen mode
    if (hsaco_name.find("causal") != std::string::npos || hsaco_name.find("_group") != std::string::npos) {
        abort();
    }
    std::string symbol;
    if (g_kernel_variant == "rtne") {
        symbol = "_ZN5aiter24fmha_fwd_hd128_bf16_rtneE";
    } else if (g_kernel_variant == "rtz") {
        symbol = "_ZN5aiter23fmha_fwd_hd128_bf16_rtzE";
    } else {
        symbol = "_ZN5aiter24fmha_fwd_hd128_bf16_rtnaE";
    }
    hip_check(hipModuleLoad(&g_module, hsaco.c_str()));
    hip_check(hipModuleGetFunction(&g_func, g_module, symbol.c_str()));
    g_loaded = true;
}

void launch_aiter(const fmha_fwd_v3_args& args, hipStream_t stream, int gdx, int gdy, int gdz) {
    auto args_copy = args;
    size_t arg_size = sizeof(args_copy);
    void* config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args_copy,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
        HIP_LAUNCH_PARAM_END
    };
    // AITER's gfx942 hd128/v128 batch-mode forward path launches with 512 threads.
    hip_check(hipModuleLaunchKernel(g_func, gdx, gdy, gdz, 512, 1, 1, 0, stream, nullptr, config));
}

} // namespace

// Switch the loaded AITER kernel variant ("rtne" | "rtna" | "rtz"). Unloads
// the current HSACO if a different variant is requested; the next launch
// reloads with the new variant. Returns 0 on success, 1 on bad input.
extern "C" int set_aiter_variant(const char* variant) {
    if (!variant) return 1;
    const std::string v = variant;
    if (v != "rtne" && v != "rtna" && v != "rtz") return 1;
    if (g_loaded && v == g_kernel_variant) return 0;
    if (g_module) { hip_check(hipModuleUnload(g_module)); g_module = nullptr; }
    g_func = nullptr;
    g_loaded = false;
    g_kernel_variant = v;
    return 0;
}

extern "C" hipError_t launch_attention_forward(const bf16_t* Q,
                                                const bf16_t* K,
                                                const bf16_t* V,
                                                bf16_t* Out,
                                                int batch,
                                                int heads,
                                                int seq_len,
                                                int head_dim,
                                                hipStream_t stream) {
    if (batch <= 0 || heads <= 0 || seq_len <= 0) return hipErrorInvalidValue;
    if (head_dim != 128) return hipErrorInvalidValue;

    ensure_kernel_loaded();

    const unsigned int in_bpe = sizeof(bf16_t);
    const unsigned int out_bpe = sizeof(bf16_t);
    const unsigned int seq_stride = static_cast<unsigned int>(head_dim);
    const unsigned int head_stride = static_cast<unsigned int>(seq_len) * head_dim;
    const unsigned int batch_stride = static_cast<unsigned int>(heads) * head_stride;
    const unsigned int ts_qo = 256;

    fmha_fwd_v3_args args{};
    args.ptr_o         = Out;
    args.ptr_q         = Q;
    args.ptr_k         = K;
    args.ptr_v         = V;
    args.ptr_lse       = nullptr;
    args.scalar        = 1.0f / std::sqrt(static_cast<float>(head_dim));
    args.s_seq_len     = seq_len;
    args.s_Seqs        = seq_stride * in_bpe;
    args.s_Ts          = ts_qo * seq_stride * in_bpe;
    args.s_Hs          = head_stride * in_bpe;
    args.s_Bs          = batch_stride * in_bpe;
    args.s_gqa         = 1;
    args.s_k_Seqs      = seq_stride * in_bpe;
    args.s_k_Hs        = head_stride * in_bpe;
    args.s_k_Bs        = batch_stride * in_bpe;
    args.s_opt         = g_s_opt;
    args.s_lse         = 0;
    args.s_kv_seq_len  = seq_len;
    args.s_qk_head_dim = head_dim;
    args.s_v_head_dim  = head_dim;
    args.s_q_head_num  = heads;
    args.s_v_Seqs      = seq_stride * in_bpe;
    args.s_v_Hs        = head_stride * in_bpe;
    args.s_v_Bs        = batch_stride * in_bpe;
    args.s_o_Seqs      = seq_stride * out_bpe;
    args.s_o_Hs        = head_stride * out_bpe;
    args.s_o_Bs        = batch_stride * out_bpe;
    args.ptr_qseq      = nullptr;
    args.ptr_kseq      = nullptr;
    args.s_lse_Hs      = 0;
    args.ptr_qseq_padding = nullptr;
    args.ptr_kseq_padding = nullptr;
    args.ptr_q_descale = nullptr;
    args.ptr_k_descale = nullptr;
    args.ptr_v_descale = nullptr;
    args.s_descale_q_Bs = 0;
    args.s_descale_q_Hs = 0;
    args.s_descale_k_Bs = 0;
    args.s_descale_k_Hs = 0;
    args.s_descale_v_Bs = 0;
    args.s_descale_v_Hs = 0;

    const int gdx = (seq_len + ts_qo - 1) / ts_qo;
    const int gdy = heads;
    const int gdz = batch;
    launch_aiter(args, stream, gdx, gdy, gdz);
    return hipGetLastError();
}
