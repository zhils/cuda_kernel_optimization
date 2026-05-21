namespace gemm_fp16 {
namespace wmma = nvcuda::wmma;

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kTileK = 32;

constexpr int kNumWarpsM = 2;
constexpr int kNumWarpsN = 4;
constexpr int kWarpSize = 32;
constexpr int kThreads = kNumWarpsM * kNumWarpsN * kWarpSize;
static_assert(kThreads == 256, "thread count");

constexpr int kWarpTilesM = 4;
constexpr int kWarpTilesN = 2;
constexpr int kWarpM = kWarpTilesM * 16;
constexpr int kWarpN = kWarpTilesN * 16;
static_assert(kNumWarpsM * kWarpM == kBlockM, "block M coverage");
static_assert(kNumWarpsN * kWarpN == kBlockN, "block N coverage");

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
static_assert(kTileK % kWmmaK == 0, "kTileK must be multiple of kWmmaK");

constexpr int kBlockThreadsX = 16;
constexpr int kBlockThreadsY = 16;

constexpr int kSmemABuf = kBlockM * kTileK;
constexpr int kSmemBBuf = kTileK * kBlockN;
constexpr size_t kSmemSize = sizeof(__half) * 2 * (kSmemABuf + kSmemBBuf);

__global__ __launch_bounds__(kThreads, 2) void GemmFP16Kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, 
    int N, 
    int K
  ) {
  // 共享内存双缓冲
  extern __shared__ __half shared_mem[];
  __half* As_buf0 = shared_mem;
  __half* As_buf1 = shared_mem + kSmemABuf;
  __half* Bs_buf0 = shared_mem + 2 * kSmemABuf;
  __half* Bs_buf1 = shared_mem + 2 * kSmemABuf + kSmemBBuf;

  // 线程与warp索引
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * kBlockThreadsX + tx;
  const int warp_id = tid / kWarpSize;
  const int warp_m = warp_id / kNumWarpsN;
  const int warp_n = warp_id % kNumWarpsN;

  // 初始化WMMA累加器
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>
      c_frag[kWarpTilesM][kWarpTilesN];
  #pragma unroll
  for (int i = 0; i < kWarpTilesM; ++i) {
    #pragma unroll
    for (int j = 0; j < kWarpTilesN; ++j) {
      wmma::fill_fragment(c_frag[i][j], 0.0f);
    }
  }

  const int num_k_tiles = (K + kTileK - 1) / kTileK;
  const int a_offset_base = warp_m * kWarpM;
  const int b_offset_base = warp_n * kWarpN;
  const int a_ld = kTileK;

  // 异步加载函数：cp.async 加载 A/B 块
  auto load_tile_async = [&](int tile_k_start, __half* As_buf, __half* Bs_buf) {
    const int total_a_half8 = (kBlockM * kTileK) / 8;
    const int a_half8_per_thread = (total_a_half8 + kThreads - 1) / kThreads;
    for (int l = 0; l < a_half8_per_thread; ++l) {
      int idx = tid * a_half8_per_thread + l;
      if (idx >= total_a_half8) continue;
      int r = idx / (kTileK / 8);
      int k_offset = (idx % (kTileK / 8)) * 8;
      int g_r = blockIdx.y * kBlockM + r;
      int g_k = tile_k_start + k_offset;
      __half* dst = As_buf + r * kTileK + k_offset;
      if (g_r < M && g_k + 7 < K) {
        __pipeline_memcpy_async(dst, &A[static_cast<size_t>(g_r) * K + g_k], 16);
      } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
          dst[i] = (g_r < M && g_k + i < K) ? A[static_cast<size_t>(g_r) * K + g_k + i] : __half(0.0f);
        }
      }
    }

    const int total_b_half8 = (kTileK * kBlockN) / 8;
    const int b_half8_per_thread = (total_b_half8 + kThreads - 1) / kThreads;
    for (int l = 0; l < b_half8_per_thread; ++l) {
      int idx = tid * b_half8_per_thread + l;
      if (idx >= total_b_half8) continue;
      int k_idx = idx / (kBlockN / 8);
      int c_offset = (idx % (kBlockN / 8)) * 8;
      int g_k = tile_k_start + k_idx;
      int g_c = blockIdx.x * kBlockN + c_offset;
      __half* dst = Bs_buf + k_idx * kBlockN + c_offset;
      if (g_k < K && g_c + 7 < N) {
        __pipeline_memcpy_async(dst, &B[static_cast<size_t>(g_k) * N + g_c], 16);
      } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
          dst[i] = (g_k < K && g_c + i < N) ? B[static_cast<size_t>(g_k) * N + g_c + i] : __half(0.0f);
        }
      }
    }
  };

  // 加载首块并同步
  load_tile_async(0, As_buf0, Bs_buf0);
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();

  // 主循环：异步加载下一块 + WMMA计算当前块
  #pragma unroll
  for (int t = 0; t < num_k_tiles; ++t) {
    __half* As_read = (t & 1) ? As_buf1 : As_buf0;
    __half* Bs_read = (t & 1) ? Bs_buf1 : Bs_buf0;

    if (t + 1 < num_k_tiles) {
      __half* As_write = (t & 1) ? As_buf0 : As_buf1;
      __half* Bs_write = (t & 1) ? Bs_buf0 : Bs_buf1;
      load_tile_async((t + 1) * kTileK, As_write, Bs_write);
      __pipeline_commit();
    }

    // WMMA计算：加载矩阵片段 + 执行乘加
    #pragma unroll
    for (int kk = 0; kk < kTileK; kk += kWmmaK) {
      wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __half, wmma::row_major>
          a_frag[kWarpTilesM];
      #pragma unroll
      for (int i = 0; i < kWarpTilesM; ++i) {
        wmma::load_matrix_sync(a_frag[i],
            As_read + (a_offset_base + i * kWmmaM) * a_ld + kk, a_ld);
      }

      #pragma unroll
      for (int j = 0; j < kWarpTilesN; ++j) {
        wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __half, wmma::row_major>
            b_frag;
        wmma::load_matrix_sync(b_frag,
            Bs_read + kk * kBlockN + b_offset_base + j * kWmmaN, kBlockN);

        #pragma unroll
        for (int i = 0; i < kWarpTilesM; ++i) {
          wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag, c_frag[i][j]);
        }
      }
    }

    if (t + 1 < num_k_tiles) {
      __pipeline_wait_prior(0);
    }
    __syncthreads();
  }

  // 写回结果
  const int out_r = blockIdx.y * kBlockM + a_offset_base;
  const int out_c = blockIdx.x * kBlockN + b_offset_base;

  #pragma unroll
  for (int i = 0; i < kWarpTilesM; ++i) {
    #pragma unroll
    for (int j = 0; j < kWarpTilesN; ++j) {
      const int g_r = out_r + i * kWmmaM;
      const int g_c = out_c + j * kWmmaN;
      if (g_r + kWmmaM <= M && g_c + kWmmaN <= N) {
        wmma::store_matrix_sync(
            C + static_cast<size_t>(g_r) * N + g_c,
            c_frag[i][j], N, wmma::mem_row_major);
      }
    }
  }
}

}  // namespace gemm_fp16
