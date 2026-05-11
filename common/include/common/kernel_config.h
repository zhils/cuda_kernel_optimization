#pragma once

#include <cuda_runtime.h>

namespace common {

constexpr int CeilDiv(int x, int y) {
  return (x + y - 1) / y;
}

struct LaunchConfig2D {
  dim3 block;
  dim3 grid;
};

struct LaunchConfig1D {
  dim3 block;
  dim3 grid;
};

struct GemmLaunchConfig {
  int block_m;
  int block_n;
  int tile_k;
  dim3 block;
  dim3 grid;
};

inline LaunchConfig2D MakeLaunchConfig2D(int m, int n, int block_m, int block_n,
                                         int block_x, int block_y) {
  LaunchConfig2D cfg;
  cfg.block = dim3(block_x, block_y);
  cfg.grid = dim3(CeilDiv(n, block_n), CeilDiv(m, block_m));
  return cfg;
}

inline LaunchConfig1D MakeLaunchConfig1D(int total_work, int block_x) {
  LaunchConfig1D cfg;
  cfg.block = dim3(block_x);
  cfg.grid = dim3(CeilDiv(total_work, block_x));
  return cfg;
}

inline LaunchConfig1D MakeWarpRowLaunchConfig(int rows, int warps_per_block,
                                              int warp_size = 32) {
  LaunchConfig1D cfg;
  cfg.block = dim3(warps_per_block * warp_size);
  cfg.grid = dim3(CeilDiv(rows, warps_per_block));
  return cfg;
}

inline GemmLaunchConfig MakeGemmLaunchConfig(int m, int n, int block_m, int block_n,
                                             int tile_k, int block_x, int block_y) {
  auto basic = MakeLaunchConfig2D(m, n, block_m, block_n, block_x, block_y);
  GemmLaunchConfig cfg;
  cfg.block_m = block_m;
  cfg.block_n = block_n;
  cfg.tile_k = tile_k;
  cfg.block = basic.block;
  cfg.grid = basic.grid;
  return cfg;
}

}  // namespace common
