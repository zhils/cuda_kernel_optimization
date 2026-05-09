#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr float kSmoothEps = 1e-6f;
constexpr float kSmoothAlpha = 0.6f;
constexpr float kSmoothScaleMin = 0.25f;
constexpr float kSmoothScaleMax = 4.0f;
constexpr float kActivationScaleGuard = 1.25f;

struct ErrorStats {
  double mae = 0.0;
  double rmse = 0.0;
  double max_abs = 0.0;
  double mean_rel = 0.0;
};

int8_t SaturateToInt8(int v) {
  if (v > 127) return static_cast<int8_t>(127);
  if (v < -128) return static_cast<int8_t>(-128);
  return static_cast<int8_t>(v);
}

float Sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

float Softplus(float x) {
  return std::max(x, 0.0f) + std::log1p(std::exp(-std::fabs(x)));
}

void InitRandom(std::vector<float>& v, std::mt19937& gen, float lo = -0.5f, float hi = 0.5f) {
  std::uniform_real_distribution<float> dist(lo, hi);
  for (float& x : v) x = dist(gen);
}

void InjectActivationOutliers(std::vector<float>& x, int B, int L, int D, std::mt19937& gen) {
  std::uniform_int_distribution<int> b_dist(0, B - 1);
  std::uniform_int_distribution<int> t_dist(0, L - 1);
  std::uniform_real_distribution<float> amp_dist(6.0f, 10.0f);
  std::uniform_real_distribution<float> sign_dist(0.0f, 1.0f);

  // Simulate channel-wise outlier pattern observed in LLM activations.
  std::vector<int> hotspot_channels;
  int num_hotspots = std::max(1, D / 32);
  for (int i = 0; i < num_hotspots; ++i) {
    hotspot_channels.push_back((i * 31) % D);
  }
  std::uniform_int_distribution<int> hot_dist(0, static_cast<int>(hotspot_channels.size()) - 1);

  const int num_outliers = std::max(1, (B * L * num_hotspots) / 16);
  for (int i = 0; i < num_outliers; ++i) {
    int b = b_dist(gen);
    int t = t_dist(gen);
    int d = hotspot_channels[hot_dist(gen)];
    float amp = amp_dist(gen) * (sign_dist(gen) > 0.5f ? 1.0f : -1.0f);
    x[(b * L + t) * D + d] += amp;
  }
}

void ComputeSmoothQuantScale(const std::vector<float>& x_calib,
                             const std::vector<float>& W_decay,
                             const std::vector<float>& W_delta,
                             const std::vector<float>& W_state,
                             int Bc, int Lc, int D, int H,
                             float alpha,
                             std::vector<float>& s) {
  s.assign(D, 1.0f);
  for (int d = 0; d < D; ++d) {
    float max_x = 0.0f;
    for (int b = 0; b < Bc; ++b) {
      for (int t = 0; t < Lc; ++t) {
        float v = std::fabs(x_calib[(b * Lc + t) * D + d]);
        max_x = std::max(max_x, v);
      }
    }
    float max_w = 0.0f;
    for (int h = 0; h < H; ++h) {
      max_w = std::max(max_w, std::fabs(W_decay[h * D + d]));
      max_w = std::max(max_w, std::fabs(W_delta[h * D + d]));
      max_w = std::max(max_w, std::fabs(W_state[h * D + d]));
    }
    max_x = std::max(max_x, kSmoothEps);
    max_w = std::max(max_w, kSmoothEps);
    s[d] = std::pow(max_x, alpha) / std::pow(max_w, 1.0f - alpha);
    s[d] = std::max(s[d], kSmoothEps);
    s[d] = std::min(std::max(s[d], kSmoothScaleMin), kSmoothScaleMax);
  }
}

void ApplySmoothQuantTransform(std::vector<float>& x,
                               std::vector<float>& W_decay,
                               std::vector<float>& W_delta,
                               std::vector<float>& W_state,
                               int B, int L, int D, int H,
                               const std::vector<float>& s) {
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < L; ++t) {
      for (int d = 0; d < D; ++d) {
        // SmoothQuant: move activation difficulty to weights.
        x[(b * L + t) * D + d] /= s[d];
      }
    }
  }

  for (int h = 0; h < H; ++h) {
    for (int d = 0; d < D; ++d) {
      W_decay[h * D + d] *= s[d];
      W_delta[h * D + d] *= s[d];
      W_state[h * D + d] *= s[d];
    }
  }
}

void QuantizePerTensor(const float* src, int n, std::vector<int8_t>& q, float& scale) {
  float max_abs = 0.0f;
  for (int i = 0; i < n; ++i) max_abs = std::max(max_abs, std::fabs(src[i]));
  scale = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
  scale = std::max(scale, 1e-8f);
  q.resize(n);
  for (int i = 0; i < n; ++i) {
    int qi = static_cast<int>(std::nearbyint(src[i] / scale));
    q[i] = SaturateToInt8(qi);
  }
}

float ComputePerTensorScaleFromCalib(const std::vector<float>& x) {
  float max_abs = 0.0f;
  for (float v : x) max_abs = std::max(max_abs, std::fabs(v));
  float s = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
  return std::max(s, 1e-8f);
}

void ComputePerChannelScaleFromCalib(const std::vector<float>& x, int B, int L, int D,
                                     std::vector<float>& scales) {
  scales.assign(D, 1.0f);
  for (int d = 0; d < D; ++d) {
    float max_abs = 0.0f;
    for (int b = 0; b < B; ++b) {
      for (int t = 0; t < L; ++t) {
        max_abs = std::max(max_abs, std::fabs(x[(b * L + t) * D + d]));
      }
    }
    float s = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
    scales[d] = std::max(s * kActivationScaleGuard, 1e-8f);
  }
}

void QuantizeWeightsPerTensor(const std::vector<float>& W, std::vector<int8_t>& qW, float& scale) {
  QuantizePerTensor(W.data(), static_cast<int>(W.size()), qW, scale);
}

void QuantizeWeightsPerChannel(const std::vector<float>& W, int H, int D,
                               std::vector<int8_t>& qW, std::vector<float>& scales) {
  qW.resize(static_cast<size_t>(H) * D);
  scales.assign(H, 1.0f);
  for (int h = 0; h < H; ++h) {
    float max_abs = 0.0f;
    for (int d = 0; d < D; ++d) {
      max_abs = std::max(max_abs, std::fabs(W[h * D + d]));
    }
    float s = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
    s = std::max(s, 1e-8f);
    scales[h] = s;
    for (int d = 0; d < D; ++d) {
      int qi = static_cast<int>(std::nearbyint(W[h * D + d] / s));
      qW[h * D + d] = SaturateToInt8(qi);
    }
  }
}

void RunFp32Reference(const std::vector<float>& x,
                      const std::vector<float>& W_decay, const std::vector<float>& b_decay,
                      const std::vector<float>& W_delta, const std::vector<float>& b_delta,
                      const std::vector<float>& W_state, const std::vector<float>& b_state,
                      int B, int L, int D, int H,
                      std::vector<float>& out) {
  std::vector<float> alpha(static_cast<size_t>(B) * L * H);
  std::vector<float> delta(static_cast<size_t>(B) * L * H);
  std::vector<float> u(static_cast<size_t>(B) * L * H);
  out.assign(static_cast<size_t>(B) * L * H, 0.0f);

  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < L; ++t) {
      const float* x_bt = &x[(b * L + t) * D];
      for (int h = 0; h < H; ++h) {
        float decay_v = b_decay[h];
        float delta_v = b_delta[h];
        float state_v = b_state[h];
        for (int d = 0; d < D; ++d) {
          decay_v += x_bt[d] * W_decay[h * D + d];
          delta_v += x_bt[d] * W_delta[h * D + d];
          state_v += x_bt[d] * W_state[h * D + d];
        }
        int idx = (b * L + t) * H + h;
        alpha[idx] = Sigmoid(decay_v);
        delta[idx] = Softplus(delta_v);
        u[idx] = state_v;
      }
    }
  }

  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      float s = 0.0f;
      for (int t = 0; t < L; ++t) {
        int idx = (b * L + t) * H + h;
        s = alpha[idx] * s + delta[idx] * u[idx];
        out[idx] = s;
      }
    }
  }
}

void RunInt8PerTensor(const std::vector<float>& x,
                      const std::vector<float>& x_calib,
                      const std::vector<float>& W_decay, const std::vector<float>& b_decay,
                      const std::vector<float>& W_delta, const std::vector<float>& b_delta,
                      const std::vector<float>& W_state, const std::vector<float>& b_state,
                      int B, int L, int D, int H,
                      std::vector<float>& out) {
  std::vector<int8_t> qW_decay, qW_delta, qW_state;
  float sW_decay = 1.0f, sW_delta = 1.0f, sW_state = 1.0f;
  QuantizeWeightsPerTensor(W_decay, qW_decay, sW_decay);
  QuantizeWeightsPerTensor(W_delta, qW_delta, sW_delta);
  QuantizeWeightsPerTensor(W_state, qW_state, sW_state);

  std::vector<float> alpha(static_cast<size_t>(B) * L * H);
  std::vector<float> delta(static_cast<size_t>(B) * L * H);
  std::vector<float> u(static_cast<size_t>(B) * L * H);
  out.assign(static_cast<size_t>(B) * L * H, 0.0f);

  const float sx = ComputePerTensorScaleFromCalib(x_calib);
  std::vector<int8_t> qx(D);
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < L; ++t) {
      const float* x_bt = &x[(b * L + t) * D];
      for (int d = 0; d < D; ++d) {
        int qi = static_cast<int>(std::nearbyint(x_bt[d] / sx));
        qx[d] = SaturateToInt8(qi);
      }
      for (int h = 0; h < H; ++h) {
        int32_t acc_decay = 0;
        int32_t acc_delta = 0;
        int32_t acc_state = 0;
        for (int d = 0; d < D; ++d) {
          int32_t qxi = static_cast<int32_t>(qx[d]);
          acc_decay += qxi * static_cast<int32_t>(qW_decay[h * D + d]);
          acc_delta += qxi * static_cast<int32_t>(qW_delta[h * D + d]);
          acc_state += qxi * static_cast<int32_t>(qW_state[h * D + d]);
        }
        float decay_v = static_cast<float>(acc_decay) * sx * sW_decay + b_decay[h];
        float delta_v = static_cast<float>(acc_delta) * sx * sW_delta + b_delta[h];
        float state_v = static_cast<float>(acc_state) * sx * sW_state + b_state[h];
        int idx = (b * L + t) * H + h;
        alpha[idx] = Sigmoid(decay_v);
        delta[idx] = Softplus(delta_v);
        u[idx] = state_v;
      }
    }
  }

  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      float s = 0.0f;
      for (int t = 0; t < L; ++t) {
        int idx = (b * L + t) * H + h;
        s = alpha[idx] * s + delta[idx] * u[idx];
        out[idx] = s;
      }
    }
  }
}

void RunInt8SmoothQuantPerChannel(const std::vector<float>& x,
                                  const std::vector<float>& x_calib,
                                  const std::vector<float>& W_decay,
                                  const std::vector<float>& b_decay,
                                  const std::vector<float>& W_delta,
                                  const std::vector<float>& b_delta,
                                  const std::vector<float>& W_state,
                                  const std::vector<float>& b_state,
                                  int B, int L, int D, int H,
                                  std::vector<float>& out) {
  std::vector<float> x_s = x;
  std::vector<float> W_decay_s = W_decay;
  std::vector<float> W_delta_s = W_delta;
  std::vector<float> W_state_s = W_state;

  std::vector<float> smooth_scale;
  ComputeSmoothQuantScale(x_calib, W_decay, W_delta, W_state,
                          B, L, D, H, kSmoothAlpha, smooth_scale);
  ApplySmoothQuantTransform(x_s, W_decay_s, W_delta_s, W_state_s, B, L, D, H, smooth_scale);
  std::vector<float> x_calib_s = x_calib;
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < L; ++t) {
      for (int d = 0; d < D; ++d) {
        x_calib_s[(b * L + t) * D + d] /= smooth_scale[d];
      }
    }
  }

  std::vector<int8_t> qW_decay, qW_delta, qW_state;
  std::vector<float> sW_decay, sW_delta, sW_state;
  QuantizeWeightsPerChannel(W_decay_s, H, D, qW_decay, sW_decay);
  QuantizeWeightsPerChannel(W_delta_s, H, D, qW_delta, sW_delta);
  QuantizeWeightsPerChannel(W_state_s, H, D, qW_state, sW_state);
  std::vector<float> sX;
  ComputePerChannelScaleFromCalib(x_calib_s, B, L, D, sX);

  // Bias correction on calibration set to absorb systematic quantization bias.
  std::vector<float> corr_decay(H, 0.0f), corr_delta(H, 0.0f), corr_state(H, 0.0f);
  const float inv_count = 1.0f / static_cast<float>(B * L);
  std::vector<int8_t> qx_calib(D);
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < L; ++t) {
      const float* x_bt = &x_calib_s[(b * L + t) * D];
      for (int d = 0; d < D; ++d) {
        int qi = static_cast<int>(std::nearbyint(x_bt[d] / sX[d]));
        qx_calib[d] = SaturateToInt8(qi);
      }
      for (int h = 0; h < H; ++h) {
        float fp_decay = b_decay[h];
        float fp_delta = b_delta[h];
        float fp_state = b_state[h];
        float q_decay = b_decay[h];
        float q_delta = b_delta[h];
        float q_state = b_state[h];
        for (int d = 0; d < D; ++d) {
          float xv = x_bt[d];
          fp_decay += xv * W_decay_s[h * D + d];
          fp_delta += xv * W_delta_s[h * D + d];
          fp_state += xv * W_state_s[h * D + d];

          int32_t qxi = static_cast<int32_t>(qx_calib[d]);
          q_decay += static_cast<float>(qxi * static_cast<int32_t>(qW_decay[h * D + d])) * sX[d] * sW_decay[h];
          q_delta += static_cast<float>(qxi * static_cast<int32_t>(qW_delta[h * D + d])) * sX[d] * sW_delta[h];
          q_state += static_cast<float>(qxi * static_cast<int32_t>(qW_state[h * D + d])) * sX[d] * sW_state[h];
        }
        corr_decay[h] += (fp_decay - q_decay) * inv_count;
        corr_delta[h] += (fp_delta - q_delta) * inv_count;
        corr_state[h] += (fp_state - q_state) * inv_count;
      }
    }
  }

  std::vector<float> alpha(static_cast<size_t>(B) * L * H);
  std::vector<float> delta(static_cast<size_t>(B) * L * H);
  std::vector<float> u(static_cast<size_t>(B) * L * H);
  out.assign(static_cast<size_t>(B) * L * H, 0.0f);

  std::vector<int8_t> qx(D);
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < L; ++t) {
      const float* x_bt = &x_s[(b * L + t) * D];
      for (int d = 0; d < D; ++d) {
        int qi = static_cast<int>(std::nearbyint(x_bt[d] / sX[d]));
        qx[d] = SaturateToInt8(qi);
      }
      for (int h = 0; h < H; ++h) {
        float decay_v = b_decay[h];
        float delta_v = b_delta[h];
        float state_v = b_state[h];
        for (int d = 0; d < D; ++d) {
          int32_t qxi = static_cast<int32_t>(qx[d]);
          int32_t qwd = static_cast<int32_t>(qW_decay[h * D + d]);
          int32_t qwl = static_cast<int32_t>(qW_delta[h * D + d]);
          int32_t qws = static_cast<int32_t>(qW_state[h * D + d]);
          decay_v += static_cast<float>(qxi * qwd) * sX[d] * sW_decay[h];
          delta_v += static_cast<float>(qxi * qwl) * sX[d] * sW_delta[h];
          state_v += static_cast<float>(qxi * qws) * sX[d] * sW_state[h];
        }
        decay_v += corr_decay[h];
        delta_v += corr_delta[h];
        state_v += corr_state[h];
        int idx = (b * L + t) * H + h;
        alpha[idx] = Sigmoid(decay_v);
        delta[idx] = Softplus(delta_v);
        u[idx] = state_v;
      }
    }
  }

  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      float s = 0.0f;
      for (int t = 0; t < L; ++t) {
        int idx = (b * L + t) * H + h;
        s = alpha[idx] * s + delta[idx] * u[idx];
        out[idx] = s;
      }
    }
  }
}

ErrorStats ComputeError(const std::vector<float>& ref, const std::vector<float>& pred) {
  ErrorStats stats;
  if (ref.size() != pred.size() || ref.empty()) return stats;

  double se = 0.0;
  double ae = 0.0;
  double re = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double diff = static_cast<double>(pred[i]) - static_cast<double>(ref[i]);
    double ad = std::fabs(diff);
    ae += ad;
    se += diff * diff;
    stats.max_abs = std::max(stats.max_abs, ad);
    re += ad / (std::fabs(static_cast<double>(ref[i])) + 1e-8);
  }
  const double n = static_cast<double>(ref.size());
  stats.mae = ae / n;
  stats.rmse = std::sqrt(se / n);
  stats.mean_rel = re / n;
  return stats;
}

void PrintErrorRow(const std::string& label, const ErrorStats& s) {
  std::cout << std::left << std::setw(34) << label
            << std::setw(14) << std::scientific << std::setprecision(4) << s.mae
            << std::setw(14) << s.rmse
            << std::setw(14) << s.max_abs
            << std::setw(14) << s.mean_rel << "\n";
}

}  // namespace

int main() {
  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/fused_gated_delta_rule_quant_eval.csv");
  ofs << "B,L,D,H,scheme,mae,rmse,max_abs,mean_rel\n";

  std::vector<std::tuple<int, int, int, int>> test_cases = {
      {1, 128, 64, 32},
      {1, 256, 128, 64},
      {2, 512, 128, 64},
      {2, 512, 256, 128},
  };

  std::cout << "=== Fused Gated Delta Rule INT8 Quantization Accuracy Eval ===\n";
  std::cout << "Compensation: SmoothQuant(alpha=" << kSmoothAlpha
            << ") + per-channel weight quantization\n\n";

  for (const auto& tc : test_cases) {
    int B = std::get<0>(tc);
    int L = std::get<1>(tc);
    int D = std::get<2>(tc);
    int H = std::get<3>(tc);

    std::vector<float> x(static_cast<size_t>(B) * L * D);
    std::vector<float> x_calib(static_cast<size_t>(B) * L * D);
    std::vector<float> W_decay(static_cast<size_t>(H) * D), b_decay(H);
    std::vector<float> W_delta(static_cast<size_t>(H) * D), b_delta(H);
    std::vector<float> W_state(static_cast<size_t>(H) * D), b_state(H);

    std::mt19937 gen(42 + B * 1000 + L);
    InitRandom(x, gen);
    InitRandom(x_calib, gen);
    InitRandom(W_decay, gen);
    InitRandom(b_decay, gen);
    InitRandom(W_delta, gen);
    InitRandom(b_delta, gen);
    InitRandom(W_state, gen);
    InitRandom(b_state, gen);

    // Make activation distribution long-tailed to expose per-tensor quant weakness.
    InjectActivationOutliers(x, B, L, D, gen);
    InjectActivationOutliers(x_calib, B, L, D, gen);

    std::vector<float> out_ref;
    std::vector<float> out_int8_naive;
    std::vector<float> out_int8_comp;

    RunFp32Reference(x, W_decay, b_decay, W_delta, b_delta, W_state, b_state,
                     B, L, D, H, out_ref);
    RunInt8PerTensor(x, x_calib, W_decay, b_decay, W_delta, b_delta, W_state, b_state,
                     B, L, D, H, out_int8_naive);
    RunInt8SmoothQuantPerChannel(x, x_calib, W_decay, b_decay, W_delta, b_delta,
                                 W_state, b_state, B, L, D, H, out_int8_comp);

    ErrorStats e_naive = ComputeError(out_ref, out_int8_naive);
    ErrorStats e_comp = ComputeError(out_ref, out_int8_comp);

    std::cout << "Case (B,L,D,H)=(" << B << "," << L << "," << D << "," << H << ")\n";
    std::cout << std::left << std::setw(34) << "scheme"
              << std::setw(14) << "MAE"
              << std::setw(14) << "RMSE"
              << std::setw(14) << "MAX_ABS"
              << std::setw(14) << "MEAN_REL" << "\n";
    std::cout << std::string(90, '-') << "\n";
    PrintErrorRow("INT8 per-tensor (baseline)", e_naive);
    PrintErrorRow("INT8 SmoothQuant + per-channel", e_comp);

    double mae_gain = (e_naive.mae > 0.0) ? (1.0 - e_comp.mae / e_naive.mae) * 100.0 : 0.0;
    double rmse_gain = (e_naive.rmse > 0.0) ? (1.0 - e_comp.rmse / e_naive.rmse) * 100.0 : 0.0;
    std::cout << "Compensation gain: MAE "
              << std::fixed << std::setprecision(2) << mae_gain
              << "%, RMSE " << rmse_gain << "%\n\n";

    ofs << B << "," << L << "," << D << "," << H << ",int8_per_tensor,"
        << e_naive.mae << "," << e_naive.rmse << "," << e_naive.max_abs << ","
        << e_naive.mean_rel << "\n";
    ofs << B << "," << L << "," << D << "," << H << ",int8_smoothquant_per_channel,"
        << e_comp.mae << "," << e_comp.rmse << "," << e_comp.max_abs << ","
        << e_comp.mean_rel << "\n";
  }

  std::cout << "CSV saved to data/results/fused_gated_delta_rule_quant_eval.csv\n";
  return 0;
}
