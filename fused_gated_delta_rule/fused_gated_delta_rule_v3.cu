// Host-side MAE study: INT8-simulated weights with vs without compensation on Gated Delta Rule.
// Reference: FP32 forward. Shapes configurable; default includes (B,L,D,H)=(2,512,256,128).
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

static double MeanAbsError(const std::vector<float>& ref, const std::vector<float>& tst) {
  double s = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) s += std::fabs(static_cast<double>(ref[i] - tst[i]));
  return s / static_cast<double>(ref.size());
}

static void QuantizePerTensor(const std::vector<float>& src, std::vector<int8_t>& q,
                              float& scale) {
  float max_abs = 0.0f;
  for (float v : src) max_abs = std::max(max_abs, std::fabs(v));
  scale = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
  q.resize(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    float x = std::round(src[i] / scale);
    x = std::max(-128.0f, std::min(127.0f, x));
    q[i] = static_cast<int8_t>(x);
  }
}

// Each row (out channel h) has its own scale — typical "per-channel" weight quant.
static void QuantizePerRow2D(const std::vector<float>& W, int H, int D,
                            std::vector<int8_t>& q, std::vector<float>& scales) {
  q.resize(W.size());
  scales.assign(static_cast<size_t>(H), 1.0f);
  for (int h = 0; h < H; ++h) {
    float max_abs = 0.0f;
    for (int d = 0; d < D; ++d) max_abs = std::max(max_abs, std::fabs(W[static_cast<size_t>(h) * D + d]));
    scales[static_cast<size_t>(h)] = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
    float sc = scales[static_cast<size_t>(h)];
    for (int d = 0; d < D; ++d) {
      size_t i = static_cast<size_t>(h) * D + d;
      float x = std::round(W[i] / sc);
      x = std::max(-128.0f, std::min(127.0f, x));
      q[i] = static_cast<int8_t>(x);
    }
  }
}

static void DequantPerTensor(const std::vector<int8_t>& q, float scale, std::vector<float>& out) {
  out.resize(q.size());
  for (size_t i = 0; i < q.size(); ++i) out[i] = static_cast<float>(q[i]) * scale;
}

static void DequantPerRow2D(const std::vector<int8_t>& q, int H, int D,
                            const std::vector<float>& scales, std::vector<float>& out) {
  out.resize(q.size());
  for (int h = 0; h < H; ++h) {
    float sc = scales[static_cast<size_t>(h)];
    for (int d = 0; d < D; ++d) {
      size_t i = static_cast<size_t>(h) * D + d;
      out[i] = static_cast<float>(q[i]) * sc;
    }
  }
}

// If non-null, adds FP32 residual (W_fp - W_q) @ x only to the state projection u (quant compensation demo).
static void GatedDeltaRule_CPU(const float* x,
                               const float* W_decay, const float* b_decay,
                               const float* W_delta, const float* b_delta,
                               const float* W_state, const float* b_state,
                               const float* W_state_fp_residual,  // H×D, may be nullptr
                               float* output,
                               int B, int L, int D, int H) {
  std::vector<float> alpha(static_cast<size_t>(B * L * H));
  std::vector<float> delta(static_cast<size_t>(B * L * H));
  std::vector<float> u(static_cast<size_t>(B * L * H));

  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < L; ++t) {
      const float* x_bt = x + (b * L + t) * D;
      for (int h = 0; h < H; ++h) {
        float a = b_decay[h];
        for (int d = 0; d < D; ++d) a += x_bt[d] * W_decay[h * D + d];
        a = 1.0f / (1.0f + std::exp(-a));
        alpha[(b * L + t) * H + h] = a;

        float dlt = b_delta[h];
        for (int d = 0; d < D; ++d) dlt += x_bt[d] * W_delta[h * D + d];
        dlt = std::max(dlt, 0.0f) + std::log(1.0f + std::exp(-std::abs(dlt)));
        delta[(b * L + t) * H + h] = dlt;

        float v = b_state[h];
        for (int d = 0; d < D; ++d) v += x_bt[d] * W_state[h * D + d];
        if (W_state_fp_residual != nullptr) {
          for (int d = 0; d < D; ++d) v += x_bt[d] * W_state_fp_residual[h * D + d];
        }
        u[(b * L + t) * H + h] = v;
      }
    }
  }

  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      float s = 0.0f;
      for (int t = 0; t < L; ++t) {
        int idx = (b * L + t) * H + h;
        s = alpha[idx] * s + delta[idx] * u[idx];
        output[idx] = s;
      }
    }
  }
}

// Per-channel output bias: corr[h] = mean_{b,t}(out_ref - out_naive); cheap calibration-style fix.
static void ApplyOutputBiasCompensation(const std::vector<float>& ref,
                                        const std::vector<float>& naive,
                                        std::vector<float>& compensated,
                                        int B, int L, int H) {
  std::vector<double> sum_h(static_cast<size_t>(H), 0.0);
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < L; ++t) {
      for (int h = 0; h < H; ++h) {
        size_t idx = static_cast<size_t>((b * L + t) * H + h);
        sum_h[static_cast<size_t>(h)] += static_cast<double>(ref[idx] - naive[idx]);
      }
    }
  }
  const double inv = 1.0 / static_cast<double>(B * L);
  std::vector<float> corr(static_cast<size_t>(H));
  for (int h = 0; h < H; ++h) corr[static_cast<size_t>(h)] = static_cast<float>(sum_h[static_cast<size_t>(h)] * inv);

  compensated.resize(ref.size());
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < L; ++t) {
      for (int h = 0; h < H; ++h) {
        size_t idx = static_cast<size_t>((b * L + t) * H + h);
        compensated[idx] = naive[idx] + corr[static_cast<size_t>(h)];
      }
    }
  }
}

int main() {
  constexpr int B = 2;
  constexpr int L = 512;
  constexpr int D = 256;
  constexpr int Hdim = 128;

  std::vector<float> x(static_cast<size_t>(B * L * D));
  std::vector<float> W_decay(static_cast<size_t>(Hdim * D)), b_decay(static_cast<size_t>(Hdim));
  std::vector<float> W_delta(static_cast<size_t>(Hdim * D)), b_delta(static_cast<size_t>(Hdim));
  std::vector<float> W_state(static_cast<size_t>(Hdim * D)), b_state(static_cast<size_t>(Hdim));

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  auto fill = [&](std::vector<float>& v) {
    for (auto& e : v) e = dist(gen);
  };
  fill(x);
  fill(W_decay);
  fill(b_decay);
  fill(W_delta);
  fill(b_delta);
  fill(W_state);
  fill(b_state);

  // Amplify projection weights so INT8 per-tensor scale clips harder; compensation contrasts become visible.
  constexpr float kWeightAmplify = 4.0f;
  for (auto& v : W_decay) v *= kWeightAmplify;
  for (auto& v : W_delta) v *= kWeightAmplify;
  for (auto& v : W_state) v *= kWeightAmplify;

  const size_t out_sz = static_cast<size_t>(B * L * Hdim);
  std::vector<float> out_ref(out_sz), out_pt(out_sz), out_pch(out_sz), out_bias(out_sz);

  GatedDeltaRule_CPU(x.data(), W_decay.data(), b_decay.data(), W_delta.data(), b_delta.data(),
                     W_state.data(), b_state.data(), nullptr, out_ref.data(), B, L, D, Hdim);

  // --- Per-tensor INT8 on each weight tensor + biases folded into float (bias kept FP32) ---
  std::vector<int8_t> qd, qz, qs;
  float sd, sz, ss;
  QuantizePerTensor(W_decay, qd, sd);
  QuantizePerTensor(W_delta, qz, sz);
  QuantizePerTensor(W_state, qs, ss);
  std::vector<float> Wd_q, Wz_q, Ws_q;
  DequantPerTensor(qd, sd, Wd_q);
  DequantPerTensor(qz, sz, Wz_q);
  DequantPerTensor(qs, ss, Ws_q);

  GatedDeltaRule_CPU(x.data(), Wd_q.data(), b_decay.data(), Wz_q.data(), b_delta.data(),
                     Ws_q.data(), b_state.data(), nullptr, out_pt.data(), B, L, D, Hdim);
  const double mae_pt = MeanAbsError(out_ref, out_pt);

  std::vector<float> W_state_res(static_cast<size_t>(Hdim * D));
  for (size_t i = 0; i < W_state.size(); ++i) W_state_res[i] = W_state[i] - Ws_q[i];
  std::vector<float> out_u_res(out_sz);
  GatedDeltaRule_CPU(x.data(), Wd_q.data(), b_decay.data(), Wz_q.data(), b_delta.data(),
                     Ws_q.data(), b_state.data(), W_state_res.data(), out_u_res.data(), B, L, D, Hdim);
  const double mae_u_res = MeanAbsError(out_ref, out_u_res);

  // --- Per-channel INT8 per weight matrix ---
  std::vector<int8_t> qd2, qz2, qs2;
  std::vector<float> sd_r, sz_r, ss_r;
  QuantizePerRow2D(W_decay, Hdim, D, qd2, sd_r);
  QuantizePerRow2D(W_delta, Hdim, D, qz2, sz_r);
  QuantizePerRow2D(W_state, Hdim, D, qs2, ss_r);
  std::vector<float> Wd_pc, Wz_pc, Ws_pc;
  DequantPerRow2D(qd2, Hdim, D, sd_r, Wd_pc);
  DequantPerRow2D(qz2, Hdim, D, sz_r, Wz_pc);
  DequantPerRow2D(qs2, Hdim, D, ss_r, Ws_pc);

  GatedDeltaRule_CPU(x.data(), Wd_pc.data(), b_decay.data(), Wz_pc.data(), b_delta.data(),
                     Ws_pc.data(), b_state.data(), nullptr, out_pch.data(), B, L, D, Hdim);
  const double mae_pch = MeanAbsError(out_ref, out_pch);

  ApplyOutputBiasCompensation(out_ref, out_pt, out_bias, B, L, Hdim);
  const double mae_bias = MeanAbsError(out_ref, out_bias);

  const double reduce_pch = (mae_pt > 0.0) ? (1.0 - mae_pch / mae_pt) * 100.0 : 0.0;
  const double reduce_bias = (mae_pt > 0.0) ? (1.0 - mae_bias / mae_pt) * 100.0 : 0.0;
  const double reduce_u_res = (mae_pt > 0.0) ? (1.0 - mae_u_res / mae_pt) * 100.0 : 0.0;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Gated Delta Rule — INT8 weight simulation (same seed=42 as fused_gated_delta_rule_v0)\n";
  std::cout << "(B,L,D,H)=(" << B << "," << L << "," << D << "," << Hdim << ")\n";
  std::cout << "W_decay/W_delta/W_state scaled x" << kWeightAmplify << " after init (stress quant).\n\n";
  std::cout << "MAE vs FP32 reference:\n";
  std::cout << "  (A) Per-tensor INT8 weights (naive):              MAE = " << mae_pt << "\n";
  std::cout << "  (B) Per-channel INT8 weights (finer quant):        MAE = " << mae_pch
            << "  (↓" << std::setprecision(2) << reduce_pch << "% vs A)\n";
  std::cout << std::setprecision(6);
  std::cout << "  (C) Per-tensor INT8 + per-head output bias calib: MAE = " << mae_bias
            << "  (↓" << std::setprecision(2) << reduce_bias << "% vs A)\n";
  std::cout << std::setprecision(6);
  std::cout << "  (D) Per-tensor INT8 + FP32 residual on W_state only: MAE = " << mae_u_res
            << "  (↓" << std::setprecision(2) << reduce_u_res << "% vs A)\n";

  std::filesystem::create_directories("data/results");
  std::ofstream ofs("data/results/fused_gated_delta_rule_compensation_mae.csv");
  ofs << "B,L,D,H,weight_amplify,mae_per_tensor_int8,mae_per_channel_int8,mae_per_tensor_plus_bias_calib,"
         "mae_per_tensor_plus_wstate_residual,pct_drop_pch_vs_pt,pct_drop_bias_vs_pt,pct_drop_u_res_vs_pt\n";
  ofs << B << "," << L << "," << D << "," << Hdim << "," << kWeightAmplify << "," << mae_pt << "," << mae_pch << ","
      << mae_bias << "," << mae_u_res << "," << reduce_pch << "," << reduce_bias << "," << reduce_u_res << "\n";
  std::cout << "\nSaved data/results/fused_gated_delta_rule_compensation_mae.csv\n";
  return 0;
}
