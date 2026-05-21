#pragma once

#include <cstring>
#include <string>

namespace common {

enum class StatusCode : int {
  kOk = 0,
  kInvalidArgument = 1,
  kUnsupported = 2,
  kUnimplemented = 3,
  kOutOfMemory = 4,
  kCudaError = 5,
  kNumericalError = 6,
  kSkip = 7,
};

struct Status {
  StatusCode code = StatusCode::kOk;
  std::string message;

  static Status Ok() { return {}; }

  static Status InvalidArgument(std::string msg) {
    return {StatusCode::kInvalidArgument, std::move(msg)};
  }

  static Status Unsupported(std::string msg) {
    return {StatusCode::kUnsupported, std::move(msg)};
  }

  static Status Unimplemented(std::string msg) {
    return {StatusCode::kUnimplemented, std::move(msg)};
  }

  static Status OutOfMemory(std::string msg) {
    return {StatusCode::kOutOfMemory, std::move(msg)};
  }

  static Status CudaError(std::string msg) {
    return {StatusCode::kCudaError, std::move(msg)};
  }

  static Status NumericalError(std::string msg) {
    return {StatusCode::kNumericalError, std::move(msg)};
  }

  static Status Skip(std::string msg) {
    return {StatusCode::kSkip, std::move(msg)};
  }

  bool ok() const { return code == StatusCode::kOk; }
  explicit operator bool() const { return ok(); }
};

inline const char* StatusCodeName(StatusCode code) {
  switch (code) {
    case StatusCode::kOk: return "ok";
    case StatusCode::kInvalidArgument: return "invalid_argument";
    case StatusCode::kUnsupported: return "unsupported";
    case StatusCode::kUnimplemented: return "unimplemented";
    case StatusCode::kOutOfMemory: return "out_of_memory";
    case StatusCode::kCudaError: return "cuda_error";
    case StatusCode::kNumericalError: return "numerical_error";
    case StatusCode::kSkip: return "skip";
  }
  return "unknown";
}

// 将 StatusCode 映射为整数供 extern "C" 调用方使用（0 = 成功）。
inline int StatusToC(StatusCode code) { return static_cast<int>(code); }

inline void CopyStatusMessage(const Status& st, char* buf, size_t len) {
  if (!buf || len == 0) return;
  std::strncpy(buf, st.message.c_str(), len - 1);
  buf[len - 1] = '\0';
}

}  // namespace common

#ifdef __cplusplus
extern "C" {
#endif

// 共享 C ABI 状态码（镜像 common::StatusCode）。
enum CkoStatus {
  CKO_OK = 0,
  CKO_INVALID_ARGUMENT = 1,
  CKO_UNSUPPORTED = 2,
  CKO_UNIMPLEMENTED = 3,
  CKO_OUT_OF_MEMORY = 4,
  CKO_CUDA_ERROR = 5,
  CKO_NUMERICAL_ERROR = 6,
  CKO_SKIP = 7,
};

#ifdef __cplusplus
}
#endif
