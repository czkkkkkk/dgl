/*!
 *  Copyright (c) 2022 by Contributors
 */
#ifndef DGL_DSF_UTILS_H_
#define DGL_DSF_UTILS_H_

#include <cuda_runtime.h>
#include <cstdlib>
#include <string>
#include <sstream>
#include <vector>

namespace dgl {
namespace dsf {

using IdType = int64_t;
static constexpr int MAX_CONN = 8;

template <typename T>
T GetEnvParam(const std::string &key, T default_value) {
  auto new_key = std::string("DGL_DSF_") + key;
  char *ptr = std::getenv(new_key.c_str());
  if (ptr == nullptr) return default_value;
  std::stringstream converter(ptr);
  T ret;
  converter >> ret;
  return ret;
}

template <typename T>
T GetEnvParam(const char *str, T default_value) {
  return GetEnvParam<T>(std::string(str), default_value);
}

template <typename T>
void SetEnvParam(const std::string &key, T value) {
  auto new_key = std::string("DGL_DSF_") + key;
  setenv(new_key.c_str(), std::to_string(value).c_str(), 1);
}
template <typename T>
void SetEnvParam(const char *key, T value) {
  SetEnvParam<T>(std::string(key), value);
}

int GetAvailablePort();
std::string GetHostName();

template <typename T>
std::vector<T> Flatten(const std::vector<std::vector<T>> &input) {
  std::vector<T> output;
  for (const auto &vec : input) {
    for (auto v : vec) {
      output.push_back(v);
    }
  }
  return output;
}

#define CUDACHECK(cmd)                                      \
  do {                                                      \
    cudaError_t e = cmd;                                    \
    if (e != cudaSuccess) {                                 \
      LOG(FATAL) << "Cuda error " << cudaGetErrorString(e); \
    }                                                       \
  } while (false);

#define SYSCHECK(call, name)                                     \
  do {                                                           \
    int ret = -1;                                                \
    while (ret == -1) {                                          \
      SYSCHECKVAL(call, name, ret);                              \
      if (ret == -1) {                                           \
        LOG(ERROR) << "Got " << strerror(errno) << ", retrying"; \
      }                                                          \
    }                                                            \
  } while (0);

#define SYSCHECKVAL(call, name, retval)                                    \
  do {                                                                     \
    retval = call;                                                         \
    if (retval == -1 && errno != EINTR && errno != EWOULDBLOCK &&          \
        errno != EAGAIN) {                                                 \
      LOG(ERROR) << "Call to " << name << " failed : " << strerror(errno); \
    }                                                                      \
  } while (0);

template <typename T>
void DSFCudaMalloc(T **ptr) {
  cudaMalloc(ptr, sizeof(T));
}

template <typename T>
void DSFCudaMalloc(T **ptr, int size) {
  cudaMalloc(ptr, sizeof(T) * size);
  cudaMemset(*ptr, 0, sizeof(T) * size);
}

template <typename T>
void DSFCUDAMallocAndCopy(T **ret, const T *src, int size) {
  DSFCudaMalloc(ret, size);
  cudaMemcpy(*ret, src, sizeof(T) * size, cudaMemcpyHostToDevice);
}

template <typename T>
void DSFCUDAMallocAndCopy(T **ret, const std::vector<T> &src) {
  DSFCUDAMallocAndCopy(ret, src.data(), src.size());
}

}  // namespace dsf
}  // namespace dgl

#endif  // DGL_DSF_UTILS_H_
