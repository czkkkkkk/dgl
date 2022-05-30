/*!
 *  Copyright (c) 2022 by Contributors
 */
#ifndef DGL_DSF_UTILS_H_
#define DGL_DSF_UTILS_H_

#include <cstdlib>
#include <string>
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

}  // namespace dsf
}  // namespace dgl

#endif  // DGL_DSF_UTILS_H_