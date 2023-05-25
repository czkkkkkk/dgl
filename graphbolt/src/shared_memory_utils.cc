/**
 *  Copyright (c) 2023 by Contributors
 *
 * @file shared_memory_utils.cc
 * @brief Share memory utility function implementation.
 */
#include "./shared_memory_utils.h"

#include <graphbolt/serialize.h>
#include <graphbolt/shared_memory.h>

namespace graphbolt {
namespace sampling {

static SharedMemoryPtr _CopyTorchArchiveToSharedMemory(
    const std::string& shared_memory_name, int64_t memory_size,
    torch::serialize::OutputArchive& archive) {
  std::stringstream serialized;
  archive.save_to(serialized);
  auto serialized_str = serialized.str();
  auto shm = std::make_unique<SharedMemory>(shared_memory_name);
  auto mem_buf = shm->Create(memory_size);
  // Use the first 8 bytes to store the size of the serialized string.
  static_cast<int64_t*>(mem_buf)[0] = serialized_str.size();
  memcpy(
      (char*)mem_buf + sizeof(int64_t), serialized_str.data(),
      serialized_str.size());
  return shm;
}

static std::pair<SharedMemoryPtr, torch::serialize::InputArchive>
_LoadTorchArchiveFromSharedMemory(
    const std::string& shared_memory_name, int64_t memory_size) {
  auto shm = std::make_unique<SharedMemory>(shared_memory_name);
  auto mem_buf = shm->Open(memory_size);
  torch::serialize::InputArchive archive;
  int64_t size = static_cast<int64_t*>(mem_buf)[0];
  archive.load_from(static_cast<const char*>(mem_buf) + sizeof(int64_t), size);
  return {std::move(shm), std::move(archive)};
}

static SharedMemoryPtr _CopyTensorsDataToSharedMemory(
    const std::string& shared_memory_name,
    const std::vector<torch::Tensor>& tensors) {
  int64_t memory_size = 0;
  for (const auto& tensor : tensors) {
    memory_size += tensor.numel() * tensor.element_size();
  }
  auto shm = std::make_unique<SharedMemory>(shared_memory_name);
  auto mem_buf = shm->Create(memory_size);
  for (auto tensor : tensors) {
    tensor = tensor.contiguous();
    int64_t size = tensor.numel() * tensor.element_size();
    memcpy(mem_buf, tensor.data_ptr(), size);
    mem_buf = static_cast<char*>(mem_buf) + size;
  }
  return shm;
}

/**
 * @brief Load tensors data from shared memory.
 * @param shared_memory_name The name of shared memory.
 * @param tensor_metas The meta info of tensors, including tensor shape and
 * dtype.
 *
 * @return A pair of shared memory holing the tensors and the tensors.
 */
static std::pair<SharedMemoryPtr, std::vector<torch::Tensor>>
_LoadTensorsDataFromSharedMemory(
    const std::string& shared_memory_name,
    const std::vector<std::pair<std::vector<int64_t>, torch::ScalarType>>&
        tensor_metas) {
  SharedMemoryPtr shm;
  int64_t memory_size = 0;
  for (const auto& meta : tensor_metas) {
    int64_t size = std::accumulate(
        meta.first.begin(), meta.first.end(), 1, std::multiplies<int64_t>());
    memory_size += size * torch::elementSize(meta.second);
  }
  auto mem_buf = shm->Open(memory_size);
  std::vector<torch::Tensor> tensors;
  for (const auto& meta : tensor_metas) {
    auto tensor = torch::from_blob(mem_buf, meta.first, meta.second);
    tensors.push_back(tensor);
    int64_t size = std::accumulate(
        meta.first.begin(), meta.first.end(), 1, std::multiplies<int64_t>());
    mem_buf = static_cast<char*>(mem_buf) + size;
  }
  return {std::move(shm), tensors};
}

std::tuple<SharedMemoryPtr, SharedMemoryPtr, std::vector<torch::Tensor>>
CopyTensorsToSharedMemory(
    const std::string& shared_memory_name,
    const std::vector<torch::Tensor>& tensors, int64_t meta_memory_size) {
  torch::serialize::OutputArchive archive;
  archive.write("num_tensors", static_cast<int64_t>(tensors.size()));
  for (size_t i = 0; i < tensors.size(); ++i) {
    archive.write("tensor_" + std::to_string(i) + "_shape", tensors[i].sizes());
    archive.write(
        "tensor_" + std::to_string(i) + "_dtype", tensors[i].scalar_type());
  }
  auto meta_shm = _CopyTorchArchiveToSharedMemory(
      shared_memory_name + "_meta", meta_memory_size, archive);
  auto data_shm =
      _CopyTensorsDataToSharedMemory(shared_memory_name + "_data", tensors);

  std::vector<torch::Tensor> ret_tensors;
  auto mem_buf = data_shm->GetMemory();
  for (auto tensor : tensors) {
    int64_t size = tensor.numel() * tensor.element_size();
    memcpy(mem_buf, tensor.data_ptr(), size);
    mem_buf = static_cast<char*>(mem_buf) + size;
    ret_tensors.push_back(
        torch::from_blob(mem_buf, tensor.sizes(), tensor.dtype()));
  }
  return {std::move(meta_shm), std::move(data_shm), std::move(ret_tensors)};
}

std::tuple<SharedMemoryPtr, SharedMemoryPtr, std::vector<torch::Tensor>>
LoadTensorsFromSharedMemory(
    const std::string& shared_memory_name, int64_t meta_memory_size) {
  SharedMemoryPtr meta_shm;
  torch::serialize::InputArchive archive;
  std::tie(meta_shm, archive) = _LoadTorchArchiveFromSharedMemory(
      shared_memory_name + "_meta", meta_memory_size);
  std::vector<std::pair<std::vector<int64_t>, torch::ScalarType>> metas;
  int64_t num_tensors = read_from_archive(archive, "num_tensors").toInt();
  for (int64_t i = 0; i < num_tensors; ++i) {
    auto shape =
        read_from_archive(archive, "tensor_" + std::to_string(i) + "_shape")
            .toIntVector();
    auto dtype =
        read_from_archive(archive, "tensor_" + std::to_string(i) + "_dtype")
            .toScalarType();
    metas.push_back({shape, dtype});
  }
  SharedMemoryPtr data_shm;
  std::vector<torch::Tensor> ret_tensors;
  std::tie(data_shm, ret_tensors) =
      _LoadTensorsDataFromSharedMemory(shared_memory_name + "_data", metas);
  return {std::move(meta_shm), std::move(data_shm), std::move(ret_tensors)};
}

}  // namespace sampling
}  // namespace graphbolt
