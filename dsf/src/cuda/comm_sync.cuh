/*!
 *  Copyright (c) 2022 by Contributors
 */
#ifndef DGL_DSF_CUDA_COMM_SYNC_H_
#define DGL_DSF_CUDA_COMM_SYNC_H_

#include <cuda_runtime.h>
#include <cstdint>

namespace dgl {
namespace dsf {

class WaitFlag {
 public:
  __host__ __device__ __forceinline__ WaitFlag(volatile uint64_t* const flag)
      : flag_(flag) {}
  __device__ uint64_t get_flag() { return *flag_; }
  __device__ __forceinline__ void unset() { post(FLAG_UNUSED); }
  __device__ __forceinline__ void wait_unset() { wait(FLAG_UNUSED); }

  __device__ __forceinline__ void wait(uint64_t val) {
    /*SPIN*/
    while ((*flag_) != val) {
    }
  }
  __device__ __forceinline__ void post(uint64_t val) { *flag_ = val; }
  static constexpr uint64_t FLAG_UNUSED = ~0ull >> 1;

 private:
  volatile uint64_t* const flag_;
};

class CommSync {
 public:
  __host__ __device__ CommSync(uint64_t* ready, uint64_t* done,
                               uint64_t* peer_ready, uint64_t* peer_done,
                               bool control)
      : ready_(ready),
        done_(done),
        peer_ready_(peer_ready),
        peer_done_(peer_done),
        control_(control) {}

  __device__ void Unset() {
    if (control_) {
      step_ = 0;
      ready_.unset();
      peer_ready_.wait_unset();
      done_.unset();
      peer_done_.wait_unset();
    }
    __syncthreads();
  }
  __device__ void PreComm() {
    if (control_) {
      ++step_;
      ready_.post(step_);
      peer_ready_.wait(step_);
    }
    __syncthreads();
  }
  __device__ void PostComm() {
    __threadfence_system();
    if (control_) {
      done_.post(step_);
      peer_done_.wait(step_);
    }
    __syncthreads();
  }

 private:
  WaitFlag ready_, done_, peer_ready_, peer_done_;
  bool control_;
  uint64_t step_;
};

}  // namespace dsf
}  // namespace dgl

#endif  // DGL_DSF_CUDA_COMM_SYNC_H_