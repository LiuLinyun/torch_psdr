#pragma once

#include <exception>
#include <sstream>

#include "cuda_runtime.h"
#include "driver_types.h"

class PtxException : public std::runtime_error {
 public:
  PtxException(const char* msg) : std::runtime_error(msg) {}

  PtxException(OptixResult res, const char* msg)
      : std::runtime_error(createMessage(res, msg).c_str()) {}

 private:
  std::string createMessage(OptixResult res, const char* msg) {
    std::ostringstream out;
    out << optixGetErrorName(res) << ": " << msg;
    return out.str();
  }
};

#ifndef OPTIX_CHECK
#define OPTIX_CHECK(call)                                                    \
  do {                                                                       \
    OptixResult res = call;                                                  \
    if (res != OPTIX_SUCCESS) {                                              \
      std::stringstream ss;                                                  \
      ss << "Optix call '" << #call << "' failed: " __FILE__ ":" << __LINE__ \
         << ")\n";                                                           \
      throw PtxException(res, ss.str().c_str());                             \
    }                                                                        \
  } while (0)
#endif  // OPTIX_CHECK

#ifndef OPTIX_CHECK_LOG
#define OPTIX_CHECK_LOG(call)                                                \
  do {                                                                       \
    OptixResult res = call;                                                  \
    const size_t sizeof_log_returned = sizeof_log;                           \
    sizeof_log = sizeof(log); /* reset sizeof_log for future calls */        \
    if (res != OPTIX_SUCCESS) {                                              \
      std::stringstream ss;                                                  \
      ss << "Optix call '" << #call << "' failed: " __FILE__ ":" << __LINE__ \
         << ")\nLog:\n"                                                      \
         << log << (sizeof_log_returned > sizeof(log) ? "<TRUNCATED>" : "")  \
         << "\n";                                                            \
      throw PtxException(res, ss.str().c_str());                             \
    }                                                                        \
  } while (0)
#endif  // OPTIX_CHECK_LOG

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t error = call;                                              \
    if (error != cudaSuccess) {                                            \
      std::stringstream ss;                                                \
      ss << "CUDA call (" << #call << " ) failed with error: '"            \
         << cudaGetErrorString(error) << "' (" __FILE__ << ":" << __LINE__ \
         << ")\n";                                                         \
      throw PtxException(ss.str().c_str());                                \
    }                                                                      \
  } while (0)
#endif  // CUDA_CHECK

#ifndef CUDA_SYNC_CHECK
#define CUDA_SYNC_CHECK()                                                  \
  do {                                                                     \
    cudaDeviceSynchronize();                                               \
    cudaError_t error = cudaGetLastError();                                \
    if (error != cudaSuccess) {                                            \
      std::stringstream ss;                                                \
      ss << "CUDA error on synchronize with error '"                       \
         << cudaGetErrorString(error) << "' (" __FILE__ << ":" << __LINE__ \
         << ")\n";                                                         \
      throw PtxException(ss.str().c_str());                                \
    }                                                                      \
  } while (0)
#endif  // CUDA_SYNC_CHECK
