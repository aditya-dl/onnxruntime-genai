// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "../generators.h"
#include "../search.h"
#include "../cpu/interface.h"
#include "interface.h"

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

#define MIGRAPHX_HIP_CHECK(call)                                                         \
  do {                                                                                   \
    hipError_t err = (call);                                                             \
    if (err != hipSuccess)                                                               \
      throw std::runtime_error(std::string("MIGraphX OGA HIP error: ") +                 \
                               hipGetErrorString(err));                                  \
  } while (0)

namespace Generators {
namespace MIGraphX {

// One global Ort::Allocator bound to the HIP device memory the MIGraphX EP uses.
// Populated by InitOrt(), which is called by Model::EnsureDeviceOrtInit() once
// the EP session is created.
Ort::Allocator* ort_allocator_{};
const char* device_label = "migraphx";

struct GpuMemory final : DeviceBuffer {
  GpuMemory(size_t size) : owned_{true} {
    size_in_bytes_ = size;
    p_device_ = static_cast<uint8_t*>(ort_allocator_->Alloc(size_in_bytes_));
  }

  GpuMemory(void* p, size_t size) : owned_{false} {
    size_in_bytes_ = size;
    p_device_ = static_cast<uint8_t*>(p);
  }

  ~GpuMemory() override {
    if (owned_)
      ort_allocator_->Free(p_device_);
    if (p_cpu_)
      free(p_cpu_);
  }

  const char* GetType() const override { return device_label; }

  void AllocateCpu() override {
    if (!p_cpu_)
      p_cpu_ = static_cast<uint8_t*>(malloc(size_in_bytes_));
  }

  void CopyDeviceToCpu() override {
    AllocateCpu();
    // hipMemcpy without an explicit stream is synchronous; the MIGraphX EP
    // owns its own stream and synchronizes around kernel execution, so this
    // is safe to call from OGA's per-step path.
    MIGRAPHX_HIP_CHECK(hipMemcpy(p_cpu_, p_device_, size_in_bytes_, hipMemcpyDeviceToHost));
  }

  void CopyCpuToDevice() override {
    assert(p_cpu_);
    MIGRAPHX_HIP_CHECK(hipMemcpy(p_device_, p_cpu_, size_in_bytes_, hipMemcpyHostToDevice));
  }

  void CopyFrom(size_t begin_dest, DeviceBuffer& source, size_t begin_source, size_t size_in_bytes) override {
    if (source.GetType() == device_label) {
      MIGRAPHX_HIP_CHECK(hipMemcpy(p_device_ + begin_dest,
                                   source.p_device_ + begin_source,
                                   size_in_bytes,
                                   hipMemcpyDeviceToDevice));
    } else {
      CopyThroughCpu(*this, begin_dest, source, begin_source, size_in_bytes);
    }
  }

  void Zero() override {
    MIGRAPHX_HIP_CHECK(hipMemset(p_device_, 0, size_in_bytes_));
  }

  bool owned_;  // If we own the memory, we free it on destruction
};

struct InterfaceImpl : DeviceInterface {
  DeviceType GetType() const override { return DeviceType::MIGRAPHX; }

  void InitOrt(const OrtApi& api, Ort::Allocator& allocator) override {
    Ort::api = &api;
    assert(!ort_allocator_);
    ort_allocator_ = &allocator;
  }

  Ort::Allocator& GetAllocator() override {
    return *ort_allocator_;
  }

  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    return std::make_shared<GpuMemory>(size);
  }

  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* p, size_t size) override {
    return std::make_shared<GpuMemory>(p, size);
  }

  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override {
    // Sampling stays on CPU for the MIGraphX path (mirrors DML/WebGPU pattern).
    return GetCpuInterface()->CreateGreedy(params);
  }

  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override {
    return GetCpuInterface()->CreateBeam(params);
  }

  void Synchronize() override {
    MIGRAPHX_HIP_CHECK(hipDeviceSynchronize());
  }
};

}  // namespace MIGraphX

static std::unique_ptr<MIGraphX::InterfaceImpl> g_migraphx_device;

DeviceInterface* GetMIGraphXInterface() {
  if (!g_migraphx_device)
    g_migraphx_device = std::make_unique<MIGraphX::InterfaceImpl>();
  return g_migraphx_device.get();
}

}  // namespace Generators
