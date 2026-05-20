// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Stage A scope (B003): GPU-resident KV cache for the MIGraphX EP path.
//
// Allocation uses the Ort::Allocator obtained from the MIGraphX session
// (which fronts the EP's HIP device allocator). Cross-device copies are
// intentionally not implemented in Stage A — for normal decode the KV cache
// stays GPU-resident and OGA never materializes it to CPU. If a code path
// hits one of the throw sites below, that path is the new investigation
// target (either implement it via ORT's cross-device APIs, or — if HIP is
// truly necessary — add ${ROCM_HOME}/include to the build and switch the
// stubs to direct hipMemcpy/hipMemset, mirroring the design originally
// drafted in commit f337bce5's interface.cpp body).

#include "../generators.h"
#include "../search.h"
#include "../cpu/interface.h"
#include "interface.h"

#include <stdexcept>

namespace Generators {
namespace MIGraphX {

// One global Ort::Allocator bound to the HIP device memory the MIGraphX EP uses.
// Populated by InitOrt(), which is called by Model::EnsureDeviceOrtInit() once
// the EP session is created.
Ort::Allocator* ort_allocator_{};
const char* device_label = "migraphx";

[[noreturn]] static void NotImplemented(const char* method) {
  throw std::runtime_error(
      std::string("MIGraphX OGA interface: ") + method +
      " not implemented for Stage A. KV cache is expected to remain "
      "GPU-resident throughout decode; reaching this throw means an OGA path "
      "is materializing GPU memory to CPU (or vice versa). Inspect the call "
      "stack and either route the path through ORT cross-device APIs or add "
      "HIP support to OGA's build (see commit f337bce5 for the original "
      "HIP-direct implementation that would slot in here).");
}

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

  // Stage A: cross-device copies intentionally not implemented.
  // See file header comment for the rationale and how to implement when needed.
  void CopyDeviceToCpu() override { NotImplemented("CopyDeviceToCpu"); }
  void CopyCpuToDevice() override { NotImplemented("CopyCpuToDevice"); }
  void CopyFrom(size_t /*begin_dest*/, DeviceBuffer& /*source*/,
                size_t /*begin_source*/, size_t /*size_in_bytes*/) override {
    NotImplemented("CopyFrom");
  }
  void Zero() override { NotImplemented("Zero"); }

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
    // GPU-side sampling/search is Stage D scope — see B003 brief.
    return GetCpuInterface()->CreateGreedy(params);
  }

  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override {
    return GetCpuInterface()->CreateBeam(params);
  }

  // The MIGraphX EP owns its HIP stream and synchronizes around its own kernel
  // execution. Matches the DML interface pattern (`void Synchronize() override {}`).
  void Synchronize() override {}
};

}  // namespace MIGraphX

static std::unique_ptr<MIGraphX::InterfaceImpl> g_migraphx_device;

DeviceInterface* GetMIGraphXInterface() {
  if (!g_migraphx_device)
    g_migraphx_device = std::make_unique<MIGraphX::InterfaceImpl>();
  return g_migraphx_device.get();
}

}  // namespace Generators
