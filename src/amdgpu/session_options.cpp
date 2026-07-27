// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "session_options.h"

#include "../generators.h"
#include "../models/session_options.h"
#include "interface.h"

#include <filesystem>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace Generators::AMDGPUExecutionProvider {

namespace {

constexpr auto ep_registration_name_ = "AMDGPUExecutionProvider";
#if defined(_WIN32)
constexpr auto ep_filename_ = "amdgpu-ep.dll";
#else
constexpr auto ep_filename_ = "libamdgpu-ep.so";
#endif

// The AMDGPU umbrella is a plugin EP: unlike legacy in-proc EPs it must be registered on the
// OrtEnv via RegisterExecutionProviderLibrary before AppendExecutionProvider_V2 can find its
// OrtEpDevice. Mirror the RyzenAI interface's self-registration (resolve the DLL next to the
// genai/ort module or the executable) so the C model_benchmark -- which has no --ep_library flag
// -- still loads the umbrella. Registration is keyed per-OrtEnv; re-registration is benign.
void EnsureUmbrellaEpRegistered() {
  namespace fs = std::filesystem;
  std::error_code ec;

  fs::path ep_path;

#if defined(_WIN32)
  const auto hmod_of = [](LPCVOID func) -> HMODULE {
    MEMORY_BASIC_INFORMATION mbi;
    if (VirtualQuery(func, &mbi, sizeof(mbi)) && mbi.AllocationBase) {
      return reinterpret_cast<HMODULE>(mbi.AllocationBase);
    }
    return nullptr;
  };

  const auto find_next_to_module = [&](HMODULE hmod) -> fs::path {
    wchar_t buffer[MAX_PATH + 1] = {0};
    if (GetModuleFileNameW(hmod, buffer, MAX_PATH + 1)) {
      if (auto dir = fs::path{buffer}.remove_filename(); !dir.empty()) {
        if (auto candidate = dir / ep_filename_; fs::exists(candidate, ec)) {
          return candidate;
        }
      }
    }
    return {};
  };

  if (ep_path.empty()) {
    // next to onnxruntime-genai.dll (GetAMDGPUInterface is a genai symbol in that module)
    if (const auto hmod = hmod_of(reinterpret_cast<LPCVOID>(&GetAMDGPUInterface))) {
      ep_path = find_next_to_module(hmod);
    }
  }
  if (ep_path.empty()) {
    // next to onnxruntime.dll
    if (const auto hmod = hmod_of(reinterpret_cast<LPCVOID>(Ort::api->RegisterExecutionProviderLibrary))) {
      ep_path = find_next_to_module(hmod);
    }
  }
  if (ep_path.empty()) {
    // next to the current executable
    if (const auto hmod = GetModuleHandleA(nullptr)) {
      ep_path = find_next_to_module(hmod);
    }
  }
#endif  // _WIN32

  if (ep_path.empty()) {
    ep_path = fs::current_path(ec) / ep_filename_;
  }

  try {
    Ort::RegisterExecutionProviderLibrary(&GetOrtEnv(), ep_registration_name_, ep_path.native().c_str());
  } catch (const Ort::Exception& e) {
    if (std::string(e.what()).find("already registered") == std::string::npos) {
      throw std::runtime_error("Failed to register AMDGPU execution provider library from '" +
                               ep_path.string() + "': " + e.what());
    }
  }
}

// Emit static-padding hints so the EP pads the prefill token axis to max_length and
// compiles it once, instead of recompiling per prompt length.
void SetStaticPaddingConfig(OrtSessionOptions& session_options, const Config& config) {
  const auto& decoder = config.model.decoder;
  const std::string seq_len = std::to_string(config.search.max_length);
  const std::string pad_inputs =
      decoder.inputs.input_ids + ":1," + decoder.inputs.position_ids + ":1";
  const std::string pad_outputs = decoder.outputs.logits + ":1";

  session_options.AddConfigEntry("ep.migraphx.static_pad_seq", "1");
  session_options.AddConfigEntry("ep.migraphx.static_pad_seq_len", seq_len.c_str());
  session_options.AddConfigEntry("ep.migraphx.static_pad_inputs", pad_inputs.c_str());
  session_options.AddConfigEntry("ep.migraphx.static_pad_outputs", pad_outputs.c_str());

  session_options.AddConfigEntry("ep.migraphx.hip_graph_enable", "1");
}

// The umbrella EP consumes its own provider options (e.g. "profile") from session-config
// entries prefixed "ep.amdgpuexecutionprovider." (see the umbrella's CreateEp), not from the
// AppendExecutionProvider_V2 ep_options channel -- those do not surface through
// GetSessionOptionsConfigEntries. Bridge the genai_config provider options into that prefixed
// form so the umbrella selects the requested backend (profile=hip -> hipgpu backend).
void ForwardUmbrellaProviderOptions(OrtSessionOptions& session_options,
                                    const Config::ProviderOptions& provider_options) {
  const std::string umbrella_prefix = "ep.amdgpuexecutionprovider.";
  for (const auto& [key, value] : provider_options.options) {
    session_options.AddConfigEntry((umbrella_prefix + key).c_str(), value.c_str());
  }
}

}  // namespace

DeviceInterface* AppendExecutionProvider(OrtSessionOptions& session_options,
                                         const Config::ProviderOptions& provider_options,
                                         const Config& config,
                                         bool /*disable_graph_capture*/) {
  SetStaticPaddingConfig(session_options, config);

  // Umbrella-level hint: the model architecture drives the EP's backend routing.
  session_options.AddConfigEntry("ep.amdgpuexecutionprovider.model_arch", config.model.type.c_str());

  // DirectML backend: host-accessible decode inputs.
  session_options.AddConfigEntry("ep.directml.enable_host_accessible", "1");

  EnsureUmbrellaEpRegistered();
  ForwardUmbrellaProviderOptions(session_options, provider_options);

  AppendExecutionProviderV2(session_options, provider_options,
                            DeviceType::AMDGPU, ep_registration_name_);

  return GetAMDGPUInterface();
}

}  // namespace Generators::AMDGPUExecutionProvider
