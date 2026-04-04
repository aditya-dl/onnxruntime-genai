// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "session_options.h"

#include "../models/session_options.h"

namespace Generators::MIGraphXExecutionProvider {

DeviceInterface* AppendExecutionProvider(OrtSessionOptions& session_options,
                                         const Config::ProviderOptions& provider_options,
                                         const Config& config,
                                         bool /*disable_graph_capture*/) {
  // MIGraphX does not have a device type specific allocator in OGA yet,
  // so we use CPU as the device type. The MIGraphX EP handles CPU<->GPU
  // transfers internally.
  if (!AppendExecutionProviderV2(session_options, provider_options,
                                 DeviceType::CPU, "MIGraphXExecutionProvider")) {
    AppendExecutionProviderV1(session_options, provider_options);
  }

  return nullptr;
}

}  // namespace Generators::MIGraphXExecutionProvider
