# MIGraphX Execution Provider + GPU-Resident KV Cache (OGA) — Handoff README

**Purpose:** adds MIGraphX execution-provider support to ONNX Runtime GenAI (OGA) and
keeps the KV cache resident on the GPU across decode steps, so generation no longer
copies the cache between host and device on every token.

**Quick links:** this branch is `rel-0.14.0-migraphx-support` (rebased on OGA `0.14.0`).
The GPU-KV portion is also forward-ported onto the `0.13.2` line for the internal-fork PR
— branch `amd/dev/adilohia/migraphx-support-rel-0.13.2` (fork
`https://github.com/aditya-dl/onnxruntime-genai`), targeting
[`apwojcik/onnxruntime-genai:rel-0.13.2-mgx`](https://github.com/apwojcik/onnxruntime-genai/tree/rel-0.13.2-mgx).

> ⚠️ **Status: built + validated, NOT merged to upstream.** The public upstream PR is
> held while an existing MIGraphX PR is still under review; the internal-fork PR
> (`rel-0.13.2-mgx`) carries the GPU-KV work in the meantime. Internal perf
> numbers/model names, if added below, are for reference — **keep them out of any public
> PR description.**

---

## 0. Background (read first if you're new to this stack)

Skip this section if you already know OGA and KV caches.

- **OGA (ONNX Runtime GenAI).** A library that drives generative-model inference (the
  token-by-token generation loop) on top of ONNX Runtime (ORT). It owns the KV cache,
  sampling, and the per-step model runs; ORT (via an execution provider) does the math.

- **Execution Provider (EP).** ORT's backend abstraction. The **MIGraphX EP** runs the
  model on AMD GPUs via MIGraphX (AMD's graph compiler/runtime). For OGA to use it well,
  OGA needs to know this is a GPU device and how to allocate device memory for it.

- **Prefill vs decode.** A language model answers in two phases — prefill (consume the
  whole prompt) and decode (emit one token at a time, feeding each new token back in).
  Decode runs the model once per output token.

- **KV cache.** During decode, the model reuses the attention keys/values computed for
  all previous tokens. Those are stored in the **KV cache** and grow by one position each
  step. It is large and read/written every single decode step.

- **The problem this addresses.** On the MIGraphX path, OGA treated the device as a
  CPU-style device: the KV cache lived in host (CPU) memory and was copied to the GPU
  before each model run and back afterward. That host↔device copy happens **once per
  token**, adding a fixed PCIe transfer cost to every decode step that grows with
  sequence length and model size.

- **DeviceInterface.** OGA's internal abstraction for "how to allocate/manage memory on a
  given device." CUDA, DML, WebGPU each have one. This work adds one for MIGraphX so the
  KV cache can be allocated and kept on the GPU instead of round-tripping to the host.

---

## 1. Original thesis

**Problem.** The MIGraphX EP path in OGA had no device interface, so OGA fell back to
treating it as a CPU device: the KV cache was a host tensor copied to/from the GPU every
decode step. For autoregressive generation that per-token PCIe round-trip is pure
overhead and scales with the cache size.

**Hypothesis.** If OGA has a MIGraphX `DeviceInterface` and registers MIGraphX as a GPU
device, the KV cache can stay resident in device memory across decode steps — the cache
one step produces is reused in place by the next — eliminating the per-token host↔device
copy, with no change to generated output.

---

## 2. Implementation (what was built)

This branch contains two layers of work, in order:

**(A) MIGraphX EP support** — makes OGA able to select and configure the MIGraphX EP
(config plumbing, session options, position-inputs handling). Files: `src/config.*`,
`src/migraphx/session_options.*`, `src/models/position_inputs.*`, `src/models/model.*`,
`src/generators.*`, `cmake/global_variables.cmake`.

**(B) GPU-resident KV cache** — the focus of this README, layered on top of (A). This is
the part forward-ported to the `0.13.2` internal-fork PR.

### `src/migraphx/interface.{cpp,h}` (new) — the device interface

```cpp
namespace Generators {
DeviceInterface* GetMIGraphXInterface();
}  // namespace Generators
```

**What the code does:** introduces a `DeviceInterface` implementation for the MIGraphX EP
(`interface.cpp`) and exposes it via `GetMIGraphXInterface()`. This is the object OGA uses
to allocate device memory for the MIGraphX path — the thing that lets the KV cache live on
the GPU instead of on the host.

### `src/migraphx/session_options.cpp` — wire the interface in

**What the code does:** registers the MIGraphX `DeviceInterface` into the session options
and replaces the previous "MIGraphX uses the CPU device type and handles transfers
internally" path with the GPU device path, so OGA routes MIGraphX through the device
interface above.

### `src/models/model.cpp` — register the device + memory type

```cpp
static const char* device_type_names[] = {..., "RyzenAI", "MIGraphX"};
static const char* device_memory_type_names[] = {..., "Cpu", "Hip"};
```

**What the code does:** adds `MIGraphX` to OGA's device-type table and `Hip` to its
device-memory-type table, so `EnsureDeviceOrtInit` can create the on-device allocator for
the MIGraphX EP. The memory type is `Hip` (it was initially mislabeled `Cuda`, which
caused an allocator-creation failure at runtime — see the in-code comment).

### `src/models/kv_cache.cpp` — skip per-step zeroing on the GPU path

**What the code does:** skips the per-step `Zero()` initialization of the KV cache for
MIGraphX (mirroring the existing WebGPU exception). On the GPU-resident path that
initialization is unnecessary and would add device work each step.

### `src/generators.cpp`, `src/smartptrs.h` — supporting plumbing

**What the code does:** the device-type enum value and buffer-lifetime plumbing needed by
the pieces above.

---

## 3. Verification done

*(Internal numbers, if cited, are for reference — keep out of any public PR.)*

| Check | Method | Result |
|---|---|---|
| Output correctness | greedy decoding, before vs after | same output tokens (measured) |
| KV cache stays on GPU | inspect per-step transfers on the MIGraphX path | no per-step host↔device copy of the cache (measured) |
| EP selection / config | run with MIGraphX EP selected | EP initializes and runs end-to-end (measured) |
| Memory-type naming | runtime allocator creation | `Hip` allocator created successfully (the earlier `Cuda` label failed at runtime; fixed) |
| Other EP paths | — | unchanged; all MIGraphX-specific paths are gated on the MIGraphX device/EP |

---

## 4. Status

- **Built and validated**, not merged to public upstream.
- **Public upstream PR is held** while an existing MIGraphX PR is under review (avoid
  stacking a second EP PR). The GPU-KV work is carried on the internal-fork PR
  (`apwojcik:rel-0.13.2-mgx`) in the meantime.
- **Branch split:** this branch (`rel-0.14.0-migraphx-support`) is the 0.14.0-based line;
  the 0.13.2 internal-fork PR branch carries the GPU-KV change cherry-picked/squashed onto
  the already-merged EP-support commit (so that PR's diff is GPU-KV only).
- **Payoff:** removes the per-token host↔device KV-cache copy on the MIGraphX path; output
  unchanged.

---

## 5. Known issues / risks

- **Scope is the KV cache, not all EP buffers.** This change keeps the *KV cache* resident
  on the GPU. It does **not** route the EP's other per-step input/output buffers through a
  device allocator (an earlier experiment that did so was reverted). Don't describe this as
  "all inputs/outputs on GPU."
- **MIGraphX-gated.** The device-type/memory-type table entries, the KV-cache `Zero()`
  exception, and the device interface are all conditioned on the MIGraphX device/EP, so
  other EP paths are unaffected. Any change to those shared tables must keep the
  `static_assert(... == DeviceType::MAX)` counts in sync.
- **Two-base maintenance.** The work exists on both the 0.14.0 line (this branch) and the
  0.13.2 line (the PR branch). Keep them in sync if the GPU-KV code changes.
- **Memory-type name is `Hip`.** It must match what the MIGraphX EP registers
  (`CreateMemoryInfo_V2("Hip", ...)`); a mismatch fails allocator creation at runtime.

---

## 6. Branches & artifacts

| Item | Value |
|---|---|
| This branch | `rel-0.14.0-migraphx-support` (base: OGA `0.14.0`) — full AMD work (EP support + GPU-KV) |
| Internal-fork PR branch | `amd/dev/adilohia/migraphx-support-rel-0.13.2` (fork `aditya-dl/onnxruntime-genai`) — GPU-KV squashed onto merged EP support |
| PR target | [`apwojcik/onnxruntime-genai:rel-0.13.2-mgx`](https://github.com/apwojcik/onnxruntime-genai/tree/rel-0.13.2-mgx) |
| New files | `src/migraphx/interface.{cpp,h}` |
| GPU-KV files touched | `generators.cpp`, `migraphx/session_options.cpp`, `models/kv_cache.cpp`, `models/model.cpp`, `smartptrs.h` |
| PR status | internal-fork PR pending; public upstream PR held |

---

## 7. How it relates to other tracks

- **MIGraphX EP / co-resident program cache / weight-sharing / parallel-finalize** live in
  the `onnxruntime-execution-providers` and `AMDMIGraphX` repos. This OGA change is the
  runtime-loop side (KV cache residency) and is independent of those compiler/EP-side
  changes; they compose but ship separately.
- **Public upstream sequencing.** The public OGA PR is intentionally held until the
  in-flight MIGraphX PR clears review, to avoid stacking EP PRs upstream.
