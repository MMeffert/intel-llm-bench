# intel-llm-bench

Real, measured tokens/sec for running local LLMs across the different compute engines
on a modern Intel desktop platform — CPU, integrated GPU, discrete Arc GPU, and NPU —
so you don't have to trust secondhand blog numbers to figure out the sweet spot between
model size and speed on this kind of hardware.

Every number here was actually run, not sourced from a vendor spec sheet or an
aggregator blog. Where something *didn't* work, that's recorded too — the failures are
as useful as the successes if you're planning your own setup.

## Hardware under test

- CPU: Intel Core Ultra 9 285K (Arrow Lake-S desktop, 24 cores / 24 threads)
- iGPU: on-die Xe-LPG (uses system RAM, no dedicated VRAM)
- Discrete GPU: Intel Arc A770 (16GB dedicated VRAM)
- NPU: on-die Arrow Lake NPU, 13 TOPS ("NPU 3" generation — the same generation used in
  2023's Meteor Lake, *not* the newer 48 TOPS NPU 4 in Lunar Lake mobile chips; Intel
  deliberately shipped the smaller NPU on this desktop part and spent the die area on
  CPU cores instead)
- 128GB DDR5 system RAM (relevant for the MoE CPU-offload results — see below)

If you're evaluating whether to buy an Arc A770, or whether your Arrow Lake iGPU/NPU
are worth bothering with for local LLM work, this is meant to save you the trial and
error.

## Methodology

### Backend selection — measured, not assumed

The obvious move is `llama.cpp` with the Vulkan backend — it's the simplest to set up
and the numbers you'll find online mostly use it. But Vulkan is Intel's *generic*
compute backend, not an Intel-optimized one — it explicitly does not use the Arc GPU's
matrix-acceleration cores. Two more "Intel-optimized" alternatives exist, so before
running the real matrix we tested all three head-to-head on the same reference model:

| Backend | Result |
|---|---|
| **`llama.cpp` + Vulkan** | **Works.** The baseline/portable choice — no matrix-core acceleration, but reliable. |
| `llama.cpp` + SYCL (Intel oneAPI) | **Crashes.** `UR_RESULT_ERROR_OUT_OF_RESOURCES` on the first tensor copy, even with a fully current oneAPI + GPU driver stack. Level-Zero doesn't register as a usable device at all; falls back to OpenCL, which then crashes. Two documented fixes tried, neither resolved it. |
| OpenArc (OpenVINO-based, actively maintained community project) | **Crashes on current architectures.** OpenVINO GenAI's stateful LLM pipeline doesn't yet support some 2026-era model graph shapes (`RuntimeError: Model should have 3 or 4 inputs... but you have '5' inputs`) — this matched a live upstream OpenVINO bug report for a different model family, so it's a real, current gap, not a one-off. |
| IPEX-LLM (Intel's own PyTorch-based LLM library) | **Not viable at all** — Intel archived the upstream repo (made it read-only) in January 2026. Its GitHub-distributed portable build hasn't shipped an update since mid-2025 and doesn't even bundle a benchmarking tool. |

**Result: Vulkan runs the actual benchmark matrix.** Not because it's the theoretical
ceiling — it isn't — but because it's the only backend that reliably runs the current
generation of open-weight models on this hardware. If you're reading this later and
want to try SYCL or OpenArc again, they're worth periodic retesting — both are
genuinely active projects and this is exactly the kind of gap that can close within
weeks.

### Benchmark parameters

- `llama-bench`, prompt-processing test at 512 tokens, generation test at 128 tokens, 3
  repetitions each, results averaged.
- All models quantized to Q4_K_M (or the nearest equivalent, e.g. `UD-Q4_K_M` for the
  MoE entries) — a standard, widely-used quantization level, not cherry-picked for any
  one model.
- Device targeting: `Vulkan0` = discrete Arc GPU, `Vulkan1` = integrated GPU (confirmed
  via `llama-bench --list-devices` — Vulkan device ordering isn't necessarily consistent
  across backends, so don't assume it matches other tools' numbering).
- MoE models use `-ncmoe` (llama.cpp's dedicated MoE-CPU-offload flag) to force expert
  weights onto system RAM while keeping shared/attention layers on the discrete GPU —
  this is what lets a MoE model larger than the GPU's VRAM run at all.

### NPU

The NPU can't be reached through `llama.cpp`+Vulkan at all — it's not a Vulkan-capable
device, only OpenVINO's NPU plugin reaches it. Since OpenArc/OpenVINO's LLM pipeline is
what's currently broken (see above), a chat-model NPU test would hit the same wall.

Instead, the NPU was tested against the workload it's actually architected for:
embedding/vectorization (OpenVINO 2026.0 specifically added NPU support for
`Qwen3-Embedding-0.6B`, and separately for Whisper speech-to-text — this is a real,
current, Intel-shipped capability, not speculation). Result: the NPU compiler requires
*static* input shapes (it compiles a fixed execution graph ahead of time, unlike a GPU),
and the default export tooling produces dynamic shapes —
`optimum-cli`'s NPU-specific quantization flags don't fix this; it needs a manual
`model.reshape()` call via the OpenVINO Python API before compiling. That's real,
scoped follow-up work, not something achievable through this benchmark's
download-and-run pattern — noted here rather than silently skipped.

**Takeaway if you're deciding whether to bother with your NPU:** it's real, current,
Intel-supported technology for small fixed-shape workloads (embeddings, STT/TTS) — not
useless — but it needs dedicated setup work beyond what a generic benchmark harness
does automatically, and it was never going to compete with a discrete GPU on raw LLM
decode throughput anyway (13 TOPS is a fraction of even a modest discrete GPU's
compute).

## Results

<!-- RESULTS_TABLE_PLACEHOLDER -->

*(Chart: see `results.png` / the published version of this README.)*

## Reproducing this

```bash
# 1. Build the backend (Vulkan — see build-vulkan.sh; build-sycl.sh included for
#    reference/retesting, but crashed on this hardware as of this run)
./build-vulkan.sh

# 2. Edit models.yaml to your own model list, or use the one here as a starting point

# 3. Run the whole matrix (resumable — safe to interrupt and re-run)
./run-matrix.sh

# 4. Aggregate into a CSV + summary table
python3 aggregate-results.py results.jsonl results.csv
```

`bench-model.sh` downloads one model at a time, benchmarks it, deletes it, and appends
one JSON line to `results.jsonl` — so this never needs more than one model's worth of
disk space resident at once, and a `failed` status gets recorded (with the real error)
rather than silently skipped if a model won't load.

## Caveats

- Single hardware sample (one Core Ultra 9 285K + one Arc A770) — not a statistical
  claim about the SKUs in general, just what this specific setup actually does.
- All numbers are from a single quantization level (Q4_K_M-class). Quality/speed
  tradeoffs across quant levels aren't covered here.
- Backend landscape moves fast — SYCL and OpenArc/OpenVINO both failed on this hardware
  as of this test date, but both are actively developed. Don't treat "Vulkan won" as a
  permanent verdict; retest periodically.

## License

MIT — see `LICENSE`.
