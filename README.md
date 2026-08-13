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
| `llama.cpp` + SYCL (Intel oneAPI) | **Crashes.** `UR_RESULT_ERROR_OUT_OF_RESOURCES` on the first tensor copy, even with a fully current oneAPI + GPU driver stack. Level-Zero doesn't register as a usable device at all; falls back to OpenCL, which then crashes. Matches a live upstream report (`ggml-org/llama.cpp#13775`) of iGPU+dGPU-together confusing SYCL enumeration. Four distinct fix attempts tried (env vars, forced device indices, full card-level device access instead of render-node-only, confirming it's not concurrent-GPU contention) — all failed, the last two converging on a hard segfault in Level-Zero's own device enumeration with zero other GPU users active. This reads as a genuine driver/kernel bug on this host, not a config problem. |
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

We went ahead and did that dedicated setup work: used OpenVINO's real Python API
(`model.reshape()`) to force the model's external inputs from dynamic to static
`[1, 512]` — confirmed working. Reloading onto the NPU hit the **exact same error at the
exact same internal line**, meaning the actual problem is an internal reshape deep
inside a self-attention layer that stays dynamic regardless of the external signature —
not something a post-hoc reshape fixes. No export-time static-shape flag exists in
`optimum-cli` either. Three real, distinct attempts, same wall each time — this looks
like a genuine gap in how this model's attention traces for the NPU compiler
specifically, not a config issue.

### Throughput isn't the whole story

Everything below measures raw tokens/sec. That tells you whether a model is *fast
enough* — it says nothing about whether it's actually *good* at tool-calling, multi-step
instruction following, or reliable structured output, which is what matters if you're
trying to run agentic/coding-assistant workloads locally rather than just chat. A model
that's fast but unreliable at tool use isn't actually usable for that purpose, and this
benchmark doesn't test for it.

Real (not vibes-based) signal from what's already in this matrix: **Qwen 3.6**'s own
release material is literally titled "Towards Real World Agents," explicitly naming
repository-level code comprehension and multi-step problem solving as design targets —
not a general chat model retrofitted for agent use. **Devstral** (Mistral) was likewise
originally built specifically for agentic coding workflows. The others here — Gemma 4,
Phi-4-mini, DeepSeek-R1-Distill — don't carry that same explicit design signal;
R1-distill models in particular are known for reasoning quality but have historically
been less reliable at strict tool-call-format compliance, since their distillation
target was reasoning traces, not tool-use output.

If you're picking a model for agentic use from this data, treat throughput as a filter
(is it fast enough to be usable at all) and treat agentic design intent as the actual
tie-breaker among what clears that bar — not the tok/s ranking by itself.

## Results

![Local LLM tokens/sec across CPU/iGPU/A770/NPU](results.png)

| Model | Params (total/active) | Device | Prompt proc. (tok/s) | Generation (tok/s) |
|---|---|---|---|---|
| phi-4-mini | 3.8B | Arc A770 | 1746.4 | **59.5** |
| deepseek-r1-distill-qwen-7b | 7.0B | Arc A770 | 929.4 | 49.8 |
| qwen2.5-coder-7b | 7.0B | Arc A770 | 925.4 | 49.8 |
| gemma-4-e4b | 7.5B | Arc A770 | 796.4 | 46.9 |
| gemma-4-12b | 12.0B | Arc A770 | 342.6 | 25.9 |
| devstral-2-24b | 24.0B | Arc A770 | 307.2 | 17.6 |
| mistral-small-3.2-24b | 24.0B | Arc A770 | 306.2 | 17.7 |
| qwen3.6-27b | 27.0B | Arc A770 | 205.5 | 12.9 |
| **gemma-4-31b** | **31.0B** | **Arc A770** | — | **FAILED — out of VRAM** |
| gemma-4-26b-a4b (MoE) | 25.2B / 3.8B active | CPU-offload | 220.9 | 6.6 |
| qwen3.6-35b-a3b (MoE) | 35.0B / 3B active | CPU-offload | 152.7 | 6.1 |
| phi-4-mini | 3.8B | iGPU | 218.7 | 10.5 |
| gemma-4-e4b | 7.5B | iGPU | 138.7 | 10.6 |

The `gemma-4-31b` failure is real data, not an error to fix: at Q4_K_M it needs roughly
17GB, just over the A770's 16GB VRAM — this is the actual ceiling for a single dense
model on this card, found empirically rather than estimated. The two 24B dense models
(Devstral, Mistral Small) landing within 0.1 tok/s of each other despite being from
different labs is a good sanity check on the methodology.

MoE models (CPU-offload, active params in system RAM alongside the GPU) trade raw speed
for fitting models that wouldn't otherwise fit in 16GB VRAM at all — both land around
6 tok/s regardless of total size, since active-parameter count (not total) dominates
memory-bandwidth-bound decode speed once experts are being streamed from system RAM.

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
