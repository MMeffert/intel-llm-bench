#!/bin/bash
# services/llm-bench/build-vulkan.sh
# Backend candidate A (control arm) — llama.cpp with the generic/portable
# Vulkan GPU backend. This is the pattern already proven on this exact
# hardware in April 2026 (VMID 181, ~35-39 tok/s on Qwen2.5-VL 7B Q8_0) —
# reused verbatim as the baseline the SYCL/IPEX-LLM candidates get measured
# against, not assumed to be the winner.
set -euo pipefail

LLAMA_DIR="/opt/llama.cpp-vulkan"

echo "=== Vulkan: install build + driver deps ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  build-essential cmake git pkg-config ca-certificates curl \
  libvulkan-dev mesa-vulkan-drivers vulkan-tools \
  glslc glslang-tools glslang-dev spirv-headers \
  libcurl4-openssl-dev python3-minimal

echo "=== Vulkan: clone llama.cpp ==="
if [[ -d "${LLAMA_DIR}/.git" ]]; then
  git -C "${LLAMA_DIR}" fetch --quiet
  git -C "${LLAMA_DIR}" pull --quiet --ff-only
else
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "${LLAMA_DIR}"
fi

echo "=== Vulkan: build llama-bench ==="
cmake -S "${LLAMA_DIR}" -B "${LLAMA_DIR}/build" \
  -DGGML_VULKAN=ON \
  -DLLAMA_CURL=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "${LLAMA_DIR}/build" --target llama-bench llama-quantize -j "$(nproc)"

echo "=== Vulkan: verify ==="
"${LLAMA_DIR}/build/bin/llama-bench" --help | head -20
echo "--- Vulkan devices ---"
vulkaninfo --summary 2>&1 | grep -A3 "GPU id" || vulkaninfo --summary 2>&1 | head -40
