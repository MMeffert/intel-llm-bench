#!/bin/bash
# services/llm-bench/build-sycl.sh
# Backend candidate B — llama.cpp with the SYCL backend (Intel oneAPI),
# the community-recommended path for Intel GPUs specifically (verified via
# fresh research 2026-08-13, not assumed from the April Vulkan precedent).
#
# Package/repo details below are Intel's current official APT instructions
# (intel.com/.../install-oneapi-toolkit-with-apt.html, checked 2026-08-13) —
# re-verify against that page if this script starts failing, package names
# on a fast-moving toolkit like this can move.
set -euo pipefail

LLAMA_DIR="/opt/llama.cpp-sycl"

echo "=== SYCL: add Intel oneAPI APT repo ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq gpg-agent wget ca-certificates

wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | tee /etc/apt/sources.list.d/oneAPI.list

echo "=== SYCL: install oneAPI Base Toolkit (this is several GB, expect it to take a while) ==="
apt-get update -qq
apt-get install -y -qq intel-basekit

echo "=== SYCL: install the Intel GPU compute-runtime (Level Zero + OpenCL driver for Arc) ==="
# NOT in Debian 13's default apt repos (confirmed live 2026-08-13 — apt-get
# reported all four packages "Unable to locate"). Pull real, current .deb
# releases directly instead of a package name that doesn't exist here:
#   driver:    https://github.com/intel/compute-runtime/releases/latest
#   compiler:  https://github.com/intel/intel-graphics-compiler/releases/latest
mkdir -p /tmp/intel-gpu-driver && cd /tmp/intel-gpu-driver

CR_URL=$(curl -sf https://api.github.com/repos/intel/compute-runtime/releases/latest \
  | python3 -c "import json,sys; d=json.load(sys.stdin); [print(a['browser_download_url']) for a in d['assets'] if a['name'].endswith('.deb') and 'dbgsym' not in a['name']]")
IGC_URL=$(curl -sf https://api.github.com/repos/intel/intel-graphics-compiler/releases/latest \
  | python3 -c "import json,sys; d=json.load(sys.stdin); [print(a['browser_download_url']) for a in d['assets'] if a['name'].endswith('.deb') and 'devel' not in a['name']]")

for url in ${CR_URL} ${IGC_URL}; do
  curl -L --fail -O "${url}"
done
ls -la /tmp/intel-gpu-driver/

apt-get install -y -qq ./*.deb
cd /
rm -rf /tmp/intel-gpu-driver

echo "=== SYCL: install build deps ==="
apt-get install -y -qq --no-install-recommends build-essential cmake git pkg-config libcurl4-openssl-dev

echo "=== SYCL: verify the toolkit + GPU are visible BEFORE building anything ==="
# setvars.sh isn't written to be `set -u` safe (references OCL_ICD_FILENAMES
# without a default, which is a real, previously-undiagnosed vendor-script
# gotcha, not a toolkit problem) — relax nounset only around the source call.
set +u
source /opt/intel/oneapi/setvars.sh
set -u
sycl-ls
echo "^^^ MUST show at least one [ext_oneapi_level_zero:gpu] or [opencl:gpu] Arc device above."
echo "    If it doesn't, STOP HERE — fix the driver/toolkit install before building."

echo "=== SYCL: clone llama.cpp ==="
if [[ -d "${LLAMA_DIR}/.git" ]]; then
  git -C "${LLAMA_DIR}" fetch --quiet
  git -C "${LLAMA_DIR}" pull --quiet --ff-only
else
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "${LLAMA_DIR}"
fi

echo "=== SYCL: build llama-bench ==="
cmake -S "${LLAMA_DIR}" -B "${LLAMA_DIR}/build" \
  -DGGML_SYCL=ON \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DLLAMA_CURL=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "${LLAMA_DIR}/build" --target llama-bench -j "$(nproc)"

echo "=== SYCL: verify ==="
"${LLAMA_DIR}/build/bin/llama-bench" --help | head -20
